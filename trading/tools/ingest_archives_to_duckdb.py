from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
import pandas as pd

DDL = """
CREATE TABLE IF NOT EXISTS ohlc_1s (
  timestamp TIMESTAMP,
  symbol VARCHAR,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  trades BIGINT,
  source_file VARCHAR
);
"""


def ingest_archives(
    *,
    archive_dir: Path | str = "logs/binance_stream/archive",
    db: Path | str = "logs/research/market.duckdb",
    symbol: str = "BTCUSDT",
    glob: str = "*.csv.gz",
    parquet_out: Path | str | None = "logs/research/ohlc_1s.parquet",
) -> int:
    archive_dir = Path(archive_dir)
    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    con.execute("CREATE TABLE IF NOT EXISTS ingest_log (source_file VARCHAR PRIMARY KEY);")

    files = sorted(archive_dir.glob(glob))
    ingested = {row[0] for row in con.execute("SELECT source_file FROM ingest_log").fetchall()}

    loaded = 0
    for chunk in files:
        chunk_str = str(chunk)
        if chunk_str in ingested:
            continue
        try:
            columns = [
                row[0]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_csv_auto(?, compression='gzip')",
                    [chunk_str],
                ).fetchall()
            ]
        except duckdb.Error:
            columns = []
        if "symbol" in columns:
            symbol_expr = "COALESCE(symbol, ?)"
        else:
            symbol_expr = "? AS symbol"
        trades_expr = "CAST(COALESCE(trades, 0) AS BIGINT)" if "trades" in columns else "0 AS trades"
        params = [symbol, chunk_str, chunk_str]

        con.execute(
            f"""
            INSERT INTO ohlc_1s
            SELECT
                CAST(timestamp AS TIMESTAMP) AS timestamp,
                {symbol_expr},
                CAST(open AS DOUBLE),
                CAST(high AS DOUBLE),
                CAST(low AS DOUBLE),
                CAST(close AS DOUBLE),
                CAST(volume AS DOUBLE),
                {trades_expr},
                ? AS source_file
            FROM read_csv_auto(?, compression='gzip')
            """,
            params,
        )
        con.execute("INSERT INTO ingest_log VALUES (?)", [chunk_str])
        loaded += 1

    if parquet_out:
        pq_path = Path(parquet_out)
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ohlc_1s TO '{pq_path}' (FORMAT PARQUET);")

    print(f"loaded_files={loaded} db={db_path}")
    return loaded


def ingest_dataframe(
    *,
    frame: pd.DataFrame,
    db: Path | str = "logs/research/market.duckdb",
    symbol: str = "BTCUSDT",
    source_file: Path | str | None = None,
) -> int:
    if frame.empty:
        return 0
    df = frame.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return 0
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "trades" not in df.columns:
        df["trades"] = 0

    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    con.execute("CREATE TABLE IF NOT EXISTS ingest_log (source_file VARCHAR PRIMARY KEY);")

    source_value = str(source_file) if source_file is not None else f"live:{symbol}"
    con.register("df_live", df)
    con.execute(
        """
        INSERT INTO ohlc_1s
        SELECT
            CAST(timestamp AS TIMESTAMP) AS timestamp,
            CAST(symbol AS VARCHAR) AS symbol,
            CAST(open AS DOUBLE),
            CAST(high AS DOUBLE),
            CAST(low AS DOUBLE),
            CAST(close AS DOUBLE),
            CAST(volume AS DOUBLE),
            CAST(trades AS BIGINT),
            ? AS source_file
        FROM df_live
        """,
        [source_value],
    )
    try:
        con.execute("INSERT INTO ingest_log VALUES (?)", [source_value])
    except duckdb.ConstraintException:
        pass
    return int(df.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default="logs/binance_stream/archive", help="Where archived .csv.gz chunks live")
    ap.add_argument("--db", default="logs/research/market.duckdb", help="DuckDB database path")
    ap.add_argument("--symbol", default="BTCUSDT", help="Fallback symbol for CSVs without one")
    ap.add_argument("--glob", default="*.csv.gz", help="Glob pattern for chunk files")
    ap.add_argument("--parquet-out", default="logs/research/ohlc_1s.parquet", help="Optional Parquet mirror path")
    args = ap.parse_args()

    ingest_archives(
        archive_dir=args.archive_dir,
        db=args.db,
        symbol=args.symbol,
        glob=args.glob,
        parquet_out=args.parquet_out,
    )


if __name__ == "__main__":
    main()
