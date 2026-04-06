from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    rel_path: str


def _list_csvs(raw_root: pathlib.Path) -> list[str]:
    paths = sorted(p for p in raw_root.rglob("*.csv") if p.is_file())
    rels: list[str] = []
    for p in paths:
        try:
            rels.append(str(p.relative_to(raw_root)))
        except ValueError:
            rels.append(str(p))
    return rels


def _filter_paths(rels: list[str], include: list[str], exclude: list[str]) -> list[str]:
    include = [p for p in include if p]
    exclude = [p for p in exclude if p]
    inc_rx = [re.compile(p) for p in include]
    exc_rx = [re.compile(p) for p in exclude]
    out: list[str] = []
    for rel in rels:
        if exc_rx and any(rx.search(rel) for rx in exc_rx):
            continue
        if inc_rx and not any(rx.search(rel) for rx in inc_rx):
            continue
        out.append(rel)
    return out


def _run_one(
    *,
    python: str,
    trading_dir: pathlib.Path,
    raw_root: str,
    rel_path: str,
    log_prefix: str,
    extra_args: list[str],
) -> int:
    # Use run_trader's own --all + --all-include filtering so behavior stays identical.
    cmd = [
        python,
        "run_trader.py",
        "--all",
        "--raw-root",
        raw_root,
        "--all-include",
        rel_path,
        "--log-prefix",
        log_prefix,
        "--inter-run-sleep",
        "0",
        "--log-level",
        "quiet",
        "--no-geometry-plots",
        "--no-tower-log",
    ]
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=str(trading_dir), check=False)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run run_trader.py --all on many tapes concurrently by spawning subprocess workers."
    )
    ap.add_argument(
        "--raw-root",
        type=str,
        default="../data/raw",
        help="Raw data root (passed through to run_trader.py --raw-root).",
    )
    ap.add_argument(
        "--log-prefix",
        type=str,
        default="../logs/trading_log_parallel",
        help="Log prefix passed through to run_trader.py --log-prefix.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of concurrent worker processes.",
    )
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only run tapes whose relative path matches ANY of these regexes (repeatable).",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip tapes whose relative path matches ANY of these regexes (repeatable).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of tapes (debug). 0 disables.",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for workers (defaults to current interpreter).",
    )
    ap.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to run_trader.py (prefix with --).",
    )
    args = ap.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    trading_dir = pathlib.Path(__file__).resolve().parents[1]
    raw_root = pathlib.Path(args.raw_root).resolve()
    if not raw_root.exists():
        raise SystemExit(f"raw root does not exist: {raw_root}")

    rels = _list_csvs(raw_root)
    rels = _filter_paths(rels, include=args.include, exclude=args.exclude)
    if args.limit and args.limit > 0:
        rels = rels[: args.limit]
    if not rels:
        raise SystemExit("no tapes matched after include/exclude filtering")

    # argparse.REMAINDER keeps the leading "--" if present; drop it for subprocess.
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    jobs = [Job(rel_path=rel) for rel in rels]
    print(f"[parallel] tapes={len(jobs)} jobs={args.jobs} raw_root={raw_root}")

    failures: list[tuple[str, int]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut_map = {
            ex.submit(
                _run_one,
                python=args.python,
                trading_dir=trading_dir,
                raw_root=str(raw_root),
                rel_path=job.rel_path,
                log_prefix=args.log_prefix,
                extra_args=extra_args,
            ): job
            for job in jobs
        }
        for fut in concurrent.futures.as_completed(fut_map):
            job = fut_map[fut]
            try:
                code = fut.result()
            except BaseException as exc:
                failures.append((job.rel_path, 999))
                print(f"[fail] {job.rel_path}: {exc}")
                continue
            if code != 0:
                failures.append((job.rel_path, code))
                print(f"[fail] {job.rel_path}: exit={code}")
            else:
                print(f"[ok] {job.rel_path}")

    if failures:
        print("[summary] failures:")
        for rel, code in failures:
            print(f"- {rel}: exit={code}")
        return 1
    print("[summary] all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

