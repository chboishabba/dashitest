# Changelog

## Unreleased
- Compression bench upgrades:
  - Replaced the lzma shim with a real range coder in `compression/rans.py`.
  - Added balanced-ternary digit planes, per-plane Z2 quotient (mag + gated sign), and contexted coding in `compression/video_bench.py`.
  - Added block reuse actions, reference stream, and masked-plane coding to approximate spatio-temporal quotient reuse.
  - Added optional motion-compensation pyramid search and train/test context split reporting.
  - Added color benchmarking with RGB and reversible YCoCg-R transforms, plus combined total bpp reporting.
  - Documented the triadic pipeline and quotient composition in `compression/triadic_pipeline.md`.
- Added severity-ranked bad-window tooling and docs:
  - `scripts/score_bad_windows.py` computes synthetic bad flags (|return| vs σ or drawdown slope) and ranks top-N windows by summed `p_bad` over a sliding window.
  - News fetch summaries now include severity (`sev_sum_p_bad`) and cap windows/days by triggers to avoid spam; 404s/future dates are treated as empty fetches.
  - `docs/bad_day.md` and `README.md` now describe regime windows (hazard view), tri-state `p_bad` posture policy, and “legal” BAN monetisation (structure/vol/fee avoidance, not forced shorting).
- Added `scripts/contextual_news.py` to aggregate contextual signals per date:
  - Reuters markets RSS (no key) filtered by keywords.
  - TradingEconomics macro calendar if `TE_API_KEY` is set; empty otherwise.
  - Stress proxies via yfinance (VIX, USDCNH, Copper) when available.
- Added `trading/training_dashboard_pg.py`: PyQtGraph-based live dashboard with price/actions/bad_flag shading, PnL+HOLD%, p_bad+bad_flag step fill, and volume panes for fast intraday visualization.
- Recorded historical `trading/run_all.py` behavior before the latest changes (single-threaded, no live view) and captured example results for the legacy run.
- Captured pre/post `trading/run_all.py` outputs for reference:
  - Legacy (5 markets): total_pnl=362,634.4097 with per-market PnL: aapl.us 99,142.2711; btc.us 99,906.5616; btc_intraday -29,813.2553; msft.us 97,964.1213; spy.us 95,434.7109.
  - After added BTC sources (7 markets): total_pnl=1,314,977.2228 with per-market PnL: btc_yf 859,973.1025; btc.us 99,906.5616; aapl.us 99,142.2711; msft.us 97,964.1213; spy.us 95,434.7109; btc_intraday_1s 92,369.7107; btc_intraday -29,813.2553.
- Added Binance-backed BTC downloads (extended 1m window, ~10h 1s bars) and wired them into the downloader CLI; `run_trader` now prefers the richer BTC files.
- Extracted `run_trader.run_trading_loop` so the trading sim can be reused across markets without clobbering logs.
- Extended `trading/run_all.py` to run every cached market, print a scoreboard, and optionally stream a live dashboard via `--live`.
- Added a project map to `README.md` describing the key scripts and components.
