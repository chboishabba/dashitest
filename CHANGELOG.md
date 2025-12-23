# Changelog

## Unreleased
- Recorded historical `run_all.py` behavior before the latest changes (single-threaded, no live view) and captured example results for the legacy run.
- Captured pre/post `run_all.py` outputs for reference:
  - Legacy (5 markets): total_pnl=362,634.4097 with per-market PnL: aapl.us 99,142.2711; btc.us 99,906.5616; btc_intraday -29,813.2553; msft.us 97,964.1213; spy.us 95,434.7109.
  - After added BTC sources (7 markets): total_pnl=1,314,977.2228 with per-market PnL: btc_yf 859,973.1025; btc.us 99,906.5616; aapl.us 99,142.2711; msft.us 97,964.1213; spy.us 95,434.7109; btc_intraday_1s 92,369.7107; btc_intraday -29,813.2553.
- Added Binance-backed BTC downloads (extended 1m window, ~10h 1s bars) and wired them into the downloader CLI; `run_trader` now prefers the richer BTC files.
- Extracted `run_trader.run_trading_loop` so the trading sim can be reused across markets without clobbering logs.
- Extended `run_all.py` to run every cached market, print a scoreboard, and optionally stream a live dashboard via `--live`.
- Added a project map to `README.md` describing the key scripts and components.
