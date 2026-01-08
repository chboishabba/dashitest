# External Integrations

**Analysis Date:** 2026-01-08

## APIs & External Services

**Market Data APIs:**
- **Binance REST API** - Public market data (klines, aggTrades)
  - Endpoint: `https://api.binance.com/api/v3/{klines,aggTrades}`
  - Integration: `trading/data_downloader.py` (lines 32-35)
  - Auth: None required (public endpoints)
  - Data: 1s/1m/daily candles, trade ticks
  - Rate limiting: Internal retry with exponential backoff

- **Stooq CSV Downloads** - Historical OHLCV data
  - Integration: `trading/data_downloader.py` (lines 97+)
  - Data storage: `data/raw/stooq/` (BTC primary: `btc_intraday.csv`, `btc_intraday_1s.csv`)
  - Politeness: 3s delays between requests

- **Yahoo Finance** - Optional yfinance library
  - Integration: `trading/data_downloader.py` (lines 20-23)
  - Cache: `data/cache/yfinance/` (writable, env-controlled)
  - Auth: None required
  - Status: Optional (gracefully degraded if missing)

- **CoinGecko** - Fallback market data source
  - Integration: `trading/data_downloader.py` references as fallback
  - Auth: None required

**News & Event APIs:**
- **NewsAPI** - News articles and events
  - Endpoint: `https://newsapi.org/v2/everything`
  - Integration: `trading/scripts/news_slice.py` (lines 27+)
  - Auth: API key in `NEWSAPI_KEY` environment variable
  - Fields: publishedAt, title, source.name, url, description
  - Status: Optional

- **GDELT Project** - Global event database
  - Endpoint: `https://api.gdeltproject.org/api/v2/events/summary`
  - Integration: `trading/scripts/news_slice.py` (lines 67+)
  - Auth: None required
  - Coverage: ~2015-06 onward
  - Outputs: Event counts + tone scores

- **RSS Feeds** - Custom news feeds
  - Integration: `trading/scripts/news_slice.py` (custom URLs via `--feed-url`)
  - Auth: None required

## Data Storage

**Databases:**
- None (file-based storage only)

**File Storage:**
- **CSV Logs** - Trading state persistence
  - Location: `logs/trading_log.csv`, `logs/trading_log_trades_*.csv`
  - Format: Per-step logs for dashboard consumption

- **NumPy Arrays** - Intermediate data
  - `dashilearn/sheet_energy.npy` - Band energy exports for Vulkan preview
  - Auto-timestamped benchmark outputs in `outputs/`

- **JSON** - Benchmark results
  - Pattern: `outputs/<benchmark>_<timestamp>/summary.json`

- **Images** - Visualizations
  - Pattern: Auto-timestamped with `%Y%m%dT%H%M%SZ` format
  - PNGs rolled into GIFs then deleted for storage efficiency

**Caching:**
- None (no Redis or memcached)

## Authentication & Identity

**Auth Provider:**
- None (no user authentication system)

**API Keys:**
- NewsAPI: `NEWSAPI_KEY` environment variable (optional)
- No other API keys required (public endpoints only)

## Monitoring & Observability

**Error Tracking:**
- None (local logging only)

**Analytics:**
- None

**Logs:**
- stdout/stderr only
- CSV logs for trading runs: `logs/trading_log*.csv`
- Benchmark logs: `outputs/` (timestamped)

## CI/CD & Deployment

**Hosting:**
- Local execution only (no cloud deployment)

**CI Pipeline:**
- None detected

## Environment Configuration

**Development:**
- Required env vars: None (all optional)
- Optional env vars:
  - `NEWSAPI_KEY` - Enable news slicing
  - `YF_CACHE_DIR` - Yahoo Finance cache location
  - `VK_ICD_FILENAMES` - Vulkan ICD selection for AMD GPU
- Secrets location: Environment variables

**Staging:**
- Not applicable (research platform)

**Production:**
- Same as development (local execution)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-01-08*
*Update when adding/removing external services*
