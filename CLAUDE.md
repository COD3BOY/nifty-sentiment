# CLAUDE.md — NIFTY Options Trading System

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variables (in .env or .streamlit/secrets.toml)
ANTHROPIC_API_KEY=...          # Claude API — required for news sentiment + trade critiques
KITE_API_KEY=...               # Zerodha Kite — required for live option chain + candles
KITE_ACCESS_TOKEN=...          # Kite session token (regenerate daily via utils/kite_auth.py)

# Run the dashboard
streamlit run app.py
```

yfinance-based sources (VIX, crude oil, global markets) work without API keys — good for testing.

## Directory Layout

```
nifty/
├── app.py                        # Streamlit entry point — tabs for paper trading, options desk, comparison
├── config.yaml                   # All config: source weights, algo params, paper trading, execution costs
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # This file
│
├── core/                         # Business logic (no Streamlit imports)
│   ├── config.py                 # YAML + .env loader, cached singleton
│   ├── engine.py                 # SentimentEngine — asyncio.gather parallel source fetching
│   ├── models.py                 # Pydantic: SentimentLevel, SentimentScore, AggregatedSentiment
│   ├── database.py               # SQLAlchemy + SQLite (sentiment, IV history, trades)
│   ├── market_hours.py           # NSE hours: Mon-Fri 9:15-15:30 IST
│   ├── event_calendar.py         # High-impact event calendar (blocks/boosts strategies)
│   ├── indicators.py             # Pure functions: VWAP, EMA, RSI, Bollinger Bands, Supertrend
│   ├── intraday_fetcher.py       # Kite primary, yfinance fallback for OHLCV candles
│   ├── nse_fetcher.py            # Kite option chain fetcher + BS IV for all strikes
│   ├── options_models.py         # StrategyName, StrikeData, OptionChainData, TradeSuggestion, etc.
│   ├── options_analytics.py      # PCR ratio, chain aggregation, liquidity scoring
│   ├── options_engine.py         # OptionsDeskEngine — orchestrates candle→indicators→analytics→suggestions
│   ├── iv_calculator.py          # Newton-Raphson + bisection BS IV solver (pure Python)
│   ├── iv_history.py             # IV percentile over 252-day lookback (SQLite)
│   ├── greeks.py                 # BS delta, probability-of-profit (POP)
│   ├── vol_distribution.py       # RV/VoV/VRP engine — 3yr yfinance data, CSV cache + 5-min TTL
│   ├── options_utils.py          # Shared helpers: get_strike_data, compute_spread_width, parse_dte, etc.
│   ├── trade_strategies.py       # 11 strategy evaluators (V1 logic)
│   ├── strategy_rules.py         # Machine-readable strategy rules for tuning/critique
│   ├── paper_trading_engine.py   # Pure paper trading: open/close/update positions, P&L, margin
│   ├── paper_trading_models.py   # PaperPosition, TradeRecord, PaperTradingState
│   ├── trade_context.py          # Market context snapshots at entry/exit
│   ├── kite_margins.py           # Kite SPAN margin API wrapper
│   ├── margin_cache.py           # JSON disk cache for margin-per-lot (<24h TTL)
│   └── criticizer_models.py      # ParameterRecommendation, TradeCritique Pydantic models
│
├── algorithms/                   # Pluggable trading algorithms (V1-V4)
│   ├── __init__.py               # @register_algorithm + discover_algorithms()
│   ├── base.py                   # TradingAlgorithm ABC: generate_suggestions(), evaluate_and_manage()
│   ├── sentinel.py               # V1: thin wrapper around V1 trade_strategies.py
│   ├── jarvis.py                 # V2: risk-first, 10 rule sections, 1% risk/trade
│   ├── optimus.py                # V3: capital preservation, 8% max annual drawdown
│   └── atlas.py                  # V4: all thresholds from vol distribution percentiles
│
├── sources/                      # Sentiment data source plugins
│   ├── __init__.py               # @register_source + discover_sources()
│   ├── base.py                   # DataSource ABC: name, fetch()
│   ├── vix.py                    # India VIX (yfinance, no API key)
│   ├── crude_oil.py              # Brent crude + USD/INR (yfinance)
│   ├── global_markets.py         # S&P 500, Nikkei, Hang Seng, FTSE (yfinance)
│   ├── gift_nifty.py             # GIFT Nifty pre-market gap (Kite)
│   ├── fii_dii.py                # FII/DII flows (nsepython)
│   ├── zerodha_pulse.py          # News sentiment (RSS + Claude, ~₹0.01/fetch)
│   ├── youtube.py                # YouTube transcript sentiment (disabled by default)
│   └── reddit.py                 # r/IndianStreetBets (disabled by default)
│
├── ui/                           # Streamlit components
│   ├── paper_trading_tab.py      # Per-algorithm paper trading UI
│   ├── options_desk_tab.py       # Intraday options desk + signals
│   └── algorithm_comparison.py   # Side-by-side algo performance
│
├── analyzers/                    # Claude-powered analysis
│   ├── claude_analyzer.py        # Sentiment structured output
│   ├── trade_criticizer.py       # Trade critique + param recommendations
│   └── eod_report.py             # End-of-day report generator
│
├── scripts/
│   └── tune_vol_params.py        # V4 Atlas backtest param tuner
│
└── data/                         # Runtime data (gitignored)
    ├── nifty_sentiment.db        # SQLite database
    ├── vol_distribution_cache.csv
    ├── margin_cache.json
    └── paper_trading_state_{algo}.json  # Per-algorithm state
```

## Key Design Patterns

### Plugin Registry
Both sources and algorithms use a decorator-based registry pattern:
- **Sources**: `@register_source` in `sources/__init__.py`, keyed by `cls.name.fget(None)` (works because `name` properties return string literals)
- **Algorithms**: `@register_algorithm` in `algorithms/__init__.py`, keyed by `cls.name` (class attribute, not property)

### Pure Function Architecture
Algorithm methods are pure: `state_in → state_out`. No side effects except logging.
- `generate_suggestions(chain, technicals, analytics) → list[TradeSuggestion]`
- `evaluate_and_manage(state, suggestions, chain, ...) → PaperTradingState`

### Two-Phase Evaluation Loop
`evaluate_and_manage()` runs in two phases every refresh cycle:
1. **Phase 1**: Manage existing positions — update LTPs, check SL/TP/delta exits, close if triggered
2. **Phase 2**: Open new positions — check global gates, portfolio structure, sizing, then open max 1 trade per cycle

### Shared Helpers (`core/options_utils.py`)
Common functions extracted from algorithms to avoid duplication:
- `get_strike_data()`, `compute_spread_width()`, `compute_bb_width_pct()`
- `parse_dte()`, `is_breakout()`, `reset_period_tracking()`, `compute_expected_move()`
- `_now_ist()` lives in `core/paper_trading_models.py`

## Common Pitfalls

- **NIFTY lot size is 65** — Do NOT change this without verifying with NSE. It was 75 historically but is currently 65. All algorithms should default to 65.
- **Anthropic SDK**: Use `output_config`, NOT `output_schema` for structured outputs
- **numpy JSON**: `bool_`, `float64`, `int64` are not JSON-serializable — use custom `json.JSONEncoder`
- **Pydantic serialization**: Use `model_dump(mode="json")` when result will be JSON-serialized
- **yfinance MultiIndex**: Check `isinstance(df.columns, pd.MultiIndex)` and use `.get_level_values(0)`
- **Streamlit widget keys**: Must be suffixed with `algo_name` to avoid duplicate key errors across tabs
- **Algorithm `name`**: Class attribute (str), NOT a property — unlike sources which use `@property`

## Core Change Policy

**Every core change MUST be logged in the Change Log below with a reason.** Do not change thresholds, lot sizes, risk parameters, formulas, or algorithm logic without documenting why the change was made and what data supports it. This prevents back-and-forth changes without supporting rationale.

## Git Workflow

- **Commit after each meaningful unit of work** — don't batch everything into one giant commit. Each feature, fix, or logical group of changes should be its own commit.
- **Do NOT push until explicitly asked** — keep commits local. Only push to remote when the user says to push.
- **Working docs stay local** — `docs/algorithms/`, `docs/architecture.md`, `docs/data-sources.md`, `docs/options-math.md` are gitignored. Only `docs/production-readiness.md` is tracked.

## Config Reference

All config lives in `config.yaml`:
- `sources.*` — per-source: enabled, weight, timeout, source-specific params
- `engine.*` — default timeout, NIFTY ticker, sentiment level thresholds
- `options_desk.*` — symbol, candle period/interval, indicator params, IV calculator, signal thresholds
- `paper_trading.*` — initial capital, lot size, auto-execute, SL/PT, execution costs, margin estimates
- `algorithms.sentinel` — V1 (no extra params, delegates to paper_trading config)
- `algorithms.jarvis` — V2: risk limits, credit/debit rules, liquidity, portfolio defense
- `algorithms.optimus` — V3: drawdown limits, allocation, IC/strangle/debit construction, VIX bands
- `algorithms.atlas` — V4: regime thresholds, 9 dynamic function coefficients, hard gates

## Data Flow

```
                     ┌─────────────────────────────────┐
                     │     Sentiment Pipeline           │
                     │                                  │
  Sources ──────────►│  engine.py (asyncio.gather)      │──► SQLite
  (vix, crude,       │  ↓                               │
   global, gift,     │  claude_analyzer.py               │
   fii_dii, pulse)   │  (structured output)             │
                     └─────────────────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────────┐
                     │     Options Desk Pipeline        │
                     │                                  │
  Kite/yfinance ────►│  intraday_fetcher → indicators   │
  option chain ─────►│  nse_fetcher → iv_calculator     │
                     │  → options_analytics              │
                     │  → trade_strategies (V1)         │
                     └──────────────┬──────────────────┘
                                    │ chain, technicals, analytics
                                    ▼
                     ┌─────────────────────────────────┐
                     │     Paper Trading (per-algo)     │
                     │                                  │
                     │  algo.generate_suggestions()     │
                     │  algo.evaluate_and_manage()      │
                     │  → state saved to JSON           │
                     │  → trade_criticizer (Claude)     │
                     └─────────────────────────────────┘
```

## Known Issues

1. ~~**`sessions_held` never incremented**~~ — **FIXED**: Now increments once per new trading day via `last_session_date` tracking.
2. **IV percentile 50.0 fallback** — `iv_history.py` returns 50.0 when no data exists or IV ≤ 0. Now logged with `logger.warning()` but still masks missing data.
3. ~~**No NSE holiday calendar**~~ — **FIXED**: 34 NSE holidays for 2025-2026 in `market_hours.py`.
4. **Neutral vol fallback (V4)** — When `get_today_vol_snapshot()` fails, Atlas uses all percentiles = 0.5. This produces valid but meaningless thresholds that mask data quality problems.
5. ~~**No chain validation**~~ — **FIXED**: `core/validation.py` has `validate_option_chain()` with 5+ checks.
6. ~~**No startup health checks**~~ — **FIXED**: `core/health.py` runs 5 checks at startup; System Health tab in dashboard.

See `docs/production-readiness.md` for the full gap analysis and prioritized checklist.

## Change Log

| Date | File(s) | Change | Why |
|------|---------|--------|-----|
| 2026-02-11 | `core/options_utils.py` (new) | Extracted 7 duplicated helper functions from jarvis/optimus/atlas | Eliminate ~160 lines of copy-pasted code across 3 algorithm files |
| 2026-02-11 | `algorithms/jarvis.py` | Replaced 6x hardcoded `lot_size = 65` with `cfg.get("lot_size", 65)` | Make lot_size configurable via config.yaml instead of buried in code; default remains 65 |
| 2026-02-11 | `algorithms/jarvis.py`, `optimus.py`, `atlas.py` | Added `is_market_open()` guard to `generate_suggestions()` | Previously only `evaluate_and_manage()` checked market hours; suggestions could be generated outside trading hours |
| 2026-02-11 | `algorithms/jarvis.py`, `optimus.py`, `atlas.py` | Import `_now_ist` from `paper_trading_models` instead of redefining | Was copy-pasted in each file; single source of truth |
| 2026-02-11 | `core/iv_history.py` | Added `logger.warning()` before silent `return 50.0` fallbacks | Operators had no way to know when real IV data was missing |
| 2026-02-11 | `config.yaml` | Added `lot_size: 65` to `algorithms.jarvis` section | Jarvis was the only algo missing lot_size in config (optimus/atlas already had it) |
| 2026-02-11 | `config.yaml`, `optimus.py`, `atlas.py` | Changed lot_size from 75 to 65 in optimus and atlas config + code defaults | NIFTY lot size is 65 (was incorrectly set to 75) |
| 2026-02-11 | `core/validation.py` (new) | 11 data validators with ValidationResult | Prevent bad data from reaching algorithms; central validation module |
| 2026-02-11 | `core/error_types.py` (new) | Typed exception hierarchy (5 error types) | Replace bare `except Exception` with specific error handling |
| 2026-02-11 | `core/iv_calculator.py` | Return None instead of 0.0 for errors, add S/K<=0 guards, convergence checks | Callers couldn't distinguish "cannot compute" from "zero vol"; K=0 crash |
| 2026-02-11 | `core/greeks.py` | bs_delta returns None for bad inputs, T<=0 guard in compute_pop | Crash on T=0, wrong POP when delta=0 due to IV error |
| 2026-02-11 | `core/vol_distribution.py` | ffill(limit=5), inf filtering, stale data rejection, VIX coverage check | Unlimited ffill created lookback bias; stale data used silently |
| 2026-02-11 | `sources/vix.py`, `global_markets.py`, `crude_oil.py` | NaN checks before arithmetic | NaN from yfinance silently propagated to score calculations |
| 2026-02-11 | `core/iv_history.py` | IV bounds [0.5,200], 5-min deduplication | Let IV>200 through; duplicate rows on frequent refreshes |
| 2026-02-11 | `core/config_schema.py` (new) | Pydantic config validation with cross-checks | Invalid config values caused runtime errors |
| 2026-02-11 | `core/config.py` | Schema validation on first load | Validate config at startup |
| 2026-02-11 | `core/database.py` | WAL mode + busy timeout | SQLite concurrent access reliability |
| 2026-02-11 | `core/paper_trading_engine.py` | Backup rotation (3 .bak files), sessions_held increment, unrealized PnL in daily loss CB | State corruption risk; sessions_held stuck at 0; daily loss ignoring open positions |
| 2026-02-11 | `core/paper_trading_models.py` | Add last_session_date field, round unrealized_pnl and net_realized_pnl to 2 decimals | Track session counting; floating-point drift in PnL |
| 2026-02-11 | `core/market_hours.py` | NSE holiday calendar 2025-2026 (34 dates), next_trading_day(), is_nse_holiday() | Trades attempted on NSE holidays |
| 2026-02-11 | `core/rate_limiter.py` (new) | Token bucket rate limiter + retry with backoff | API rate limit protection for yfinance/Kite/Claude |
| 2026-02-11 | `core/circuit_breaker.py` (new) | Circuit breaker pattern with registry | Prevent cascading failures from repeated API errors |
| 2026-02-11 | `core/logging_config.py` (new) | Structured JSON logging with 3 rotating file handlers | No structured logging; no log rotation |
| 2026-02-11 | `core/trade_audit.py` (new) | Trade decision audit trail with ring buffer | No way to trace why trades were opened/closed/rejected |
| 2026-02-11 | `core/engine.py` | asyncio.gather with return_exceptions=True | One source exception crashed entire fetch pipeline |
| 2026-02-11 | `core/nse_fetcher.py` | Kite token error detection, is_kite_token_valid() health check | Token errors caught as generic Exception; no health check |
| 2026-02-11 | `app.py` | Add logging setup call at startup | Logs not structured or rotated |
| 2026-02-11 | `requirements.txt` | Pin all dependency versions | Unpinned versions could break on update |
| 2026-02-11 | `Dockerfile`, `docker-compose.yml`, `.dockerignore` (new) | Deployment artifacts | No containerized deployment option |
| 2026-02-11 | `tests/` (new, 8 files) | 319 tests covering IV, greeks, indicators, vol, validation, models, engine, atlas | Zero tests previously |
| 2026-02-11 | `docs/production-readiness.md` | Full rewrite with audit findings and remediation status | Previous version listed gaps without solutions |
| 2026-02-11 | `core/rate_limiter.py` | Add `acquire_sync()` with `threading.Lock` for sync callers | Existing `acquire()` was async-only; sync callers (nse_fetcher, vol_distribution, etc.) couldn't use it |
| 2026-02-11 | `core/api_guard.py` (new) | Combined rate limiter + circuit breaker guard functions | Thin wrappers that combine both mechanisms per API type, keeping call-site changes minimal |
| 2026-02-11 | `sources/vix.py`, `global_markets.py`, `crude_oil.py` | Wire yfinance rate limiter + circuit breaker guards | yfinance calls had no rate limiting or failure tracking |
| 2026-02-11 | `core/vol_distribution.py` | Wire yfinance guards around both `yf.download()` calls | yfinance calls had no rate limiting or failure tracking |
| 2026-02-11 | `core/engine.py` | Wire yfinance guard in `update_market_actuals()` | yfinance call had no rate limiting or failure tracking |
| 2026-02-11 | `core/intraday_fetcher.py` | Wire Kite + yfinance guards around API calls | Both Kite and yfinance calls had no rate limiting or failure tracking |
| 2026-02-11 | `core/nse_fetcher.py` | Wire Kite guards around `instruments()`, `quote()` calls | Kite calls had no rate limiting or failure tracking |
| 2026-02-11 | `core/kite_margins.py` | Wire Kite guards around `instruments()`, `basket_order_margins()`, `get_virtual_contract_note()` | Kite calls had no rate limiting or failure tracking |
| 2026-02-11 | `analyzers/claude_analyzer.py` | Wire Claude guards around both `messages.create()` calls | Claude API calls had no rate limiting or failure tracking |
| 2026-02-11 | `analyzers/trade_criticizer.py` | Wire Claude guard around `messages.create()` call | Claude API call had no rate limiting or failure tracking |
| 2026-02-11 | `core/health.py` (new) | 5 startup health checks (config, database, data dir, Kite creds, Anthropic key) | Dashboard started even if DB/config/API keys missing (Known Issue #6) |
| 2026-02-11 | `ui/system_health_tab.py` (new) | System Health monitoring tab: startup checks, circuit breakers, data freshness, audit trail | No system health visibility in dashboard |
| 2026-02-11 | `app.py` | Add startup health checks + System Health tab as first tab | No health monitoring; Known Issue #6 |
