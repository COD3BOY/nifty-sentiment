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
│   ├── context_models.py         # Pydantic: SessionContext, DailyContext, WeeklyContext, VolContext, MarketContext
│   ├── context_engine.py         # ContextEngine — bootstrap daily/weekly from yfinance, session from 5-min candles
│   ├── options_utils.py          # Shared helpers: get_strike_data, compute_spread_width, parse_dte, etc.
│   ├── trade_strategies.py       # 11 strategy evaluators (V1 logic)
│   ├── strategy_rules.py         # Machine-readable strategy rules for tuning/critique
│   ├── paper_trading_engine.py   # Pure paper trading: open/close/update positions, P&L, margin
│   ├── paper_trading_models.py   # PaperPosition, TradeRecord, PaperTradingState
│   ├── trade_context.py          # Market context snapshots at entry/exit
│   ├── kite_margins.py           # Kite SPAN margin API wrapper
│   ├── margin_cache.py           # JSON disk cache for margin-per-lot (<24h TTL)
│   ├── criticizer_models.py      # ParameterRecommendation, TradeCritique Pydantic models
│   ├── improvement_models.py     # TradeClassification, SignalReliability, ImprovementProposal, ReviewSession
│   ├── improvement_ledger.py     # Change lifecycle: propose/approve/apply/revert + cooling/reversion monitoring
│   └── parameter_bounds.py       # Hard bounds for all 50 tunable param_keys + safety constants
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
│   ├── eod_report.py             # End-of-day report generator
│   └── daily_review.py           # Self-improvement: trade classification, signal reliability, calibration
│
├── scripts/
│   └── tune_vol_params.py        # V4 Atlas backtest param tuner
│
├── simulation/                   # Market simulation & stress testing
│   ├── __init__.py               # Package init
│   ├── __main__.py               # python -m simulation entry point
│   ├── clock.py                  # VirtualClock — time compression, patches _now_ist/is_market_open
│   ├── price_engine.py           # GBM + jumps + OU price path generator + VIX co-generation
│   ├── chain_synthesizer.py      # Synthetic option chain (IV smile, BS LTPs, OI, bid-ask)
│   ├── scenario_models.py        # Pydantic models for YAML scenario definitions
│   ├── data_assembler.py         # Bridges synthetic data → real indicators/analytics/observation
│   ├── adversarial.py            # SL hunting, false breakouts, whipsaw, liquidity vacuum, delta spike
│   ├── runner.py                 # Orchestrator — runs algos against synthetic data tick-by-tick
│   ├── results.py                # SimulationResult + self-improvement bridge (daily_review)
│   ├── cli.py                    # Click CLI: run/run-all/compare/review/list-scenarios
│   ├── sim_dashboard.py          # Streamlit visualization (run/results/compare)
│   └── scenarios/                # YAML scenario library (28 scenarios)
│       ├── normal.yaml           # Trending up/down, range-bound, slow grind, low-volume
│       ├── gaps.yaml             # Small/large/extreme gaps, gap-fill, gap-and-run
│       ├── crisis.yaml           # Flash crash, overnight crisis, multi-day meltdown, COVID cascade
│       ├── volatility.yaml       # VIX spike, compression, expiry-day crush, post-event relief
│       ├── patterns.yaml         # Whipsaw, V-recovery, false breakout, slow bleed
│       └── adversarial.yaml      # SL hunting, delta spikes, liquidity vacuum
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
7. **Streamlit `use_container_width` deprecation** — Streamlit removed `use_container_width` after 2025-12-31. Replace with `width='stretch'` (for `True`) or `width='content'` (for `False`) across `app.py` and `ui/` components.

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
| 2026-02-12 | `core/options_models.py` | Add `FetchMeta` model + `candle_meta`/`chain_meta` fields on `OptionsDeskSnapshot` | Track data source and fetch timing for health monitoring |
| 2026-02-12 | `core/intraday_fetcher.py` | Add `last_fetch_source` attribute to track Kite vs yfinance | Operators need to know which source served candle data |
| 2026-02-12 | `core/options_engine.py` | Populate `FetchMeta` after chain/candle fetches in `fetch_snapshot()` | Wire source metadata into snapshot for downstream consumption |
| 2026-02-12 | `ui/options_desk_tab.py` | Set `last_chain_fetch_ts`/`last_candle_fetch_ts` from snapshot metadata | Fix Data Freshness always showing "No data" for chain/candle (bug) |
| 2026-02-12 | `ui/system_health_tab.py` | Add Live Technical Data section with per-indicator traffic-light table | No per-indicator freshness monitoring; operators couldn't verify primary data source |
| 2026-02-12 | `core/paper_trading_models.py` | Add `trade_status_notes: list[str]` field to `PaperTradingState` | Carry per-cycle status notes from algorithm to UI |
| 2026-02-12 | `algorithms/atlas.py` | Populate `trade_status_notes` with regime, stand-down reasons, and Phase 2 blocking reasons | Operators had no visibility into why Atlas was idle |
| 2026-02-12 | `algorithms/optimus.py` | Populate `trade_status_notes` with VIX, IV percentile, credit regime blocks, and Phase 2 blocking reasons | Operators had no visibility into why Optimus was idle |
| 2026-02-12 | `algorithms/jarvis.py` | Populate `trade_status_notes` with VIX, daily loss, and Phase 2 blocking reasons | Operators had no visibility into why Jarvis was idle |
| 2026-02-12 | `ui/paper_trading_tab.py` | Add "Trade Status" expander with color-coded notes + Atlas dynamic params display | No dashboard visibility for trade rejection reasons |
| 2026-02-12 | `core/paper_trading_models.py`, `core/paper_trading_engine.py`, `ui/paper_trading_tab.py` | Change `strategy_type`, `status`, `exit_reason` fields from enum to `str` type; convert enum→`.value` at 3 assignment points | Fix PydanticSerializationUnexpectedValue warnings caused by `model_copy` bypassing `use_enum_values` validation |
| 2026-02-12 | `core/observation.py` (new) | Observation Period Engine: 6 Pydantic models (OpeningGap, OpeningRange, InitialTrend, VolumeProfile, VWAPContext, ObservationSnapshot) + pure computation functions for gap/range/trend/volume/VWAP/bias | System only passed point-in-time indicator values; no accumulated observation context for 9:15-10:00 window |
| 2026-02-12 | `core/options_models.py` | Add `observation: ObservationSnapshot \| None` field to `OptionsDeskSnapshot` | Carry observation data through snapshot pipeline to algorithms |
| 2026-02-12 | `core/options_engine.py` | Compute observation in `fetch_snapshot()` after candle fetch | Wire observation computation into existing data pipeline |
| 2026-02-12 | `core/options_utils.py` | Add `is_observation_period()` utility function | V2/V3/V4 need to check if current time is in observation window to block new entries |
| 2026-02-12 | `algorithms/base.py` | Add `observation=None` parameter to `generate_suggestions()` and `evaluate_and_manage()` ABC methods | Algorithms need observation context for informed trade decisions |
| 2026-02-12 | `algorithms/sentinel.py`, `core/paper_trading_engine.py` | Accept `observation` parameter (pass-through, no logic change) | ABC compliance; V1 already respects entry_start_time |
| 2026-02-12 | `algorithms/jarvis.py`, `optimus.py`, `atlas.py` | Add observation period entry guard + observation context in trade_status_notes | V2/V3/V4 didn't respect 10:00 AM entry_start_time; no observation-based reasoning in trade decisions |
| 2026-02-12 | `app.py`, `ui/paper_trading_tab.py` | Pass `observation` through call chain to algorithms | Wire observation data from snapshot to algorithm evaluate calls |
| 2026-02-12 | `ui/options_desk_tab.py` | Enhanced warmup view with live observation data + post-warmup "Morning Context" expander | Operators had no visibility into observation window data accumulation |
| 2026-02-12 | `config.yaml`, `core/config_schema.py` | Add `observation` config section with 7 params + `ObservationConfig` Pydantic validation | Observation thresholds need to be configurable and validated |
| 2026-02-12 | `tests/test_observation.py` (new) | 46 tests covering all observation models, computation functions, edge cases, and `is_observation_period()` | Comprehensive test coverage for new observation engine |
| 2026-02-12 | `core/paper_trading_engine.py` | `load_state()`: try/except + backup fallback (.bak → .bak1 → .bak2), skip empty files | Primary file corruption caused total data loss with no fallback |
| 2026-02-12 | `core/paper_trading_engine.py` | `save_state()`: empty-state guard (refuse to overwrite data with empty state) + backup rotation only on trade/position count change | Empty state saved on every render cycle overwrote all data + backups within 3 minutes |
| 2026-02-12 | `core/paper_trading_engine.py` | New `force_save_state()` for intentional resets (bypasses guard, still rotates backups) | Reset button needs to bypass empty-state guard |
| 2026-02-12 | `ui/paper_trading_tab.py` | `_get_state()` logging, two-step Reset confirmation with trade/position counts, `_force_set_state()` helper | One-click Reset with no confirmation wiped all data; no logging on state load |
| 2026-02-12 | `tests/test_paper_trading_engine.py` | 4 new tests: backup fallback, empty-overwrite guard, rotation-on-change-only, force-save bypass | Verify state persistence safety (369 total tests) |
| 2026-02-12 | `docs/self_improvement_protocol.md` (new) | Comprehensive methodology doc: trade classification matrix (A/B/C/D), signal reliability analysis, parameter calibration via MAE/MFE, safety rails, change lifecycle, daily review protocol, anti-patterns | No documented process for reviewing trades and making evidence-based parameter changes |
| 2026-02-12 | `core/parameter_bounds.py` (new) | Hard bounds for all 50 param_keys + safety constants (15% max step, 40% max drift, 15 min sample, 10 cooling trades, 5 loss reversion trigger, 3 max changes/session) + validation helpers | No guardrails on parameter changes — could drift arbitrarily |
| 2026-02-12 | `core/improvement_models.py` (new) | 5 Pydantic models: TradeClassification, SignalReliability, ParameterCalibration, ImprovementProposal, ReviewSession | Type-safe data structures for the self-improvement pipeline |
| 2026-02-12 | `core/database.py` | Add `ImprovementLedgerRow` and `ReviewSessionRow` tables + 6 new methods (save/update/get ledger, save/get review sessions) | Persistent tracking of parameter change lifecycle (proposed→applied→confirmed/reverted) |
| 2026-02-12 | `core/improvement_ledger.py` (new) | Business logic layer: propose/approve/reject/defer/apply/revert changes, cooling period checks, reversion trigger monitoring, review session persistence | Wraps raw DB ops with safety rail enforcement and lifecycle management |
| 2026-02-12 | `analyzers/daily_review.py` (new) | Analysis engine: classify_trade (A/B/C/D matrix), signal extraction from entry_context, compute_signal_reliability (rolling window), calibrate_parameters (MAE/MFE), check_safety_rails, run_daily_review orchestrator | No automated trade analysis or evidence-based parameter recommendation pipeline |
| 2026-02-12 | `core/trade_strategies.py` | Wire `get_active_overrides()` into `generate_trade_suggestions()` — overrides now loaded from database instead of hardcoded `{}` | Dead code: 50 param_keys had override plumbing (`_p()`) but `overrides={}` was always passed |
| 2026-02-12 | `config.yaml` | Add `self_improvement` section with 7 safety rail parameters | Safety rail thresholds need to be configurable |
| 2026-02-12 | `core/config_schema.py` | Add `SelfImprovementConfig` Pydantic model with cross-validation (drift >= step) | Validate self-improvement config at startup |
| 2026-02-12 | `tests/test_daily_review.py` (new) | 42 tests: trade classification (6), signal extraction (4), regime fit (5), signal reliability (4), parameter calibration (3), safety rails (5), parameter bounds (8), full review (4), models (3) | Test coverage for entire self-improvement system (411 total tests) |
| 2026-02-13 | `simulation/` (new, 11 files) | Market Simulation & Stress Testing System — VirtualClock (time patching), GBM+jumps+OU price engine, BS-consistent chain synthesizer, data assembler bridging synthetic→real indicators/analytics/observation, tick-by-tick runner with progressive candle reveal, adversarial perturbation engine (SL hunt, false breakout, whipsaw, liquidity vacuum, delta spike), SimulationResult with self-improvement bridge to daily_review, Click CLI (run/run-all/compare/review/list-scenarios), Streamlit dashboard (live view + post-sim results + algo comparison) | Algorithms only train against 6-hour live sessions; need synthetic adversarial scenarios to accelerate self-improvement pipeline |
| 2026-02-13 | `simulation/scenarios/` (6 YAML files, 28 scenarios) | normal (5), gaps (5), crisis (6 incl. multi-day), volatility (4), patterns (5), adversarial (3) | Comprehensive scenario library covering trending/range-bound/gaps/crashes/whipsaws/VIX spikes |
| 2026-02-13 | `tests/test_sim_*.py` (5 new files, 75 tests) | test_sim_clock (13), test_sim_price_engine (18), test_sim_chain (14), test_sim_assembler (10), test_sim_runner (15 incl. end-to-end integration) | Full test coverage for simulation system (486 total tests) |
| 2026-02-13 | `core/paper_trading_engine.py` | Lines 778, 843: use `refresh_ts` instead of `_time.time()` for cooldown/timestamp | In simulation, 375 ticks execute in seconds (wall clock), so cooldown never expired — only 1 trade per sim day. Now uses virtual time via `refresh_ts` with wall-clock fallback |
| 2026-02-13 | `ui/simulation_tab.py` | Add "Run All" sub-tab (batch all 4 algos x all scenarios), seed help tooltip | No batch execution — each algo x scenario had to be run manually; seed purpose was unexplained |
| 2026-02-13 | `ui/simulation_tab.py`, `ui/options_desk_tab.py` | Two-phase sim execution: button click sets `sim_running=True` + stores params + `st.rerun()`; next rerun skips `st_autorefresh` then executes sim | Options desk 60s autorefresh injected JS timer before sim started (tab render order); setting flag in same rerun was too late — timer already registered |
| 2026-02-14 | `core/trade_strategies.py` | Add IV penalty (-15) to Bull Call Spread, Bear Put Spread (threshold 20), Long CE, Long PE (threshold 18) | Debit strategies lacked IV-based scoring — entered buying expensive premium in high-IV regimes (all 9 Cat-D trades had IV=28) |
| 2026-02-14 | `core/trade_strategies.py` | Add BB width penalty (-15) to Bull Call Spread and Bear Put Spread (threshold 1.5%) | Bear Put Spread entered after breakout already happened (BB width 3.9%); no check on expanded Bollinger Bands |
| 2026-02-14 | `core/strategy_rules.py` | Add `iv_high_penalty_threshold` and `bb_width_expanded_pct` scoring rule entries to 4 debit strategies | Machine-readable rules for new penalties so criticizer and tuning system can reference them |
| 2026-02-14 | `core/parameter_bounds.py` | Add bounds: `iv_high_penalty_threshold` (12-30), `bb_width_expanded_pct` (0.8-3.0) | Safety bounds for new tunable param_keys |
| 2026-02-14 | `core/paper_trading_engine.py` | Add `max_trades_per_day` gate in Phase 2 (default 6) — counts open + today's closed trades | Sentinel fired 10+ trades in crisis scenarios; no daily trade count limit |
| 2026-02-14 | `config.yaml`, `core/config_schema.py` | Add `max_trades_per_day: 6` config + Pydantic validation | Configurable daily trade limit |
| 2026-02-14 | `core/paper_trading_engine.py`, `config.yaml`, `core/config_schema.py` | Add `debit_min_score_to_trade: 61` — blocks all V1 debit trades (max score=60) | 3-cycle sim analysis: debit strategies 0-14% WR across 2,381 trades while credit is 95-100% WR. Raising bar to 61 makes sentinel credit-only → 100% WR, +655K across 28 scenarios |
| 2026-02-14 | `core/trade_strategies.py`, `core/strategy_rules.py` | Lower IV penalty thresholds: all debit strategies 20/18→14 | Median entry IV=14.5%; threshold of 14 blocks debit entries at any elevated IV level |
| 2026-02-14 | `config.yaml` | Change `debit_max_hold_minutes: 120` → `240`, debit SL 40%→25%, debit PT 50%→25% | 96% of exits were `closed_time_limit`; tighter SL cuts losses faster, lower PT more achievable |
| 2026-02-14 | `core/paper_trading_engine.py` | Explicitly pass `entry_time=_now_ist()` and `entry_date` in `open_position()` | **Bug fix**: Pydantic `default_factory=_now_ist` captured pre-mock reference — simulation positions got wall-clock time, not virtual time. This broke `max_trades_per_day` gate (never matched dates) and `debit_max_hold_minutes` (hold time was wrong). Fixing this alone reduced trades from 1,514 to 65 per round |
| 2026-02-14 | `ui/simulation_tab.py` | Add algorithm selector to "Run All" tab — pick one algo or all | Batch all scenarios for one algorithm without running full 4-algo matrix |
| 2026-02-15 | `simulation/chain_synthesizer.py` | Add `_compute_dynamic_atm_iv()` — ATM IV reacts to spot moves (asymmetric: down 1.5x) + VIX scaling; add `_directional_spread_multiplier()` — PE spreads widen on down moves, CE on up | IV was static across all 375 ticks — credit strategies never faced IV expansion when spot moved against them |
| 2026-02-15 | `simulation/chain_synthesizer.py` | `update_chain()` accepts `open_price`, `current_vix`, `iv_bump`, `extra_spread_mult` params; uses dynamic IV + directional spreads | Wire dynamic IV and adversarial inputs into chain update pipeline |
| 2026-02-15 | `simulation/scenario_models.py` | Add `iv_reactivity: float = 0.5` to `ChainConfig`; add `AdversarialConfig` model (mode + intensity); add `adversarial` field to `Scenario` and `DayConfig` | Control IV sensitivity per scenario; enable adversarial engine configuration in YAML |
| 2026-02-15 | `simulation/adversarial.py` | Add `iv_bump: float` to `PerturbResult`; add `IV_EXPANSION` mode (reflexive IV ramp when positions losing) + `SPREAD_BLOW` mode (3-5x spreads when algo needs to exit) | Adversarial engine was dead code; new modes specifically target credit strategy weaknesses |
| 2026-02-15 | `simulation/runner.py` | Wire `AdversarialPerturber` into `run_day()` tick loop; pass `open_price`/`vix`/`iv_bump`/`extra_spread_mult` to `update_chain()` | Connect adversarial engine + dynamic IV to runner; previously adversarial.py was never called |
| 2026-02-15 | `simulation/scenarios/credit_killers.yaml` (new, 6 scenarios) | iv_expansion_no_move, persistent_gap_down_no_fill, gamma_bleed_oscillation, overnight_gap_through, expiry_day_crush, credit_crush_combo | Credit strategies showed 100% WR across 28 scenarios; new scenarios target vega risk, gap-through, gamma bleed, expiry crush — credit_crush_combo produces -172K loss for sentinel |
| 2026-02-15 | `core/trade_strategies.py` | Add regime penalties to all 5 credit strategies: IV percentile >75 → -10, ATM IV >20 → -10 | Credit strategies entered freely in high-IV stress environments; needed scoring pressure to reduce entries in adverse regimes |
| 2026-02-15 | `core/trade_strategies.py` | Raise debit IV penalty threshold 14→18; add pre-breakout BB tight bonus (+5) to Bull Call/Bear Put Spread | Improve debit entry quality — higher IV threshold allows entries in moderate IV; BB tight bonus rewards compression before expansion |
| 2026-02-15 | `core/paper_trading_models.py` | Add `entry_atm_iv: float = 0.0` to PaperPosition; add `CLOSED_IV_EXPANSION` to PositionStatus | Track ATM IV at entry for IV-expansion exit; new exit reason for audit trail |
| 2026-02-15 | `core/paper_trading_engine.py` | Add IV-expansion exit in Phase 1: close credit positions when ATM IV rises 3+ pts from entry; set `entry_atm_iv` in `open_position()` | Credit positions had no defense against IV expansion during hold; SL only triggered on premium, not on underlying risk driver (IV) |
| 2026-02-15 | `config.yaml` | `credit_fallback_multiplier` 2.0→1.5; `debit_min_score_to_trade` 50→70 (credit-focused); debit SL 25→35%, PT 25→50%, hold 240→300 min | Tighter credit SL; debit threshold 70 makes V1 credit-focused (V1 debit scoring produces 0% WR — systematic opposing entries cancel out); debit params retained for future use |
| 2026-02-15 | `core/paper_trading_engine.py` | Relax confidence gate for debit: accept "Medium" (score >= 35) instead of only "High" | Lower barrier for debit entries when score threshold is met (currently inactive with threshold 70) |
| 2026-02-15 | `core/context_models.py` (new) | 5 Pydantic models: SessionContext, DailyContext, WeeklyContext, VolContext, MarketContext | Multi-level temporal context for algorithms (session/daily/weekly/vol regime) |
| 2026-02-15 | `core/context_engine.py` (new) | ContextEngine class: bootstrap_history (60d yfinance daily candles → DailyContext + WeeklyContext), update_session (intraday), end_of_day, get_context assembly + 7 pure computation helpers | Algorithms only saw point-in-time data; no weekly trend, prior day summary, or vol regime persistence |
| 2026-02-15 | `core/database.py` | Add 3 new tables (daily_context, weekly_context, vol_regime_log) + 7 new methods | Persistent storage for context engine data |
| 2026-02-15 | `algorithms/base.py` | Add `context: MarketContext \| None = None` param to both ABC methods | Algorithms receive multi-level context (pass-through for now) |
| 2026-02-15 | `algorithms/sentinel.py`, `jarvis.py`, `optimus.py`, `atlas.py` | Accept `context` parameter in generate_suggestions + evaluate_and_manage | ABC compliance; no logic changes yet |
| 2026-02-15 | `core/paper_trading_engine.py` | Add `context=None` to V1 evaluate_and_manage signature | Accept context kwarg from caller (pass-through) |
| 2026-02-15 | `core/options_models.py` | Add `context: Any = None` to OptionsDeskSnapshot | Carry context through snapshot pipeline |
| 2026-02-15 | `app.py` | Initialize ContextEngine singleton, call update_session every refresh, pass context to algorithms | Wire context into the refresh loop |
| 2026-02-15 | `ui/options_desk_tab.py` | Store candle_df in session_state; add Market Context expander (prior day, weekly, vol regime, session) | Context visibility in dashboard |
| 2026-02-15 | `ui/paper_trading_tab.py` | Accept and pass `context` parameter to evaluate_fn | Wire context to algorithm evaluate calls |
| 2026-02-15 | `config.yaml`, `core/config_schema.py` | Add `context_engine` section (6 params) + `ContextEngineConfig` Pydantic model | Configurable bootstrap days, ticker, and per-level enable flags |
| 2026-02-15 | `tests/test_context_engine.py` (new) | 51 tests covering candle classification, daily/weekly/session/vol context computation, bootstrap, assembly, persistence, backward compat, serialization | Full test coverage for context engine (537 total tests) |
| 2026-02-15 | `core/trade_strategies.py` | Add `_apply_context_adjustments()` post-processor + `context` param to `generate_trade_suggestions()` | Layer multi-level context (vol regime, trend, prior day, session) onto V1 scoring; runs after evaluators, before dedup |
| 2026-02-15 | `algorithms/sentinel.py` | Forward `context` to `_v1_generate_suggestions()` and `_v1_evaluate_and_manage()` | Sentinel now passes context through to V1 logic (was ignored) |
| 2026-02-15 | `core/paper_trading_engine.py` | Add vol stand-down min_score bump, context summary in `trade_status_notes`, notes on all Phase 2 return paths | Stand-down regime raises entry bar; operators see context state in dashboard |
| 2026-02-15 | `config.yaml` | Add `context_adjustments` section (14 params) under `trade_suggestions` + `standdown_min_score_bump: 10` under `paper_trading` | All context scoring adjustments configurable; stand-down bump configurable |
| 2026-02-15 | `core/parameter_bounds.py` | Add bounds for 14 context adjustment params + `standdown_min_score_bump` | Safety bounds for new tunable params |
| 2026-02-15 | `core/config_schema.py` | Add `ContextAdjustmentsConfig` Pydantic model (14 fields) | Validate context adjustment config at startup |
| 2026-02-15 | `core/strategy_rules.py` | Add `[CTX]` scoring rules (vol_regime, multi_day_trend, session_trend) to all 11 strategies | Machine-readable context rules for criticizer and self-improvement system |
| 2026-02-15 | `tests/test_context_adjustments.py` (new) | 34 tests: credit/debit vol regime, RV trend, trend alignment/conflict, prior day doji/range, session range, score floor, confidence recompute, reasoning prefix, max swing, graceful degradation | Full test coverage for context integration (571 total tests) |
| 2026-02-16 | `core/trade_strategies.py` | Add 6 new context blocks to `_apply_context_adjustments()`: session EMA/RSI/BB/VWAP, observation bias, weekly trend, regime stability, consecutive days; add `observation` param to function + `generate_trade_suggestions()` | Directional credit (BPS/BCS) can accumulate +15-25 pts from context, pushing scores into High confidence range; 8→23 context signals used |
| 2026-02-16 | `core/paper_trading_engine.py` | Change 2: Confidence gate accepts Medium for credit with 3+ positive CTX and 0 negative; Change 3: Re-entry after PT with cooldown (SL/IV/context exits always blocked); Change 4: Context exits (neutral trend, directional reversal, vol regime shift); Change 5: Debit unlock in buy_premium regime with aligned trend+EMA | Unlock BPS/BCS when context supports directional bias; exit positions when context turns adverse; allow debit in buy_premium |
| 2026-02-16 | `core/paper_trading_models.py` | Add `CLOSED_CONTEXT_EXIT` to PositionStatus; add `entry_vol_regime` field to PaperPosition | New exit reason for context-driven closes; track vol regime at entry for regime shift exit |
| 2026-02-16 | `algorithms/sentinel.py` | Forward `observation` to `_v1_generate_suggestions()` | Sentinel passes observation to context adjustments for bias-based scoring |
| 2026-02-16 | `config.yaml` | Add 15 new context adjustment params + 5 paper trading params (`reentry_after_pt_cooldown_minutes`, `context_exit_enabled/range_pct/min_hold_minutes`, `debit_context_unlock_threshold`) | All new context signals and exit/entry logic configurable |
| 2026-02-16 | `core/config_schema.py` | Add 15 fields to `ContextAdjustmentsConfig` + 5 fields to `PaperTradingConfig` | Validate all new config params at startup |
| 2026-02-16 | `core/parameter_bounds.py` | Add bounds for 19 new params (15 context adjustments + 4 paper trading) | Safety bounds for all new tunable params |
| 2026-02-16 | `core/strategy_rules.py` | Add new `[CTX]` scoring rules (session EMA, RSI, BB, VWAP, observation, weekly, stability, instability, consecutive days) to BPS, BCS, SS, SStr, IC | Machine-readable rules for criticizer/self-improvement system |
| 2026-02-16 | `tests/test_context_adjustments.py` | +38 new tests: enhanced context (16), confidence gate (3), re-entry (4), context exits (8), debit unlock (3), integration (4) | Full test coverage for dynamic sentinel changes (609 total tests) |
| 2026-02-16 | `simulation/scenario_models.py` | Add `regime_duration_days` (int=5), `regime_changes_30d` (int=2), `rv_trend` (str="stable") to `VolRegimeConfig` | Context engine needs regime persistence and RV trend info; defaults are neutral so existing YAML unchanged |
| 2026-02-16 | `simulation/context_assembler.py` (new) | `SimContextBuilder`: 55 synthetic daily bars (GBM), DailyContext/WeeklyContext/VolContext from scenario params, per-tick `build_context()` → MarketContext, `end_of_day()` for multi-day | Runner had no context → all 5 context-driven behaviors were dead code in simulation |
| 2026-02-16 | `simulation/runner.py` | Add `context_builder` param to `run_day()`, create `SimContextBuilder`, pass `context=` to `generate_suggestions()` + `evaluate_and_manage()` per tick; wire `end_of_day()` in `run_multi_day()` | Context adjustments, vol stand-down, context exits, confidence loosening, debit unlock now active in simulation |
| 2026-02-16 | `tests/test_sim_context.py` (new) | 42 tests: init (8), vol context (5), build context (6), end of day (6), multi-day (4), reproducibility (2), scenario fields (3), helpers (8) | Full test coverage for simulation context assembler (651 total tests) |
| 2026-02-25 | `core/database.py` | Add `ClosedTradeRow` table + `save_trade()`, `get_trades()`, `get_all_trades()` methods | Trade history only lived in JSON state files; lost on state resets or file corruption |
| 2026-02-25 | `ui/paper_trading_tab.py` | Add `_persist_trades_to_db()` helper, wire into 3 close paths (eval engine, Close All, Manual Close) | Every closed trade now persisted to SQLite as durable source of truth |
| 2026-02-25 | `ui/paper_trading_tab.py` | Rewrite `_render_trade_history()` to read from SQLite via `_load_trade_history()` instead of `state.trade_log` | Trade history survives state resets; DB is the source of truth for display |
| 2026-02-25 | `ui/paper_trading_tab.py` | Add `_migrate_state_trades_to_db()` one-time migration on first load per algo | Existing JSON trades migrated to SQLite on first session after upgrade |
