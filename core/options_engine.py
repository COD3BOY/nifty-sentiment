"""Orchestrator for the Intraday Options Desk."""

import logging
import time

import pandas as pd

from core.config import load_config
from core.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
    compute_supertrend,
    compute_vwap,
)
from core.intraday_fetcher import IntradayCandleFetcher
from core.nse_fetcher import KiteOptionChainFetcher
from core.greeks import compute_chain_deltas
from core.options_analytics import build_analytics
from core.options_models import (
    FetchMeta,
    OptionChainData,
    OptionsDeskSnapshot,
    OptionsAnalytics,
    SignalCard,
    SignalDirection,
    TechnicalIndicators,
)

logger = logging.getLogger(__name__)


class OptionsDeskEngine:
    """Fetches data and computes analytics for the Options Desk tab."""

    def __init__(self) -> None:
        cfg = load_config().get("options_desk", {})
        symbol = cfg.get("symbol", "NIFTY")
        ticker = cfg.get("nifty_ticker", "^NSEI")
        self._symbol = symbol
        self._kite_chain = KiteOptionChainFetcher()
        self._candle = IntradayCandleFetcher(ticker=ticker)
        self._last_df: pd.DataFrame | None = None
        self._cfg = cfg

    def fetch_snapshot(self) -> OptionsDeskSnapshot:
        """Fetch option chain + candles, compute everything, return snapshot."""
        errors: list[str] = []
        chain: OptionChainData | None = None
        analytics: OptionsAnalytics | None = None
        technicals: TechnicalIndicators | None = None
        chain_meta: FetchMeta | None = None
        candle_meta: FetchMeta | None = None

        # Fetch option chain via Kite Connect
        try:
            chain = self._kite_chain.fetch(self._symbol)
            if not chain.strikes:
                raise ValueError("Kite returned empty data")
            analytics = build_analytics(chain)

            # Compute greeks (delta) for all strikes with IV
            if chain.expiry:
                from datetime import date as _date, datetime as _dt
                try:
                    expiry_date = _dt.strptime(chain.expiry, "%d-%b-%Y").date()
                    calendar_days = (expiry_date - _date.today()).days
                    T = max(calendar_days / 365.0, 0.5 / 365.0)
                    iv_cfg = self._cfg.get("iv_calculator", {})
                    rfr = iv_cfg.get("risk_free_rate", 0.065)
                    chain = chain.model_copy(
                        update={"strikes": compute_chain_deltas(chain, T, risk_free_rate=rfr)}
                    )
                except ValueError:
                    logger.warning("Could not parse expiry date for delta computation")

            # Save IV reading and compute IV percentile
            if analytics.atm_iv > 0:
                try:
                    from core.database import SentimentDatabase
                    from core.iv_history import get_iv_percentile, save_iv_reading
                    db = SentimentDatabase()
                    save_iv_reading(db, self._symbol, analytics.atm_iv)
                    analytics = analytics.model_copy(
                        update={"iv_percentile": get_iv_percentile(db, self._symbol, analytics.atm_iv)}
                    )
                except Exception as iv_exc:
                    logger.warning("IV history update failed: %s", iv_exc)

            chain_meta = FetchMeta(source="Kite Connect", fetch_ts=time.time(), is_primary=True)
            logger.info("Option chain loaded via Kite Connect (%d strikes)", len(chain.strikes))
        except Exception as exc:
            logger.error("Option chain fetch failed: %s", exc)
            errors.append(f"Option chain unavailable: {exc}")
            chain = None

        # Fetch candles
        candle_cfg = self._cfg.get("candles", {})
        period = candle_cfg.get("period", "5d")
        interval = candle_cfg.get("interval", "5m")
        try:
            df = self._candle.fetch(period=period, interval=interval)
            if not df.empty:
                self._last_df = df
                technicals = self._build_technicals(df)
                source = self._candle.last_fetch_source or "unknown"
                candle_meta = FetchMeta(
                    source=source,
                    fetch_ts=time.time(),
                    is_primary="Kite" in source,
                )
        except Exception as exc:
            logger.error("Candle fetch failed: %s", exc)
            errors.append(f"Candle data unavailable: {exc}")

        # Compute observation snapshot from candle data
        observation = None
        if df is not None and not df.empty:
            try:
                from core.observation import compute_observation_snapshot
                obs_end = load_config().get("paper_trading", {}).get("entry_start_time", "10:00")
                observation = compute_observation_snapshot(df, observation_end_time=obs_end)
            except Exception as obs_exc:
                logger.warning("Observation computation failed: %s", obs_exc)

        # Data quality warnings
        if analytics and analytics.atm_iv == 0.0:
            errors.append("IV data unavailable — IV-dependent scores unreliable")
            logger.warning("IV data unavailable (atm_iv=0.0) — IV scores unreliable")
        if analytics and technicals:
            logger.info(
                "Data quality: spot=%.0f, rsi=%.1f, atm_iv=%.1f, bb_width=%.2f%%",
                technicals.spot, technicals.rsi, analytics.atm_iv,
                ((technicals.bb_upper - technicals.bb_lower) / technicals.bb_middle * 100) if technicals.bb_middle > 0 else 0.0,
            )

        signals = self._build_signals(analytics, technicals)

        return OptionsDeskSnapshot(
            chain=chain,
            analytics=analytics,
            technicals=technicals,
            signals=signals,
            trade_suggestions=[],
            errors=errors,
            candle_meta=candle_meta,
            chain_meta=chain_meta,
            observation=observation,
        )

    def get_candle_dataframe(self) -> pd.DataFrame | None:
        """Return the last fetched candle DataFrame (for Plotly charting)."""
        return self._last_df

    def _build_technicals(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Compute all technical indicators from candle data."""
        close = df["Close"]
        last_close = float(close.iloc[-1])

        # Spot change from first bar of latest session
        dates = df.index.date
        today_mask = dates == dates[-1]
        today_df = df[today_mask]
        if len(today_df) > 1:
            first_open = float(today_df["Open"].iloc[0])
            spot_change = last_close - first_open
            spot_change_pct = (spot_change / first_open) * 100.0 if first_open else 0.0
        else:
            spot_change = 0.0
            spot_change_pct = 0.0

        ind_cfg = self._cfg.get("indicators", {})

        vwap = compute_vwap(df)
        ema9 = compute_ema(close, span=ind_cfg.get("ema_fast", 9))
        ema20 = compute_ema(close, span=20)
        ema21 = compute_ema(close, span=ind_cfg.get("ema_mid", 21))
        ema50 = compute_ema(close, span=ind_cfg.get("ema_slow", 50))
        rsi = compute_rsi(close, period=ind_cfg.get("rsi_period", 9))

        st_period = ind_cfg.get("supertrend_period", 10)
        st_mult = ind_cfg.get("supertrend_multiplier", 3.0)
        st_vals, st_dirs = compute_supertrend(df, period=st_period, multiplier=st_mult)

        bb_period = ind_cfg.get("bb_period", 20)
        bb_std = ind_cfg.get("bb_std", 2.0)
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, period=bb_period, std_dev=bb_std)

        # Data staleness: minutes since the last candle timestamp
        last_candle_ts = df.index[-1]
        if last_candle_ts.tzinfo is None:
            # yfinance often returns tz-aware (Asia/Kolkata); handle both
            from datetime import datetime, timezone, timedelta
            _IST = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(_IST)
            # Assume candle is IST if tz-naive
            staleness_min = (now.replace(tzinfo=None) - last_candle_ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        else:
            from datetime import datetime, timezone, timedelta
            _IST = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(_IST)
            staleness_min = (now - last_candle_ts.to_pydatetime().astimezone(_IST)).total_seconds() / 60
        staleness_min = max(0.0, staleness_min)

        return TechnicalIndicators(
            spot=float(last_close),
            spot_change=float(spot_change),
            spot_change_pct=float(spot_change_pct),
            vwap=float(vwap.iloc[-1]),
            ema_9=float(ema9.iloc[-1]),
            ema_20=float(ema20.iloc[-1]),
            ema_21=float(ema21.iloc[-1]),
            ema_50=float(ema50.iloc[-1]),
            rsi=float(rsi.iloc[-1]),
            supertrend=float(st_vals.iloc[-1]),
            supertrend_direction=int(st_dirs.iloc[-1]),
            bb_upper=float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else float(last_close),
            bb_middle=float(bb_mid.iloc[-1]) if pd.notna(bb_mid.iloc[-1]) else float(last_close),
            bb_lower=float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else float(last_close),
            data_staleness_minutes=round(staleness_min, 1),
        )

    def _build_signals(
        self,
        analytics: OptionsAnalytics | None,
        technicals: TechnicalIndicators | None,
    ) -> list[SignalCard]:
        """Aggregate data into 4 signal cards."""
        signals: list[SignalCard] = []
        thresholds = self._cfg.get("signal_thresholds", {})

        # --- Trend Signal ---
        trend_reasons: list[str] = []
        trend_score = 0
        if technicals:
            if technicals.ema_9 > technicals.ema_21 > technicals.ema_50:
                trend_reasons.append("EMA 9 > 21 > 50 (bullish alignment)")
                trend_score += 1
            elif technicals.ema_9 < technicals.ema_21 < technicals.ema_50:
                trend_reasons.append("EMA 9 < 21 < 50 (bearish alignment)")
                trend_score -= 1
            else:
                trend_reasons.append("EMAs not aligned (mixed)")

            if technicals.supertrend_direction == 1:
                trend_reasons.append("Supertrend: Bullish")
                trend_score += 1
            else:
                trend_reasons.append("Supertrend: Bearish")
                trend_score -= 1

            if technicals.spot > technicals.vwap:
                trend_reasons.append(f"Price above VWAP ({technicals.vwap:.0f})")
                trend_score += 1
            else:
                trend_reasons.append(f"Price below VWAP ({technicals.vwap:.0f})")
                trend_score -= 1

        trend_dir = (
            SignalDirection.BULLISH if trend_score >= 2
            else SignalDirection.BEARISH if trend_score <= -2
            else SignalDirection.NEUTRAL
        )
        signals.append(SignalCard(label="Trend", direction=trend_dir, reasoning=trend_reasons))

        # --- Momentum Signal ---
        momentum_reasons: list[str] = []
        momentum_score = 0
        if technicals:
            rsi_ob = thresholds.get("rsi_overbought", 70)
            rsi_os = thresholds.get("rsi_oversold", 30)

            if technicals.rsi > rsi_ob:
                momentum_reasons.append(f"RSI {technicals.rsi:.1f} — Overbought")
                momentum_score -= 1
            elif technicals.rsi < rsi_os:
                momentum_reasons.append(f"RSI {technicals.rsi:.1f} — Oversold")
                momentum_score += 1
            elif technicals.rsi > 55:
                momentum_reasons.append(f"RSI {technicals.rsi:.1f} — Bullish momentum")
                momentum_score += 1
            elif technicals.rsi < 45:
                momentum_reasons.append(f"RSI {technicals.rsi:.1f} — Bearish momentum")
                momentum_score -= 1
            else:
                momentum_reasons.append(f"RSI {technicals.rsi:.1f} — Neutral")

            bb_range = technicals.bb_upper - technicals.bb_lower
            if bb_range > 0:
                bb_pct = (technicals.spot - technicals.bb_lower) / bb_range
                if bb_pct > 0.8:
                    momentum_reasons.append(f"Near upper Bollinger Band ({bb_pct:.0%})")
                    momentum_score -= 1
                elif bb_pct < 0.2:
                    momentum_reasons.append(f"Near lower Bollinger Band ({bb_pct:.0%})")
                    momentum_score += 1
                else:
                    momentum_reasons.append(f"Mid Bollinger Band ({bb_pct:.0%})")

        momentum_dir = (
            SignalDirection.BULLISH if momentum_score >= 1
            else SignalDirection.BEARISH if momentum_score <= -1
            else SignalDirection.NEUTRAL
        )
        signals.append(SignalCard(label="Momentum", direction=momentum_dir, reasoning=momentum_reasons))

        # --- Options Flow Signal ---
        options_reasons: list[str] = []
        options_score = 0
        if analytics:
            pcr_bull = thresholds.get("pcr_bullish", 1.2)
            pcr_bear = thresholds.get("pcr_bearish", 0.7)

            if analytics.pcr > pcr_bull:
                options_reasons.append(f"PCR {analytics.pcr:.2f} — Bullish (heavy put writing)")
                options_score += 1
            elif analytics.pcr < pcr_bear:
                options_reasons.append(f"PCR {analytics.pcr:.2f} — Bearish (low put writing)")
                options_score -= 1
            else:
                options_reasons.append(f"PCR {analytics.pcr:.2f} — Neutral range")

            if technicals and analytics.max_pain > 0:
                diff_pct = ((technicals.spot - analytics.max_pain) / analytics.max_pain) * 100
                if diff_pct > 0.5:
                    options_reasons.append(f"Spot above Max Pain {analytics.max_pain:.0f} — bearish pull")
                    options_score -= 1
                elif diff_pct < -0.5:
                    options_reasons.append(f"Spot below Max Pain {analytics.max_pain:.0f} — bullish pull")
                    options_score += 1
                else:
                    options_reasons.append(f"Spot near Max Pain {analytics.max_pain:.0f}")

            if analytics.iv_skew > 2:
                options_reasons.append(f"IV Skew {analytics.iv_skew:+.1f} — put premium (fear)")
                options_score -= 1
            elif analytics.iv_skew < -2:
                options_reasons.append(f"IV Skew {analytics.iv_skew:+.1f} — call premium (greed)")
                options_score += 1
            elif analytics.iv_skew != 0:
                options_reasons.append(f"IV Skew {analytics.iv_skew:+.1f} — balanced")
        else:
            options_reasons.append("Option chain data unavailable")

        options_dir = (
            SignalDirection.BULLISH if options_score >= 2
            else SignalDirection.BEARISH if options_score <= -2
            else SignalDirection.NEUTRAL
        )
        signals.append(SignalCard(label="Options Flow", direction=options_dir, reasoning=options_reasons))

        # --- Volatility Signal ---
        vol_reasons: list[str] = []
        vol_score = 0
        if analytics and analytics.atm_iv > 0:
            iv_high = thresholds.get("iv_high", 18)
            iv_low = thresholds.get("iv_low", 10)

            if analytics.atm_iv > iv_high:
                vol_reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — High volatility")
                vol_score -= 1
            elif analytics.atm_iv < iv_low:
                vol_reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — Low volatility")
                vol_score += 1
            else:
                vol_reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — Moderate")

        if technicals:
            bb_width = technicals.bb_upper - technicals.bb_lower
            if technicals.bb_middle > 0:
                bb_width_pct = (bb_width / technicals.bb_middle) * 100
                if bb_width_pct > 2.0:
                    vol_reasons.append(f"BB width {bb_width_pct:.1f}% — Expanding")
                    vol_score -= 1
                elif bb_width_pct < 0.8:
                    vol_reasons.append(f"BB width {bb_width_pct:.1f}% — Squeezing")
                    vol_score += 1
                else:
                    vol_reasons.append(f"BB width {bb_width_pct:.1f}% — Normal")

        vol_dir = (
            SignalDirection.BULLISH if vol_score >= 1
            else SignalDirection.BEARISH if vol_score <= -1
            else SignalDirection.NEUTRAL
        )
        signals.append(SignalCard(label="Volatility", direction=vol_dir, reasoning=vol_reasons))

        return signals
