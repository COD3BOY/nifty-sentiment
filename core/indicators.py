"""Pure functions for technical indicator computation."""

import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP with daily reset. Expects columns: High, Low, Close, Volume."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = typical_price * df["Volume"]

    # Group by date for daily reset
    dates = df.index.date
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 9) -> pd.Series:
    """RSI using exponential weighted moving average of gains/losses."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def compute_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator. Returns (supertrend_values, directions).

    direction: 1 = bullish (price above band), -1 = bearish.
    """
    atr = compute_atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2.0

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    close = df["Close"]
    n = len(df)

    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)

    # Arrays for sequential computation
    ub = upper_band.values.copy()
    lb = lower_band.values.copy()
    cl = close.values

    for i in range(1, n):
        # Adjust bands based on previous values
        if ub[i] < ub[i - 1] or cl[i - 1] > ub[i - 1]:
            pass  # keep current upper band
        else:
            ub[i] = ub[i - 1]

        if lb[i] > lb[i - 1] or cl[i - 1] < lb[i - 1]:
            pass  # keep current lower band
        else:
            lb[i] = lb[i - 1]

        # Determine direction
        if supertrend[i - 1] == ub[i - 1]:
            # Was bearish
            if cl[i] > ub[i]:
                direction[i] = 1
                supertrend[i] = lb[i]
            else:
                direction[i] = -1
                supertrend[i] = ub[i]
        else:
            # Was bullish
            if cl[i] < lb[i]:
                direction[i] = -1
                supertrend[i] = ub[i]
            else:
                direction[i] = 1
                supertrend[i] = lb[i]

    st_series = pd.Series(supertrend, index=df.index)
    dir_series = pd.Series(direction, index=df.index)
    return st_series, dir_series


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower
