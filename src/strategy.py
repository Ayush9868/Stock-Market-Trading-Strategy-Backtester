"""
strategy.py
============

Implements the trading strategy logic: computing a Simple Moving Average
(SMA) and generating discrete BUY/SELL signals only on true crossovers.

Avoiding look-ahead bias
------------------------
A signal on day `t` is derived only from information available at the close
of day `t` (today's close/SMA and yesterday's close/SMA via `.shift(1)`).
No future values are used. The backtester (see `backtester.py`) then
deliberately *executes* that signal at the **next day's opening price**,
which is the earliest price at which a trader could realistically act on
information learned at today's close. This two-step separation is the
standard way to avoid look-ahead bias in a daily-bar backtest.
"""

from __future__ import annotations

import pandas as pd

BUY = 1
SELL = -1
HOLD = 0


def calculate_moving_average(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.Series:
    """
    Compute a Simple Moving Average (SMA) over `price_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `price_col`.
    window : int
        Number of trading days in the rolling window (e.g. 20).
    price_col : str, default "Close"
        Column to average.

    Returns
    -------
    pd.Series
        SMA values; the first `window - 1` entries are NaN by construction
        (not enough history yet) and are left as NaN rather than
        back-filled, so downstream signal logic naturally ignores them.
    """
    if window < 2:
        raise ValueError("Moving average window must be at least 2 days.")
    return df[price_col].rolling(window=window, min_periods=window).mean()


def generate_signals(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """
    Add `SMA_{window}` and `Signal` columns to a copy of `df`.

    A signal fires **only on the day of an actual crossover**, not on every
    day the price happens to be above/below the average:

    * `Signal ==  1` (BUY)  : price crosses from at-or-below the SMA to above it.
    * `Signal == -1` (SELL) : price crosses from at-or-above the SMA to below it.
    * `Signal ==  0`        : no crossover that day.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `price_col` (default "Close").
    window : int
        Moving-average window in trading days.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with two new columns: `SMA_{window}` and `Signal`.
    """
    out = df.copy()
    sma_col = f"SMA_{window}"
    out[sma_col] = calculate_moving_average(out, window, price_col=price_col)

    price = out[price_col]
    sma = out[sma_col]
    prev_price = price.shift(1)
    prev_sma = sma.shift(1)

    valid = sma.notna() & prev_sma.notna()

    crossed_up = valid & (price > sma) & (prev_price <= prev_sma)
    crossed_down = valid & (price < sma) & (prev_price >= prev_sma)

    out["Signal"] = HOLD
    out.loc[crossed_up, "Signal"] = BUY
    out.loc[crossed_down, "Signal"] = SELL

    # Rows before the SMA has warmed up cannot produce a signal.
    out.loc[sma.isna(), "Signal"] = HOLD

    return out
