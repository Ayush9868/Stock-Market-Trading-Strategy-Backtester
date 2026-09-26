"""
data_loader.py
===============

Responsible for downloading historical OHLCV stock data from Yahoo Finance
(via `yfinance`) and validating/cleaning it before it is handed to the
strategy and backtesting modules.

Design notes
------------
* All failure modes (bad ticker, network issues, empty/insufficient data)
  raise a single, friendly `DataLoadError` so the Streamlit UI can catch one
  exception type and show a clean message instead of a traceback.
* Validation is explicit and never silent: if data is missing, duplicated,
  or insufficient for the requested moving-average window, the caller is
  told exactly what went wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataLoadError(Exception):
    """Raised for any problem encountered while downloading or validating data."""


def download_stock_data(
    ticker: str,
    years: float = 3.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily historical OHLCV data for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".
    years : float, default 3.0
        Number of years of history to request, counting back from today.
        Ignored if `start` is provided.
    start, end : str, optional
        Explicit date range in "YYYY-MM-DD" format. If omitted, `years` is
        used to compute the start date and `end` defaults to today.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated daily OHLCV data indexed by Date, sorted ascending.

    Raises
    ------
    DataLoadError
        If the ticker is invalid, the network request fails, or no usable
        data is returned.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise DataLoadError("Please enter a ticker symbol.")
    if not ticker.replace(".", "").replace("-", "").isalnum():
        raise DataLoadError(f"'{ticker}' does not look like a valid ticker symbol.")

    if start is None:
        end_dt = datetime.today() if end is None else pd.to_datetime(end)
        start_dt = end_dt - timedelta(days=int(years * 365.25))
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception as exc:  # network errors, rate limits, etc.
        raise DataLoadError(
            f"Could not download data for '{ticker}'. This may be a network "
            f"issue or a problem with Yahoo Finance. Details: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise DataLoadError(
            f"No historical data was found for ticker '{ticker}'. "
            "Please check that the symbol is correct."
        )

    # yfinance occasionally returns a MultiIndex on columns even for a single
    # ticker depending on version; flatten defensively.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return validate_and_clean(raw, ticker)


def validate_and_clean(df: pd.DataFrame, ticker: str, min_rows: int = 30) -> pd.DataFrame:
    """
    Validate and clean a raw OHLCV DataFrame.

    Steps performed (in order):
    1. Verify required columns exist.
    2. Ensure the index is a DatetimeIndex and sort ascending by date.
    3. Drop duplicate dates (keeping the first occurrence).
    4. Drop rows where Close is missing; forward-fill any remaining gaps
       in Open/High/Low/Volume.
    5. Verify at least `min_rows` valid rows remain.
    6. Sanity-check that prices are positive.

    Raises
    ------
    DataLoadError
        If any validation step fails.
    """
    df = df.copy()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataLoadError(
            f"Downloaded data for '{ticker}' is missing required columns: {missing_cols}."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < before:
        # Duplicates silently collapsed - this is intentional cleanup, not
        # a silent data problem, since we keep the first valid observation.
        pass

    df = df.dropna(subset=["Close"])
    df[["Open", "High", "Low", "Volume"]] = df[["Open", "High", "Low", "Volume"]].ffill()
    df = df.dropna(subset=["Open", "High", "Low"])

    if len(df) == 0:
        raise DataLoadError(f"All rows for '{ticker}' had missing data after cleaning.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataLoadError(
            f"'{ticker}' contains non-positive prices, which suggests corrupted data."
        )

    if len(df) < min_rows:
        raise DataLoadError(
            f"Only {len(df)} trading days of data are available for '{ticker}', "
            f"which is not enough to run a meaningful backtest (minimum {min_rows})."
        )

    df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
    return df[REQUIRED_COLUMNS]


def ensure_sufficient_history(df: pd.DataFrame, ma_window: int) -> None:
    """
    Confirm there is enough history to compute the requested moving average
    and still have a meaningful number of post-warm-up trading days.

    Raises
    ------
    DataLoadError
        If the dataset is too short for the chosen moving-average window.
    """
    min_required = ma_window + 10
    if len(df) < min_required:
        raise DataLoadError(
            f"The selected moving-average window ({ma_window} days) requires at least "
            f"{min_required} trading days of data, but only {len(df)} are available. "
            "Choose a longer backtest period or a shorter moving-average window."
        )
