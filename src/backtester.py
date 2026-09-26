"""
backtester.py
==============

Simulates the strategy day by day and produces:

1. A daily portfolio DataFrame (cash, shares, position, portfolio value,
   daily return) for the SMA crossover strategy.
2. A trade-by-trade history of completed round trips.
3. A Buy & Hold benchmark series that invests the same starting capital in
   the stock on day one and holds it for the full period.

Execution model (avoiding look-ahead bias)
-------------------------------------------
`generate_signals()` produces a `Signal` on the day the crossover is
*detected* (based on that day's close). A real trader could not have acted
on that information until the market reopened, so this backtester executes
every signal at the **next trading day's Open** price:

    Signal Generation (close of day t)
            -> Trade Execution (open of day t+1)
                    -> Portfolio Valuation (close of day t+1 onward)
                            -> Performance Analysis (end of backtest)

Only one position may be open at a time: a BUY signal is ignored while
already in a position, and a SELL signal is ignored while flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .strategy import BUY, SELL

DEFAULT_TRANSACTION_COST = 0.001  # 0.1%


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    cash_invested: float
    proceeds: float

    @property
    def profit_loss(self) -> float:
        return self.proceeds - self.cash_invested

    @property
    def return_pct(self) -> float:
        return (self.profit_loss / self.cash_invested) * 100 if self.cash_invested else 0.0

    @property
    def holding_period_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    portfolio: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 0.0
    transaction_cost: float = DEFAULT_TRANSACTION_COST


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> BacktestResult:
    """
    Run the SMA-crossover backtest over `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `Open`, `Close`, and `Signal` columns (see
        `strategy.generate_signals`).
    initial_capital : float, default 10000.0
        Starting cash.
    transaction_cost : float, default 0.001
        Fractional cost applied on both entry and exit (e.g. 0.001 = 0.1%).

    Returns
    -------
    BacktestResult
        Daily portfolio DataFrame plus the list of completed trades.
    """
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if transaction_cost < 0:
        raise ValueError("Transaction cost cannot be negative.")

    # Execute today what was *signaled* yesterday, at today's open.
    exec_signal = df["Signal"].shift(1).fillna(0)

    n = len(df)
    cash = float(initial_capital)
    shares = 0.0
    position = 0  # 0 = flat, 1 = long

    cash_series = np.empty(n)
    shares_series = np.empty(n)
    position_series = np.empty(n, dtype=int)
    value_series = np.empty(n)

    trades: List[Trade] = []
    open_entry_date = None
    open_entry_price = None
    open_entry_shares = None
    open_cash_invested = None

    opens = df["Open"].to_numpy()
    closes = df["Close"].to_numpy()
    dates = df.index

    for i in range(n):
        sig = exec_signal.iloc[i]
        open_price = opens[i]

        if sig == BUY and position == 0 and cash > 0:
            shares = cash / (open_price * (1 + transaction_cost))
            open_cash_invested = cash
            open_entry_price = open_price
            open_entry_date = dates[i]
            open_entry_shares = shares
            cash = 0.0
            position = 1

        elif sig == SELL and position == 1 and shares > 0:
            proceeds = shares * open_price * (1 - transaction_cost)
            trades.append(
                Trade(
                    entry_date=open_entry_date,
                    entry_price=open_entry_price,
                    exit_date=dates[i],
                    exit_price=open_price,
                    shares=open_entry_shares,
                    cash_invested=open_cash_invested,
                    proceeds=proceeds,
                )
            )
            cash = proceeds
            shares = 0.0
            position = 0
            open_entry_date = open_entry_price = open_entry_shares = open_cash_invested = None

        close_price = closes[i]
        portfolio_value = cash + shares * close_price

        cash_series[i] = cash
        shares_series[i] = shares
        position_series[i] = position
        value_series[i] = portfolio_value

    portfolio = pd.DataFrame(
        {
            "Close": df["Close"],
            "Signal": df["Signal"],
            "Position": position_series,
            "Cash": cash_series,
            "Shares": shares_series,
            "Portfolio_Value": value_series,
        },
        index=df.index,
    )
    sma_cols = [c for c in df.columns if c.startswith("SMA_")]
    for col in sma_cols:
        portfolio[col] = df[col]

    portfolio["Daily_Return"] = portfolio["Portfolio_Value"].pct_change().fillna(0.0)

    return BacktestResult(
        portfolio=portfolio,
        trades=trades,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )


def calculate_buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Simulate investing `initial_capital` entirely in the stock at the first
    available Open price and holding until the last available Close,
    applying the same one-time entry transaction cost as the strategy.

    Returns
    -------
    pd.DataFrame
        Indexed like `df`, with `Portfolio_Value` and `Daily_Return` columns,
        starting at exactly `initial_capital` (so it is directly comparable
        to the strategy's portfolio series).
    """
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    first_open = df["Open"].iloc[0]
    shares = initial_capital / (first_open * (1 + transaction_cost))
    values = shares * df["Close"]
    values.iloc[0] = initial_capital  # anchor both series to the same start

    out = pd.DataFrame(index=df.index)
    out["Portfolio_Value"] = values
    out["Daily_Return"] = out["Portfolio_Value"].pct_change().fillna(0.0)
    return out


def generate_trade_history(trades: List[Trade]) -> pd.DataFrame:
    """
    Convert a list of `Trade` objects into a readable trade-history table.

    Columns: Trade #, Entry Date, Entry Price, Exit Date, Exit Price,
    Shares, Profit/Loss, Return %, Holding Period (days).
    """
    if not trades:
        return pd.DataFrame(
            columns=[
                "Trade #",
                "Entry Date",
                "Entry Price",
                "Exit Date",
                "Exit Price",
                "Shares",
                "Profit/Loss",
                "Return %",
                "Holding Period (days)",
            ]
        )

    rows = []
    for idx, t in enumerate(trades, start=1):
        rows.append(
            {
                "Trade #": idx,
                "Entry Date": t.entry_date.date(),
                "Entry Price": round(t.entry_price, 2),
                "Exit Date": t.exit_date.date(),
                "Exit Price": round(t.exit_price, 2),
                "Shares": round(t.shares, 4),
                "Profit/Loss": round(t.profit_loss, 2),
                "Return %": round(t.return_pct, 2),
                "Holding Period (days)": t.holding_period_days,
            }
        )
    return pd.DataFrame(rows)
