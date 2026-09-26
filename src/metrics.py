"""
metrics.py
===========

Performance and risk metric calculations shared by the strategy and the
Buy & Hold benchmark. All functions take a portfolio value series (or daily
returns) and compute a single, well-defined statistic.

Metric definitions
-------------------
* Total Return %      : (Final Value / Initial Value - 1) * 100
* CAGR                 : Compound Annual Growth Rate - the constant annual
                         rate of return that would take Initial Value to
                         Final Value over the observed number of years.
* Max Drawdown         : The largest peak-to-trough percentage decline in
                         portfolio value observed over the period.
* Annualized Volatility: Standard deviation of daily returns, scaled to a
                         yearly figure by multiplying by sqrt(252).
* Sharpe Ratio         : Mean daily excess return over the risk-free rate,
                         divided by the standard deviation of daily excess
                         returns, annualized by multiplying by sqrt(252).
                         A higher Sharpe Ratio means better risk-adjusted
                         return.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_total_return(portfolio_value: pd.Series) -> float:
    """Total return over the period, as a percentage."""
    if len(portfolio_value) < 2 or portfolio_value.iloc[0] == 0:
        return 0.0
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100


def calculate_cagr(portfolio_value: pd.Series) -> float:
    """
    Compound Annual Growth Rate, as a percentage.

    CAGR = (Final / Initial) ** (1 / years) - 1
    """
    if len(portfolio_value) < 2 or portfolio_value.iloc[0] <= 0:
        return 0.0

    n_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    years = n_days / 365.25
    if years <= 0:
        return 0.0

    ratio = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
    if ratio < 0:
        return 0.0
    return (ratio ** (1 / years) - 1) * 100


def calculate_drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    """
    Drawdown at every point in time, as a fraction (e.g. -0.15 = -15%).

    Drawdown = (Portfolio Value - Running Maximum) / Running Maximum
    """
    running_max = portfolio_value.cummax()
    return (portfolio_value - running_max) / running_max


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """Maximum drawdown over the period, as a percentage (negative number)."""
    dd = calculate_drawdown_series(portfolio_value)
    return float(dd.min() * 100) if len(dd) else 0.0


def calculate_annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility (std. dev. of daily returns), as a percentage."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe Ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    risk_free_rate : float, default 0.0
        Annual risk-free rate (e.g. 0.04 for 4%), converted internally to a
        daily rate for the excess-return calculation.
    """
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily_returns - daily_rf
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_win_rate(trades: List) -> Dict[str, float]:
    """
    Compute winning/losing trade counts and win rate from a list of Trade
    objects (must expose a `.profit_loss` attribute).
    """
    total = len(trades)
    if total == 0:
        return {"winning_trades": 0, "losing_trades": 0, "win_rate": 0.0}
    wins = sum(1 for t in trades if t.profit_loss > 0)
    losses = sum(1 for t in trades if t.profit_loss <= 0)
    return {
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round((wins / total) * 100, 2),
    }


def calculate_performance_metrics(
    portfolio_value: pd.Series,
    daily_returns: pd.Series,
    initial_capital: float,
    trades: List | None = None,
) -> Dict[str, float]:
    """
    Bundle every headline metric into a single dictionary, ready to render
    as KPI cards in the dashboard.
    """
    trades = trades or []
    final_value = float(portfolio_value.iloc[-1]) if len(portfolio_value) else initial_capital
    metrics = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "profit_loss": final_value - initial_capital,
        "total_return_pct": calculate_total_return(portfolio_value),
        "cagr_pct": calculate_cagr(portfolio_value),
        "max_drawdown_pct": calculate_max_drawdown(portfolio_value),
        "annualized_volatility_pct": calculate_annualized_volatility(daily_returns),
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns),
        "num_trades": len(trades),
    }
    metrics.update(calculate_win_rate(trades))
    return metrics


def calculate_annual_returns(
    strategy_value: pd.Series, benchmark_value: pd.Series
) -> pd.DataFrame:
    """
    Build a year-by-year return comparison table between the strategy and
    the Buy & Hold benchmark.

    For each calendar year present in the data, the return is calculated
    from the first available value in that year to the last available
    value in that year (so partial first/last years are handled sensibly
    rather than assumed to span a full 365 days).
    """
    df = pd.DataFrame(
        {"Strategy": strategy_value, "Benchmark": benchmark_value}
    ).dropna()
    if df.empty:
        return pd.DataFrame(columns=["Year", "Strategy Return %", "Buy & Hold Return %"])

    df["Year"] = df.index.year
    rows = []
    for year, group in df.groupby("Year"):
        strat_start, strat_end = group["Strategy"].iloc[0], group["Strategy"].iloc[-1]
        bench_start, bench_end = group["Benchmark"].iloc[0], group["Benchmark"].iloc[-1]
        rows.append(
            {
                "Year": int(year),
                "Strategy Return %": round((strat_end / strat_start - 1) * 100, 2),
                "Buy & Hold Return %": round((bench_end / bench_start - 1) * 100, 2),
            }
        )
    return pd.DataFrame(rows)
