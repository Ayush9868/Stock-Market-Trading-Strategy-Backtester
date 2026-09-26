"""Unit tests for src/metrics.py using small synthetic series."""

import pandas as pd
import pytest

from src.backtester import Trade
from src.metrics import (
    calculate_annual_returns,
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_win_rate,
)


def make_series(values, freq="D"):
    dates = pd.date_range("2023-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=dates)


def test_total_return():
    s = make_series([100, 110, 120, 150])
    assert calculate_total_return(s) == pytest.approx(50.0)


def test_cagr_one_year_doubling():
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    s = pd.Series([100] * (len(dates) - 1) + [200], index=dates)
    cagr = calculate_cagr(s)
    # ~1 year to double => CAGR should be close to 100%.
    assert cagr == pytest.approx(100.0, abs=5.0)


def test_cagr_handles_degenerate_series():
    assert calculate_cagr(make_series([100])) == 0.0
    assert calculate_cagr(make_series([0, 100])) == 0.0


def test_max_drawdown():
    s = make_series([100, 120, 90, 110, 80, 130])
    dd = calculate_max_drawdown(s)
    # Peak 120 -> trough 80 = -33.33%
    assert dd == pytest.approx(-33.33, abs=0.05)


def test_drawdown_series_is_never_positive():
    s = make_series([100, 90, 95, 80, 120, 60])
    dd = calculate_drawdown_series(s)
    assert (dd <= 1e-9).all()
    assert dd.iloc[0] == pytest.approx(0.0)


def test_volatility_zero_for_constant_returns():
    returns = make_series([0.0] * 10)
    assert calculate_annualized_volatility(returns) == pytest.approx(0.0)


def test_volatility_positive_for_varying_returns():
    returns = make_series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01])
    assert calculate_annualized_volatility(returns) > 0.0


def test_sharpe_ratio_zero_when_no_variance():
    returns = make_series([0.001] * 20)
    assert calculate_sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_upward_drifting_returns():
    returns = make_series([0.01, 0.02, 0.015, 0.005, 0.012, 0.008] * 5)
    assert calculate_sharpe_ratio(returns) > 0.0


def test_win_rate_counts_trades_correctly():
    trades = [
        Trade(pd.Timestamp("2023-01-01"), 10, pd.Timestamp("2023-01-05"), 12, 10, 100, 120),  # win: +20
        Trade(pd.Timestamp("2023-02-01"), 10, pd.Timestamp("2023-02-05"), 8, 10, 100, 80),    # loss: -20
        Trade(pd.Timestamp("2023-03-01"), 10, pd.Timestamp("2023-03-05"), 10, 10, 100, 100),  # breakeven -> loss
    ]
    stats = calculate_win_rate(trades)
    assert stats["winning_trades"] == 1
    assert stats["losing_trades"] == 2
    assert stats["win_rate"] == pytest.approx(33.33, abs=0.05)


def test_win_rate_with_no_trades():
    stats = calculate_win_rate([])
    assert stats == {"winning_trades": 0, "losing_trades": 0, "win_rate": 0.0}


def test_annual_returns_splits_by_calendar_year():
    dates = pd.date_range("2023-01-01", "2024-06-30", freq="D")
    strategy = pd.Series(range(100, 100 + len(dates)), index=dates, dtype=float)
    benchmark = pd.Series(range(100, 100 + len(dates)), index=dates, dtype=float)

    table = calculate_annual_returns(strategy, benchmark)
    assert set(table["Year"]) == {2023, 2024}
    assert (table["Strategy Return %"] == table["Buy & Hold Return %"]).all()
