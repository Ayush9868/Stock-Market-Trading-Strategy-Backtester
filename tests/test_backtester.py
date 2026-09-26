"""Unit tests for src/backtester.py using small synthetic datasets."""

import pandas as pd
import pytest

from src.backtester import calculate_buy_and_hold, generate_trade_history, run_backtest
from src.strategy import BUY, HOLD, SELL


def make_signal_df(opens, closes, signals):
    dates = pd.date_range("2023-01-01", periods=len(opens), freq="D")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) for o, c in zip(opens, closes)],
            "Low": [min(o, c) for o, c in zip(opens, closes)],
            "Close": closes,
            "SMA_3": [None] * len(opens),
            "Signal": signals,
        },
        index=dates,
    )


def test_backtest_executes_on_next_day_open_no_cost():
    # Signal fires (BUY) on day 1's close -> executed at day 2's open.
    # Signal fires (SELL) on day 3's close -> executed at day 4's open.
    opens = [10, 10, 12, 15, 16]
    closes = [10, 10, 12, 15, 16]
    signals = [HOLD, BUY, HOLD, SELL, HOLD]
    df = make_signal_df(opens, closes, signals)

    result = run_backtest(df, initial_capital=1000.0, transaction_cost=0.0)
    portfolio = result.portfolio

    # Nothing happens until day index 2 (the day AFTER the BUY signal).
    assert portfolio["Position"].iloc[0] == 0
    assert portfolio["Position"].iloc[1] == 0
    assert portfolio["Position"].iloc[2] == 1  # bought at open of day 2

    expected_shares = 1000.0 / 12  # bought at day-2 open price of 12
    assert portfolio["Shares"].iloc[2] == pytest.approx(expected_shares)
    assert portfolio["Portfolio_Value"].iloc[2] == pytest.approx(1000.0)

    # Sold at day 4's open (16), triggered by the SELL signal on day 3.
    assert portfolio["Position"].iloc[4] == 0
    expected_final_cash = expected_shares * 16
    assert portfolio["Cash"].iloc[4] == pytest.approx(expected_final_cash)
    assert portfolio["Portfolio_Value"].iloc[4] == pytest.approx(expected_final_cash)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(12)
    assert trade.exit_price == pytest.approx(16)
    assert trade.profit_loss == pytest.approx(expected_final_cash - 1000.0)
    assert trade.return_pct == pytest.approx(
        (expected_final_cash - 1000.0) / 1000.0 * 100
    )


def test_transaction_costs_reduce_final_value():
    opens = [10, 10, 12, 15, 16]
    closes = [10, 10, 12, 15, 16]
    signals = [HOLD, BUY, HOLD, SELL, HOLD]
    df = make_signal_df(opens, closes, signals)

    no_cost = run_backtest(df, initial_capital=1000.0, transaction_cost=0.0)
    with_cost = run_backtest(df, initial_capital=1000.0, transaction_cost=0.01)

    final_no_cost = no_cost.portfolio["Portfolio_Value"].iloc[-1]
    final_with_cost = with_cost.portfolio["Portfolio_Value"].iloc[-1]
    assert final_with_cost < final_no_cost


def test_only_one_position_open_at_a_time():
    # Two BUY signals in a row (no SELL between) must not double the position.
    opens = [10, 10, 12, 12, 12]
    closes = [10, 10, 12, 12, 12]
    signals = [HOLD, BUY, HOLD, BUY, HOLD]
    df = make_signal_df(opens, closes, signals)

    result = run_backtest(df, initial_capital=1000.0, transaction_cost=0.0)
    portfolio = result.portfolio

    expected_shares = 1000.0 / 12
    # Shares should not change after the second BUY signal is ignored.
    assert portfolio["Shares"].iloc[2] == pytest.approx(expected_shares)
    assert portfolio["Shares"].iloc[4] == pytest.approx(expected_shares)
    assert len(result.trades) == 0  # never sold, so no completed round trip


def test_sell_signal_ignored_when_flat():
    opens = [10, 10, 10]
    closes = [10, 10, 10]
    signals = [HOLD, SELL, HOLD]  # SELL with no open position
    df = make_signal_df(opens, closes, signals)

    result = run_backtest(df, initial_capital=1000.0, transaction_cost=0.0)
    assert result.portfolio["Position"].eq(0).all()
    assert result.portfolio["Cash"].eq(1000.0).all()
    assert len(result.trades) == 0


def test_run_backtest_rejects_invalid_inputs():
    df = make_signal_df([10, 10], [10, 10], [HOLD, HOLD])
    with pytest.raises(ValueError):
        run_backtest(df, initial_capital=0)
    with pytest.raises(ValueError):
        run_backtest(df, initial_capital=1000, transaction_cost=-0.01)


def test_buy_and_hold_starts_at_initial_capital_and_tracks_close():
    opens = [10, 10, 10]
    closes = [10, 12, 15]
    df = make_signal_df(opens, closes, [HOLD, HOLD, HOLD])

    bh = calculate_buy_and_hold(df, initial_capital=1000.0, transaction_cost=0.0)
    assert bh["Portfolio_Value"].iloc[0] == pytest.approx(1000.0)
    # shares = 1000 / 10 = 100; value should track close price exactly.
    assert bh["Portfolio_Value"].iloc[1] == pytest.approx(1200.0)
    assert bh["Portfolio_Value"].iloc[2] == pytest.approx(1500.0)


def test_trade_history_dataframe_shape():
    opens = [10, 10, 12, 15, 16]
    closes = [10, 10, 12, 15, 16]
    signals = [HOLD, BUY, HOLD, SELL, HOLD]
    df = make_signal_df(opens, closes, signals)

    result = run_backtest(df, initial_capital=1000.0, transaction_cost=0.0)
    history = generate_trade_history(result.trades)

    assert len(history) == 1
    assert list(history.columns) == [
        "Trade #", "Entry Date", "Entry Price", "Exit Date", "Exit Price",
        "Shares", "Profit/Loss", "Return %", "Holding Period (days)",
    ]


def test_trade_history_empty_when_no_trades():
    history = generate_trade_history([])
    assert history.empty
