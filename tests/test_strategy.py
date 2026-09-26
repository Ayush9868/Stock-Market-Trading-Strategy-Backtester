"""Unit tests for src/strategy.py using small synthetic datasets."""

import pandas as pd
import pytest

from src.strategy import BUY, HOLD, SELL, calculate_moving_average, generate_signals


def make_df(closes):
    dates = pd.date_range("2023-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes},
        index=dates,
    )


def test_moving_average_values():
    closes = [1, 2, 3, 4, 5, 6]
    df = make_df(closes)
    sma = calculate_moving_average(df, window=3)

    assert sma.iloc[:2].isna().all()
    assert sma.iloc[2] == pytest.approx(2.0)
    assert sma.iloc[3] == pytest.approx(3.0)
    assert sma.iloc[4] == pytest.approx(4.0)
    assert sma.iloc[5] == pytest.approx(5.0)


def test_moving_average_rejects_small_window():
    df = make_df([1, 2, 3])
    with pytest.raises(ValueError):
        calculate_moving_average(df, window=1)


def test_signals_fire_only_on_crossover_not_every_day():
    # Flat -> up plateau -> down plateau, window=3.
    closes = [10, 10, 10, 10, 20, 20, 20, 20, 5, 5, 5, 5]
    df = make_df(closes)
    result = generate_signals(df, window=3)

    signals = result["Signal"].tolist()

    # Exactly one BUY and one SELL should fire, not one per day above/below.
    assert signals.count(BUY) == 1
    assert signals.count(SELL) == 1

    assert signals[4] == BUY   # first day price closes above a warmed-up SMA
    assert signals[8] == SELL  # first day price closes below a warmed-up SMA

    # No signal should fire before the SMA has warmed up.
    assert result["Signal"].iloc[:2].eq(HOLD).all()


def test_signals_have_no_lookahead():
    # A signal on day t must be derivable purely from Close[t] and SMA[t]
    # plus their day t-1 values - never from anything later than day t.
    closes = [10, 10, 10, 30, 30, 5, 5]
    df = make_df(closes)
    result = generate_signals(df, window=3)

    for i in range(len(result)):
        truncated = generate_signals(df.iloc[: i + 1], window=3)
        assert result["Signal"].iloc[i] == truncated["Signal"].iloc[-1]


def test_no_signal_when_data_shorter_than_window():
    df = make_df([1, 2, 3])
    result = generate_signals(df, window=5)
    assert result["Signal"].eq(HOLD).all()
    assert result["SMA_5"].isna().all()
