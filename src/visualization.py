"""
visualization.py
==================

Builds the interactive Plotly figures used by the Streamlit dashboard:

* Price chart with the moving average and BUY/SELL markers.
* Strategy vs. Buy & Hold portfolio value chart.
* Drawdown chart.

Kept separate from `app.py` so the chart styling can be reused or unit
tested independently of Streamlit.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .strategy import BUY, SELL

BUY_COLOR = "#16a34a"   # green
SELL_COLOR = "#dc2626"  # red
PRICE_COLOR = "#2563eb"  # blue
SMA_COLOR = "#f59e0b"    # amber
STRATEGY_COLOR = "#2563eb"
BENCHMARK_COLOR = "#94a3b8"

CHART_LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40),
    font=dict(family="Inter, Arial, sans-serif", size=13),
)


def plot_price_chart(df: pd.DataFrame, sma_col: str, ticker: str) -> go.Figure:
    """
    Line chart of closing price + moving average, with BUY/SELL markers at
    the exact days a crossover signal fired.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close Price",
            mode="lines",
            line=dict(color=PRICE_COLOR, width=1.6),
            hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[sma_col],
            name=sma_col.replace("_", " "),
            mode="lines",
            line=dict(color=SMA_COLOR, width=1.6, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>" + sma_col + ": $%{y:.2f}<extra></extra>",
        )
    )

    buys = df[df["Signal"] == BUY]
    sells = df[df["Signal"] == SELL]

    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Close"],
            name="BUY Signal",
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color=BUY_COLOR,
                        line=dict(width=1, color="white")),
            hovertemplate="BUY<br>%{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Close"],
            name="SELL Signal",
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color=SELL_COLOR,
                        line=dict(width=1, color="white")),
            hovertemplate="SELL<br>%{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{ticker} Price & {sma_col.replace('_', ' ')} with Trade Signals",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        **CHART_LAYOUT,
    )
    return fig


def plot_portfolio_chart(strategy_value: pd.Series, benchmark_value: pd.Series) -> go.Figure:
    """Strategy vs. Buy & Hold portfolio value over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value,
            name="Strategy",
            mode="lines",
            line=dict(color=STRATEGY_COLOR, width=2),
            hovertemplate="%{x|%b %d, %Y}<br>Strategy: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value,
            name="Buy & Hold",
            mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}<br>Buy & Hold: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Portfolio Value: Strategy vs. Buy & Hold",
        yaxis_title="Portfolio Value (USD)",
        xaxis_title="Date",
        **CHART_LAYOUT,
    )
    return fig


def plot_drawdown_chart(strategy_dd: pd.Series, benchmark_dd: pd.Series) -> go.Figure:
    """Drawdown-over-time chart for the strategy and the benchmark."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_dd.index,
            y=strategy_dd * 100,
            name="Strategy Drawdown",
            mode="lines",
            fill="tozeroy",
            line=dict(color=STRATEGY_COLOR, width=1.5),
            hovertemplate="%{x|%b %d, %Y}<br>Strategy: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_dd.index,
            y=benchmark_dd * 100,
            name="Buy & Hold Drawdown",
            mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}<br>Buy & Hold: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Drawdown Over Time",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        **CHART_LAYOUT,
    )
    return fig
