"""
app.py
=======

Streamlit dashboard for the Stock Market Trading Strategy Backtester.

This file only handles UI/layout and orchestration; all data downloading,
signal generation, backtesting, and metric calculations live in `src/`.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.backtester import calculate_buy_and_hold, generate_trade_history, run_backtest
from src.data_loader import DataLoadError, download_stock_data, ensure_sufficient_history
from src.metrics import (
    calculate_annual_returns,
    calculate_drawdown_series,
    calculate_performance_metrics,
)
from src.strategy import generate_signals
from src.visualization import plot_drawdown_chart, plot_portfolio_chart, plot_price_chart

st.set_page_config(
    page_title="Stock Market Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
)

POPULAR_TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "Custom..."]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Backtest Configuration")

    choice = st.sidebar.selectbox("Stock", POPULAR_TICKERS, index=0)
    if choice == "Custom...":
        ticker = st.sidebar.text_input("Enter ticker symbol", value="").strip().upper()
    else:
        ticker = choice

    ma_window = st.sidebar.slider("Moving Average Window (days)", min_value=5, max_value=200, value=20)
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=500.0
    )
    transaction_cost_pct = st.sidebar.number_input(
        "Transaction Cost (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05
    )
    years = st.sidebar.slider("Backtest Period (years)", min_value=1, max_value=10, value=3)

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ Educational project only. Not financial advice. "
        "Past performance does not guarantee future results."
    )

    return {
        "ticker": ticker,
        "ma_window": ma_window,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost_pct / 100.0,
        "years": years,
        "run_clicked": run_clicked,
    }


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
def render_header() -> None:
    st.title("📈 Stock Market Trading Strategy Backtester")
    st.markdown(
        "#### Evaluate a Moving Average Crossover Strategy Against Buy-and-Hold"
    )
    st.markdown("---")


# --------------------------------------------------------------------------
# KPI section
# --------------------------------------------------------------------------
def render_kpis(
    current_price: float,
    current_sma: float,
    strategy_metrics: dict,
    benchmark_metrics: dict,
) -> None:
    st.subheader("Market Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price", f"${current_price:,.2f}")
    c2.metric("Moving Average", f"${current_sma:,.2f}" if not np.isnan(current_sma) else "N/A")
    c3.metric("Strategy Return", f"{strategy_metrics['total_return_pct']:.2f}%")
    c4.metric("Buy & Hold Return", f"{benchmark_metrics['total_return_pct']:.2f}%")
    c5.metric("Final Portfolio Value", f"${strategy_metrics['final_value']:,.2f}")
    c6.metric("Number of Trades", f"{strategy_metrics['num_trades']}")


def render_metrics_tables(strategy_metrics: dict, benchmark_metrics: dict) -> None:
    st.subheader("Performance & Risk Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strategy**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Initial Capital", "Final Portfolio Value", "Total Profit/Loss",
                        "Total Return %", "CAGR %", "Max Drawdown %",
                        "Annualized Volatility %", "Sharpe Ratio",
                        "Number of Trades", "Winning Trades", "Losing Trades", "Win Rate %",
                    ],
                    "Value": [
                        f"${strategy_metrics['initial_capital']:,.2f}",
                        f"${strategy_metrics['final_value']:,.2f}",
                        f"${strategy_metrics['profit_loss']:,.2f}",
                        f"{strategy_metrics['total_return_pct']:.2f}%",
                        f"{strategy_metrics['cagr_pct']:.2f}%",
                        f"{strategy_metrics['max_drawdown_pct']:.2f}%",
                        f"{strategy_metrics['annualized_volatility_pct']:.2f}%",
                        f"{strategy_metrics['sharpe_ratio']:.2f}",
                        strategy_metrics["num_trades"],
                        strategy_metrics["winning_trades"],
                        strategy_metrics["losing_trades"],
                        f"{strategy_metrics['win_rate']:.2f}%",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("**Buy & Hold Benchmark**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Initial Capital", "Final Portfolio Value", "Total Profit/Loss",
                        "Total Return %", "CAGR %", "Max Drawdown %", "Annualized Volatility %",
                        "Sharpe Ratio",
                    ],
                    "Value": [
                        f"${benchmark_metrics['initial_capital']:,.2f}",
                        f"${benchmark_metrics['final_value']:,.2f}",
                        f"${benchmark_metrics['profit_loss']:,.2f}",
                        f"{benchmark_metrics['total_return_pct']:.2f}%",
                        f"{benchmark_metrics['cagr_pct']:.2f}%",
                        f"{benchmark_metrics['max_drawdown_pct']:.2f}%",
                        f"{benchmark_metrics['annualized_volatility_pct']:.2f}%",
                        f"{benchmark_metrics['sharpe_ratio']:.2f}",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_strategy_explanation() -> None:
    with st.expander("ℹ️ How the Strategy Works"):
        st.markdown(
            """
1. **Download historical stock data** for the selected ticker (daily Open, High, Low, Close, Volume).
2. **Calculate the moving average** — the average closing price over the last *N* days, which smooths out day-to-day noise.
3. **Detect crossovers** — the moment the price moves from below the moving average to above it (or vice versa).
4. **Generate BUY/SELL signals** only on the day a crossover actually happens, not every day the price stays above or below the average.
5. **Simulate trades** — a BUY signal invests all available cash the next trading day at the opening price; a SELL signal exits the entire position the next trading day at the opening price. This next-day-open execution avoids using information that wasn't actually available yet.
6. **Track portfolio value** every day: cash + (shares held × current price).
7. **Compare against Buy & Hold** — what would have happened if you had simply bought the stock on day one and held it, using the same starting capital.
8. **Calculate performance and risk metrics** (returns, CAGR, drawdown, volatility, Sharpe ratio) so the two approaches can be compared fairly.
            """
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    render_header()
    config = render_sidebar()

    if "result" not in st.session_state:
        st.session_state["result"] = None

    if config["run_clicked"]:
        if not config["ticker"]:
            st.error("Please select or enter a valid ticker symbol.")
            st.stop()
        try:
            with st.spinner(f"Downloading data for {config['ticker']}..."):
                raw = download_stock_data(config["ticker"], years=config["years"])
                ensure_sufficient_history(raw, config["ma_window"])

            signals_df = generate_signals(raw, window=config["ma_window"])
            sma_col = f"SMA_{config['ma_window']}"

            backtest = run_backtest(
                signals_df,
                initial_capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
            )
            benchmark = calculate_buy_and_hold(
                signals_df,
                initial_capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
            )

            st.session_state["result"] = {
                "ticker": config["ticker"],
                "sma_col": sma_col,
                "signals_df": signals_df,
                "backtest": backtest,
                "benchmark": benchmark,
                "config": config,
            }
        except DataLoadError as exc:
            st.error(f"⚠️ {exc}")
            st.session_state["result"] = None
            st.stop()
        except Exception as exc:  # final safety net - never show a raw traceback
            st.error(f"⚠️ Something went wrong while running the backtest: {exc}")
            st.session_state["result"] = None
            st.stop()

    result = st.session_state["result"]
    if result is None:
        st.info("👈 Configure your backtest in the sidebar and click **Run Backtest** to get started.")
        render_strategy_explanation()
        return

    ticker = result["ticker"]
    sma_col = result["sma_col"]
    signals_df = result["signals_df"]
    backtest = result["backtest"]
    benchmark = result["benchmark"]
    config = result["config"]

    portfolio = backtest.portfolio
    strategy_metrics = calculate_performance_metrics(
        portfolio["Portfolio_Value"], portfolio["Daily_Return"],
        config["initial_capital"], backtest.trades,
    )
    benchmark_metrics = calculate_performance_metrics(
        benchmark["Portfolio_Value"], benchmark["Daily_Return"], config["initial_capital"],
    )

    current_price = float(signals_df["Close"].iloc[-1])
    current_sma = float(signals_df[sma_col].iloc[-1])

    render_kpis(current_price, current_sma, strategy_metrics, benchmark_metrics)
    st.markdown("---")

    st.plotly_chart(plot_price_chart(signals_df, sma_col, ticker), use_container_width=True)

    st.plotly_chart(
        plot_portfolio_chart(portfolio["Portfolio_Value"], benchmark["Portfolio_Value"]),
        use_container_width=True,
    )

    strategy_dd = calculate_drawdown_series(portfolio["Portfolio_Value"])
    benchmark_dd = calculate_drawdown_series(benchmark["Portfolio_Value"])
    st.plotly_chart(plot_drawdown_chart(strategy_dd, benchmark_dd), use_container_width=True)

    st.markdown("---")
    render_metrics_tables(strategy_metrics, benchmark_metrics)

    st.markdown("---")
    st.subheader("Annual Performance")
    annual = calculate_annual_returns(portfolio["Portfolio_Value"], benchmark["Portfolio_Value"])
    st.dataframe(annual, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Trade History")
    trade_history = generate_trade_history(backtest.trades)
    st.dataframe(trade_history, hide_index=True, use_container_width=True)
    if not trade_history.empty:
        st.download_button(
            "⬇️ Download Trade History (CSV)",
            trade_history.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_trade_history.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Underlying Data")
    data_table = signals_df.copy()
    data_table["Position"] = portfolio["Position"]
    data_table["Portfolio_Value"] = portfolio["Portfolio_Value"]
    st.dataframe(
        data_table[["Open", "High", "Low", "Close", "Volume", sma_col, "Signal", "Position", "Portfolio_Value"]],
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download Processed Dataset (CSV)",
        data_table.to_csv().encode("utf-8"),
        file_name=f"{ticker}_processed_data.csv",
        mime="text/csv",
    )

    st.markdown("---")
    render_strategy_explanation()

    st.caption(
        "This tool is for educational purposes only and does not constitute financial advice. "
        "Historical backtest results do not guarantee future performance."
    )


if __name__ == "__main__":
    main()
