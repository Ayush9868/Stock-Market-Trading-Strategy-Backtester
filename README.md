# 📈 Stock Market Trading Strategy Backtester

An interactive Python application that downloads real historical stock data and backtests a **Simple Moving Average (SMA) crossover trading strategy** against a **Buy & Hold benchmark**, complete with performance metrics, risk analysis, and an interactive Streamlit dashboard.

> ⚠️ **Educational project only.** This tool is for learning and portfolio purposes. It is **not financial advice** and should never be used to make real trading decisions. Historical backtest results do not guarantee future performance.

---

## Overview

Most "backtesting demos" online are toy scripts that plot a single equity curve and stop there. This project instead builds out a complete, modular backtesting **system**:

- Real historical data from Yahoo Finance (`yfinance`)
- A rules-based technical trading strategy with correctly-detected crossovers (no signal spam)
- A realistic trade-execution model that avoids look-ahead bias
- A proper **Buy & Hold benchmark** so the strategy's results mean something
- Return, risk, and drawdown metrics used in real quantitative finance
- A polished, interactive Streamlit + Plotly dashboard
- A unit-tested, importable Python package (`src/`) instead of one giant script

## Features

- 🔍 Choose from popular tickers (AAPL, TSLA, MSFT, AMZN, NVDA, GOOGL, META) or enter any custom symbol
- ⚙️ Adjustable moving-average window, initial capital, transaction cost, and backtest period
- 📊 Interactive price chart with BUY (green) / SELL (red) markers at exact crossover points
- 💰 Strategy vs. Buy & Hold portfolio value chart
- 📉 Drawdown chart to visualize periods of significant loss
- 📆 Year-by-year return comparison table
- 📋 Full trade-history table (entry/exit price, P&L, return %, holding period) with CSV export
- 🗃️ Downloadable processed dataset (OHLCV + signals + portfolio value)
- ✅ Robust error handling for invalid tickers, network failures, and insufficient data
- 🧪 Unit-tested core logic (strategy, backtester, metrics) using synthetic data — no network required to run tests

## Strategy

**Simple Moving Average (SMA) Crossover**

1. Compute the N-day simple moving average of the closing price (default N = 20).
2. **BUY** when the closing price crosses from at-or-below the SMA to above it.
3. **SELL** when the closing price crosses from at-or-above the SMA to below it.

Crucially, a signal fires **only on the day the crossover actually happens** — not on every day the price happens to sit above or below the average. This is a common bug in beginner backtests and is explicitly tested for (see `tests/test_strategy.py`).

## Technologies

- **Python 3.10+**
- **Pandas** / **NumPy** — data manipulation and calculations
- **yfinance** — historical market data
- **Plotly** — interactive charting
- **Streamlit** — interactive web dashboard
- **pytest** — unit testing

## How It Works

The backtester deliberately separates four stages so that no step accidentally leaks future information into an earlier one:

```
Signal Generation  →  Trade Execution  →  Portfolio Valuation  →  Performance Analysis
 (src/strategy.py)     (src/backtester.py)  (src/backtester.py)    (src/metrics.py)
```

1. **Download** ~3 years of daily OHLCV data for the selected ticker (`src/data_loader.py`).
2. **Validate & clean**: sort by date, drop duplicate dates, handle missing values, verify required columns, and confirm there's enough history for the chosen moving-average window.
3. **Generate signals** (`src/strategy.py`): compute the SMA and flag BUY/SELL only on true crossovers, using `.shift(1)` to compare against the *previous* day so no future data is used.
4. **Execute trades** (`src/backtester.py`): a signal detected at the **close** of day *t* is executed at the **open** of day *t + 1* — the earliest point a real trader could have actually acted on it. Only one position can be open at a time; a BUY is ignored while already invested, and a SELL is ignored while flat. An optional transaction cost (default 0.1%) is applied on both entry and exit.
5. **Track portfolio value** every day as `cash + shares held × closing price`.
6. **Compute the Buy & Hold benchmark**: invest the *same* starting capital in the stock on day one and hold it for the full period, so the comparison is apples-to-apples.
7. **Calculate performance & risk metrics** (`src/metrics.py`) for both the strategy and the benchmark.
8. **Render everything** in an interactive Streamlit dashboard (`app.py`).

## Performance Metrics

| Metric | Meaning |
|---|---|
| **Total Return %** | Overall percentage gain/loss over the backtest period. |
| **CAGR** | Compound Annual Growth Rate — the constant yearly return that would produce the same total growth. Lets you compare periods of different lengths fairly. |
| **Max Drawdown** | The largest peak-to-trough decline in portfolio value. A key measure of downside risk. |
| **Annualized Volatility** | How much daily returns fluctuate, scaled to a yearly figure. Higher = more erratic returns. |
| **Sharpe Ratio** | Return earned per unit of risk taken (excess return ÷ volatility, annualized). Higher is better; it lets you compare strategies with different risk levels. |
| **Win Rate** | Percentage of completed trades that were profitable. |
| **Number of Trades** | Total completed round-trip trades (buy + matching sell). |

All of these are calculated for **both** the strategy and the Buy & Hold benchmark so you can see whether the added complexity of active trading was actually worth it.

## Project Structure

```text
stock-market-backtester/
│
├── app.py                    # Streamlit dashboard (UI/orchestration only)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── .gitkeep              # Downloaded data is not committed to git
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Download + validate/clean historical data
│   ├── strategy.py           # Moving average + crossover signal generation
│   ├── backtester.py         # Trade execution, portfolio simulation, trade history
│   ├── metrics.py            # Return, risk, and drawdown metric calculations
│   └── visualization.py      # Plotly chart builders
│
└── tests/
    ├── __init__.py
    ├── test_strategy.py
    ├── test_backtester.py
    └── test_metrics.py
```

## Installation

```bash
git clone <repository-url>
cd stock-market-backtester
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), pick a stock and parameters in the sidebar, and click **Run Backtest**.

## Running the Tests

The unit tests use small synthetic datasets and **do not require an internet connection or Yahoo Finance access**:

```bash
pytest tests/ -v
```

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically and give you a shareable public URL.
5. Any time you push new commits to the connected branch, the deployed app redeploys automatically.

## Example Results

Actual results depend entirely on the ticker, moving-average window, transaction cost, and backtest period selected — the app never hard-codes or fakes numbers. In some periods the SMA crossover strategy will beat Buy & Hold (e.g., by sidestepping a prolonged downtrend); in others, especially strong, steadily-trending bull markets, Buy & Hold tends to win because the strategy incurs whipsaw losses and transaction costs from frequent crossovers. Run the app on a few different tickers and time windows to see this for yourself.

## Limitations

- Historical performance does **not** guarantee future results.
- Simple moving-average strategies can perform poorly in choppy, sideways markets ("whipsaws").
- Transaction costs and slippage are simplified; real-world execution costs (spread, market impact) may differ.
- Dividends are not included in this version.
- Yahoo Finance data may occasionally be delayed, adjusted, or temporarily unavailable.
- This is an educational backtesting project — **not** a live-trading system and **not** financial advice.

## Future Improvements

- RSI (Relative Strength Index) strategy
- MACD strategy
- Bollinger Bands strategy
- Multiple/combined moving averages (e.g., golden cross / death cross)
- Stop-loss and take-profit rules
- Position sizing and risk-based allocation
- Multi-asset portfolio backtesting
- Additional benchmarks (e.g., S&P 500 index)
- Slippage modeling
- Walk-forward and out-of-sample testing
- Automated parameter optimization

---

*Built as a portfolio project to demonstrate Python, Pandas, financial data analysis, and interactive data visualization skills.*
