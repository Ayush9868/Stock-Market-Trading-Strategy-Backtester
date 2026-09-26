"""
Stock Market Trading Strategy Backtester
==========================================

A modular backtesting engine for evaluating a Simple Moving Average (SMA)
crossover trading strategy against a Buy & Hold benchmark, using real
historical data pulled from Yahoo Finance via `yfinance`.

Modules
-------
data_loader     : Download and validate historical OHLCV data.
strategy        : Moving-average calculation and BUY/SELL signal generation.
backtester      : Trade simulation, portfolio valuation, and trade history.
metrics         : Performance and risk metric calculations.
visualization   : Plotly chart builders used by the Streamlit dashboard.

This package is intended for educational purposes only and does not
constitute financial advice.
"""

__version__ = "1.0.0"
