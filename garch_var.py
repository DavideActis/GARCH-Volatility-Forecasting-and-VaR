#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:37:12 2026

@author: davideactis
"""

#!/usr/bin/env python3
"""
GARCH(1,1) Volatility Forecasting and Value-at-Risk Estimation

This script estimates GARCH(1,1) models on AAPL daily log returns and computes
a 10-day 99% Value-at-Risk (VaR) for a long equity position.

The project compares two innovation assumptions:
- Normal innovations
- Student-t innovations

It illustrates how conditional volatility models can be used for market risk
measurement and short-horizon risk forecasting.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from arch import arch_model


TICKER = "AAPL"
STOOQ_TICKER = "AAPL.US"

START_DATE = pd.Timestamp("2005-01-01")
AS_OF_DATE = pd.Timestamp("2025-12-31")

POSITION_VALUE = 1_000_000
HORIZON_DAYS = 10
ALPHA_VAR = 0.01


def load_price_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Load historical adjusted closing prices."""
    try:
        from pandas_datareader import data as pdr

        prices = pdr.DataReader(STOOQ_TICKER, "stooq", start_date, end_date)
        prices = prices.sort_index()
        prices = prices.rename(columns={"Close": "Adj Close"})

        if "Adj Close" not in prices.columns:
            raise ValueError("Missing adjusted close price column.")

        return prices[["Adj Close"]]

    except Exception as stooq_error:
        print(f"[Info] Stooq download failed: {stooq_error}")
        print("[Info] Trying Yahoo Finance...")

    try:
        import yfinance as yf

        prices = yf.download(
            TICKER,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False
        )

        if prices is None or prices.empty or "Adj Close" not in prices.columns:
            raise ValueError("Yahoo Finance returned empty data.")

        prices = prices.sort_index()
        return prices[["Adj Close"]]

    except Exception as yahoo_error:
        raise RuntimeError(f"Data download failed: {yahoo_error}")


def compute_log_returns(prices: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from adjusted closing prices."""
    returns = np.log(prices["Adj Close"]).diff().dropna()
    returns.name = "log_return"
    return returns


def fit_garch_var(
    returns_pct: pd.Series,
    distribution: str,
    position_value: float,
    horizon_days: int,
    alpha: float
) -> dict:
    """Fit GARCH(1,1) and compute multi-day Value-at-Risk."""
    model = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist=distribution
    )

    result = model.fit(disp="off")
    forecast = result.forecast(horizon=horizon_days, reindex=False)

    daily_variance_pct = forecast.variance.values[-1, :]
    daily_variance_dec = daily_variance_pct / (100.0 ** 2)

    sigma_horizon = float(np.sqrt(np.sum(daily_variance_dec)))

    mu_daily_pct = float(result.params.get("mu", 0.0))
    mu_horizon = horizon_days * (mu_daily_pct / 100.0)

    if distribution.lower() in ["normal", "gaussian"]:
        quantile = norm.ppf(alpha)
    elif distribution.lower() in ["t", "studentst", "student"]:
        nu = float(result.params["nu"])
        raw_quantile = student_t.ppf(alpha, df=nu)
        quantile = raw_quantile * np.sqrt((nu - 2.0) / nu)
    else:
        raise ValueError("distribution must be 'normal' or 't'.")

    return_quantile = mu_horizon + quantile * sigma_horizon
    var_dollars = max(0.0, -position_value * return_quantile)

    return {
        "model_result": result,
        "daily_variance_forecast": daily_variance_dec,
        "sigma_horizon": sigma_horizon,
        "mu_horizon": mu_horizon,
        "quantile": quantile,
        "return_quantile": return_quantile,
        "var_dollars": var_dollars,
    }


def plot_price_series(prices: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices["Adj Close"], color="steelblue")
    plt.title(f"{TICKER} Adjusted Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("aapl_price_series.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_log_returns(returns: pd.Series) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(returns.index, returns, color="steelblue", linewidth=0.8)
    plt.title(f"{TICKER} Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig("aapl_log_returns.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_conditional_volatility(result, label: str) -> None:
    conditional_volatility = result.conditional_volatility / 100.0

    plt.figure(figsize=(10, 5))
    plt.plot(conditional_volatility.index, conditional_volatility, color="steelblue")
    plt.title(f"GARCH(1,1) Conditional Volatility - {label}")
    plt.xlabel("Date")
    plt.ylabel("Conditional Volatility")
    plt.tight_layout()
    plt.savefig(f"conditional_volatility_{label.lower()}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_var_comparison(normal_output: dict, student_t_output: dict) -> None:
    labels = ["Normal GARCH", "Student-t GARCH"]
    values = [
        normal_output["var_dollars"],
        student_t_output["var_dollars"]
    ]

    colors = ["#1f77b4", "#d62728"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)

    plt.title("10-Day 99% Value-at-Risk Comparison")
    plt.ylabel("Value-at-Risk ($)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"${value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    difference = values[1] - values[0]
    difference_pct = difference / values[0] * 100

    plt.figtext(
        0.5,
        -0.03,
        f"Student-t VaR is ${difference:,.0f} higher than Normal VaR "
        f"({difference_pct:.1f}% difference).",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_summary(label: str, output: dict, last_observation_date: pd.Timestamp) -> None:
    result = output["model_result"]
    params = result.params

    print("\n" + "=" * 72)
    print(f"{label} innovations | GARCH(1,1) on {TICKER} returns")
    print(f"Last observation date: {last_observation_date.date()}")
    print("=" * 72)

    for parameter in ["mu", "omega", "alpha[1]", "beta[1]"]:
        if parameter in params.index:
            print(f"{parameter:>8s}: {params[parameter]: .6f}")

    if "nu" in params.index:
        print(f"{'nu':>8s}: {params['nu']: .6f}")

    if "alpha[1]" in params.index and "beta[1]" in params.index:
        persistence = params["alpha[1]"] + params["beta[1]"]
        print(f"\nVolatility persistence alpha + beta: {persistence:.6f}")

    print("\n10-day 99% VaR estimate:")
    print(f"Expected return over horizon: {output['mu_horizon']:.6f}")
    print(f"Conditional volatility:       {output['sigma_horizon']:.6f}")
    print(f"1% return quantile:           {output['return_quantile']:.6f}")
    print(f"10-day 99% VaR:               ${output['var_dollars']:,.2f}")


def main() -> None:
    prices = load_price_data(START_DATE, AS_OF_DATE)
    prices = prices.loc[:AS_OF_DATE].dropna()

    if prices.empty:
        raise RuntimeError("No price data available.")

    last_observation_date = prices.index[-1]
    print(f"[Info] Last available observation: {last_observation_date.date()}")

    returns = compute_log_returns(prices)
    returns_pct = 100.0 * returns

    normal_output = fit_garch_var(
        returns_pct,
        distribution="normal",
        position_value=POSITION_VALUE,
        horizon_days=HORIZON_DAYS,
        alpha=ALPHA_VAR
    )

    student_t_output = fit_garch_var(
        returns_pct,
        distribution="t",
        position_value=POSITION_VALUE,
        horizon_days=HORIZON_DAYS,
        alpha=ALPHA_VAR
    )

    print_summary("Normal", normal_output, last_observation_date)
    print_summary("Student-t", student_t_output, last_observation_date)

    print("\n" + "-" * 72)
    print("VaR comparison")
    print(f"Normal GARCH 10-day 99% VaR:    ${normal_output['var_dollars']:,.2f}")
    print(f"Student-t GARCH 10-day 99% VaR: ${student_t_output['var_dollars']:,.2f}")
    print("-" * 72)

    plot_price_series(prices)
    plot_log_returns(returns)
    plot_conditional_volatility(normal_output["model_result"], label="normal")
    plot_conditional_volatility(student_t_output["model_result"], label="student_t")
    plot_var_comparison(normal_output, student_t_output)


if __name__ == "__main__":
    main()