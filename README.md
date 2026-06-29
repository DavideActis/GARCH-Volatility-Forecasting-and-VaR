# GARCH(1,1) Volatility Forecasting and Value-at-Risk

A Python implementation of a **GARCH(1,1)** volatility model for forecasting market risk and estimating the **10-day 99% Value-at-Risk (VaR)** of an equity portfolio.

The project uses historical **AAPL** daily returns and compares two innovation assumptions:

- Normal distribution
- Student-t distribution

The objective is to illustrate how conditional volatility models can be applied to short-horizon market risk measurement.

---

# Project Overview

Financial returns exhibit **volatility clustering**, meaning periods of high volatility tend to be followed by high volatility, while tranquil periods tend to persist over time.

To capture this behaviour, the project estimates a **GARCH(1,1)** model using historical daily log returns.

The fitted model is then used to:

- estimate conditional volatility;
- forecast future variance over a 10-day horizon;
- compute the 10-day 99% Value-at-Risk (VaR);
- compare Normal and Student-t innovation assumptions.

---

# Methodology

The workflow consists of the following steps:

1. Download historical AAPL prices
2. Compute daily log returns
3. Estimate two GARCH(1,1) models
    - Normal innovations
    - Student-t innovations
4. Forecast conditional volatility
5. Forecast the next 10 trading days of variance
6. Estimate the 10-day 99% Value-at-Risk
7. Compare the resulting risk measures

---

# Model Specification

Daily log returns are modeled as

rₜ = μ + εₜ

where

εₜ = σₜ · zₜ

and conditional variance follows the GARCH(1,1) process

σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁

where:

- **ω** is the long-run variance parameter;
- **α** measures the impact of recent market shocks;
- **β** captures volatility persistence over time.

Two innovation distributions are considered:

- Normal (Gaussian)
- Student-t

The Student-t specification is particularly useful because financial returns typically exhibit **fat tails**, leading to larger Value-at-Risk estimates during periods of elevated market uncertainty.

---

# Project Structure

```
.
├── garch_var.py
├── requirements.txt
├── README.md
├── LICENSE
├── aapl_price_series.png
├── aapl_log_returns.png
├── conditional_volatility_normal.png
├── conditional_volatility_student_t.png
└── var_comparison.png
```

---

# Results

The model estimates both conditional volatility and portfolio Value-at-Risk.

Example output:

| Model | 10-Day 99% VaR |
|-------|---------------:|
| Normal GARCH | ~$75,900 |
| Student-t GARCH | ~$80,300 |

As expected, the Student-t specification produces a larger VaR because it captures the heavier tails of financial return distributions.

---

# Visualizations

## AAPL Price Series

![Price](https://raw.githubusercontent.com/DavideActis/GARCH-Volatility-Forecasting-and-VaR/main/aapl_price_series.png)

---

## Daily Log Returns

![Returns](https://raw.githubusercontent.com/DavideActis/GARCH-Volatility-Forecasting-and-VaR/main/aapl_log_returns.png)

---

## Conditional Volatility (Normal)

![Normal](https://raw.githubusercontent.com/DavideActis/GARCH-Volatility-Forecasting-and-VaR/main/conditional_volatility_normal.png)

---

## Conditional Volatility (Student-t)

![Student](https://raw.githubusercontent.com/DavideActis/GARCH-Volatility-Forecasting-and-VaR/main/conditional_volatility_student_t.png)

---

## 10-Day Value-at-Risk Comparison

![VaR](https://raw.githubusercontent.com/DavideActis/GARCH-Volatility-Forecasting-and-VaR/main/var_comparison.png)

# How to Run

Install the required packages

```bash
pip install -r requirements.txt
```

Run the script

```bash
python garch_var.py
```

The program automatically:

- downloads historical market data;
- estimates both GARCH models;
- computes the 10-day 99% Value-at-Risk;
- generates all figures.

---

# Requirements

- Python 3.10+
- NumPy
- Pandas
- SciPy
- Matplotlib
- arch
- pandas-datareader
- yfinance

---

# Applications

This project demonstrates practical applications of:

- Quantitative Risk Management
- Market Risk Analytics
- Financial Econometrics
- Volatility Forecasting
- Value-at-Risk Estimation
- Time Series Modeling

---

# License

Released under the MIT License.
