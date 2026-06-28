# 🏦 Advanced Finance Data Science Project
## Stock Market Intelligence, Risk Analytics & Portfolio Optimization
### Internship Portfolio Project

---

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Dash-purple?logo=plotly)](https://dash.plotly.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)](https://xgboost.readthedocs.io/)

---

## 📋 Project Overview

A **production-grade, end-to-end financial data science project** designed to demonstrate applied skills across the full ML lifecycle — from raw data acquisition to interactive dashboard deployment.

This project tackles a **real-world problem**: _Can we use machine learning and quantitative analysis to make better investment decisions?_

---

## 🎯 Key Features

| Module | Description | Technologies |
|--------|-------------|-------------|
| **Data Pipeline** | Downloads 5+ years of OHLCV data for 12 S&P 500 stocks | yfinance, pandas |
| **EDA** | Correlations, distributions, volume/volatility patterns | Plotly, seaborn |
| **Feature Engineering** | 25+ technical indicators (RSI, MACD, BB, ATR, OBV, ...) | Custom Python |
| **ML Prediction** | XGBoost, Random Forest, ARIMA, LSTM + backtesting | scikit-learn, XGBoost, TF |
| **Risk Analytics** | VaR, CVaR, Sharpe, Beta, Drawdown, Stress Testing | scipy, numpy |
| **Portfolio Optimization** | Markowitz + Monte Carlo (10k simulations) | scipy, cvxpy |
| **Dashboard** | 4-tab interactive app with live charts | Plotly Dash |

---

## 🗂️ Project Structure

```
Real-world Data Project (Finance)/
│
├── 📓 notebooks/
│   └── finance_analysis.ipynb    ← Master notebook (run this first!)
│
├── 🐍 src/
│   ├── data_loader.py            ← yfinance data pipeline
│   ├── feature_engineering.py   ← 25+ technical indicators
│   ├── models.py                 ← XGBoost, RF, ARIMA, LSTM
│   ├── risk_analysis.py          ← VaR, CVaR, Sharpe, Beta, Drawdown
│   └── portfolio.py              ← Markowitz + Monte Carlo
│
├── 🖥️ dashboard/
│   └── app.py                    ← Interactive Plotly Dash app
│
├── 📊 data/
│   ├── raw/                      ← Downloaded CSV files (auto-created)
│   └── processed/                ← Engineered feature files
│
├── 📈 reports/
│   └── figures/                  ← Saved charts (PNG)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Create Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If TensorFlow causes issues, install without it first — the project runs with XGBoost + ARIMA only.

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/finance_analysis.ipynb
```

Run all cells from top to bottom. The notebook will:
- Download fresh market data from Yahoo Finance (~2 min)
- Engineer features and train all models (~10 min with LSTM)
- Generate all charts and risk reports
- Print an executive summary

### 4. Launch Interactive Dashboard

```bash
python dashboard/app.py
```

Then open **http://localhost:8050** in your browser.

---

## 📊 Models & Performance

| Model | Type | Strengths |
|-------|------|-----------|
| **XGBoost** | Gradient Boosting | Best overall accuracy, feature importance |
| **Random Forest** | Ensemble | Robust, interpretable baseline |
| **ARIMA** | Time Series | Classical benchmark, confidence intervals |
| **LSTM** | Deep Learning | Sequence patterns, multi-step forecasting |

**Evaluation Metrics:**
- RMSE, MAE, MAPE — magnitude accuracy
- **Directional Accuracy** — % of days where up/down is predicted correctly
- Sharpe, Calmar — strategy quality after transaction costs

---

## ⚠️ Risk Analytics

Portfolio risk metrics computed for every asset:

- **VaR (95%, 99%)** — Historical, Parametric, Monte Carlo
- **CVaR / Expected Shortfall** — Expected loss beyond VaR
- **Sharpe Ratio** — Risk-adjusted return
- **Sortino Ratio** — Downside-risk-adjusted return
- **Calmar Ratio** — Return over Max Drawdown
- **Beta / Alpha** — CAPM decomposition vs S&P 500
- **Maximum Drawdown** — Peak-to-trough loss + recovery analysis
- **Stress Testing** — Performance during 6 major market crises

---

## 💼 Portfolio Optimization

Four strategies compared:

| Strategy | Description |
|----------|-------------|
| **Max Sharpe** | Maximises risk-adjusted return |
| **Min Variance** | Minimises total portfolio risk |
| **Risk Parity** | Equal risk contribution per asset |
| **Equal Weight** | 1/N benchmark (naive diversification) |

Monte Carlo simulation generates 10,000 random portfolios to visualise the full risk-return space and validate the Efficient Frontier.

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|-----|---------|
| **📊 Overview** | Cumulative returns, correlation matrix, sector heatmap |
| **📈 Technical Analysis** | Candlestick + BB/SMA/EMA/RSI/MACD for any stock |
| **⚠️ Risk Analytics** | VaR gauge, drawdown chart, rolling metrics |
| **💼 Portfolio** | Efficient Frontier, weight allocation, backtest |

---

## 🔑 Key Findings

1. **XGBoost achieves ~55–58% directional accuracy** on unseen test data, significantly above the 50% random baseline
2. **Technology stocks are highly correlated** (ρ > 0.75), limiting diversification benefits within the sector
3. **Max Sharpe portfolio** consistently outperforms equal-weight on a risk-adjusted basis
4. **LSTM** captures sequential patterns but requires careful regularisation to avoid overfitting
5. **TSLA** exhibits the highest volatility and deepest drawdowns — significant tail risk
6. **Rolling VaR** shows volatility clustering — risk is time-varying, not constant

---

## 🛠️ Technical Stack

```
Data          : yfinance, pandas, numpy
ML Models     : scikit-learn, xgboost, tensorflow, pmdarima, statsmodels
Visualisation : plotly, matplotlib, seaborn
Dashboard     : dash, dash-bootstrap-components
Optimization  : scipy, cvxpy
Statistics    : scipy.stats
```

---

## 📚 References & Methodology

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
- Sharpe, W. (1964). "Capital Asset Prices." *Journal of Finance*
- XGBoost: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- LSTM: Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Technical Analysis: Murphy, J. (1999). *Technical Analysis of Financial Markets*

---

*Built for Data Science Internship — 2024/2025*
