"""
generate_reports.py — One-Click Report Generator
================================================
Generates all charts and visualizations for the finance project.

Usage:
    python generate_reports.py

Output:
    - Saves 15-20 PNG charts to reports/figures/
    - Prints summary of generated files

Author: Data Science Internship Project
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_all, build_price_matrix, compute_returns, TICKERS
from src.feature_engineering import build_features, get_feature_columns
from src.models import (XGBoostPredictor, RandomForestPredictor, prepare_xy, 
                         train_test_split_ts, plot_predictions, plot_feature_importance)
from src.risk_analysis import (full_risk_report, plot_drawdown, plot_return_distribution,
                                plot_rolling_risk, plot_correlation_heatmap)
from src.portfolio import (max_sharpe_portfolio, min_variance_portfolio, equal_weight_portfolio,
                            efficient_frontier, monte_carlo_simulation, 
                            plot_efficient_frontier, plot_portfolio_weights, plot_portfolio_performance,
                            backtest_portfolio)

print("=" * 80)
print("  📊 FINANCE PROJECT REPORT GENERATOR")
print("=" * 80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nThis will generate 15-20 professional charts for your portfolio.")
print("Estimated time: 3-5 minutes\n")

# ============================================================================
# STEP 1: Data Acquisition
# ============================================================================

print("📥 [1/5] Downloading market data...")
tickers_list = list(TICKERS.keys())
data = download_all(tickers_list, force=False)

stock_data = {t: data[t] for t in tickers_list if t in data}
price_matrix = build_price_matrix(stock_data)
simple_returns, log_returns = compute_returns(price_matrix)

print(f"  ✅ Downloaded {len(data)} tickers")
print(f"  ✅ Date range: {price_matrix.index.min().date()} → {price_matrix.index.max().date()}")

# ============================================================================
# STEP 2: Risk Analysis Charts
# ============================================================================

print("\n⚠️  [2/5] Generating risk analysis charts...")

# Get SPY as benchmark
spy_returns = simple_returns['SPY'] if 'SPY' in simple_returns.columns else simple_returns['JPM']

# Generate risk charts for top 3 stocks
top_stocks = ['AAPL', 'MSFT', 'NVDA']
for ticker in top_stocks:
    if ticker in simple_returns.columns:
        returns = simple_returns[ticker]
        
        # Drawdown chart
        plot_drawdown(returns, ticker=ticker, save=True)
        
        # Return distribution
        plot_return_distribution(returns, ticker=ticker, save=True)
        
        # Rolling risk metrics
        plot_rolling_risk(returns, ticker=ticker, save=True)

# Correlation heatmap
plot_correlation_heatmap(simple_returns, save=True)

print(f"  ✅ Generated risk charts for {len(top_stocks)} stocks")

# ============================================================================
# STEP 3: Machine Learning Models
# ============================================================================

print("\n🤖 [3/5] Training ML models and generating predictions...")

# Train on AAPL
ticker = 'AAPL'
df = data[ticker].copy()

feature_df = build_features(df, forward_days=1, verbose=False)
feature_cols = get_feature_columns(feature_df)

X, y, dates = prepare_xy(feature_df, feature_cols, 'Target_Return')
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split_ts(X, y, dates, test_ratio=0.2)

# Train XGBoost
xgb_model = XGBoostPredictor(ticker=ticker)
xgb_model.fit(X_train, y_train, X_test, y_test, feature_names=feature_cols)
y_pred_xgb = xgb_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestPredictor(ticker=ticker)
rf_model.fit(X_train, y_train, feature_names=feature_cols)
y_pred_rf = rf_model.predict(X_test)

# Generate plots
plot_predictions(y_test, y_pred_xgb, dates_test, model_name='XGBoost', ticker=ticker, save=True)
plot_predictions(y_test, y_pred_rf, dates_test, model_name='RandomForest', ticker=ticker, save=True)

# Feature importance
importance_df = xgb_model.feature_importance(top_n=20)
plot_feature_importance(importance_df, model_name='XGBoost', ticker=ticker, save=True)

print(f"  ✅ Trained XGBoost and Random Forest models for {ticker}")

# ============================================================================
# STEP 4: Portfolio Optimization
# ============================================================================

print("\n💼 [4/5] Running portfolio optimization...")

# Use recent 3 years of data for optimization
recent_returns = simple_returns[simple_returns.index >= '2022-01-01']
opt_tickers = list(recent_returns.columns)[:8]  # Top 8 stocks
recent_returns = recent_returns[opt_tickers]

mu = recent_returns.mean().values
cov = recent_returns.cov().values

# Optimize portfolios
max_sharpe = max_sharpe_portfolio(mu, cov, rf=0.05)
min_var = min_variance_portfolio(mu, cov)
equal_w = equal_weight_portfolio(mu, cov)

portfolios = {
    'Max Sharpe': max_sharpe,
    'Min Variance': min_var,
    'Equal Weight': equal_w
}


# Monte Carlo simulation
print("  🎲 Running Monte Carlo simulation (5,000 portfolios)...")
mc_df = monte_carlo_simulation(mu, cov, n_portfolios=5000, rf=0.05)

# Efficient Frontier
print("  📈 Computing Efficient Frontier...")
ef_df = efficient_frontier(mu, cov, n_points=100, rf=0.05)

# Generate plots
plot_efficient_frontier(mc_df, ef_df, 
                       special_portfolios={k: {'return': v['return'], 
                                              'volatility': v['volatility'],
                                              'sharpe': v['sharpe']}
                                         for k, v in portfolios.items()},
                       save=True)

plot_portfolio_weights(portfolios, opt_tickers, save=True)

# Backtest portfolios
perf_dict = {}
for name, pf in portfolios.items():
    perf = backtest_portfolio(pf['weights'], recent_returns)
    perf_dict[name] = perf

plot_portfolio_performance(perf_dict, save=True)

print(f"  ✅ Portfolio optimization complete with {len(portfolios)} strategies")

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================

print("\n📊 [5/5] Generating summary statistics...")

# Count generated files
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
generated_files = list(FIG_DIR.glob("*.png"))

print("\n" + "=" * 80)
print("  ✅ REPORT GENERATION COMPLETE!")
print("=" * 80)

print(f"\n📁 Generated {len(generated_files)} chart files:")
print(f"   Location: {FIG_DIR}\n")

# Categorize files
risk_files = [f for f in generated_files if any(x in f.name for x in ['drawdown', 'dist', 'rolling', 'correlation'])]
ml_files = [f for f in generated_files if any(x in f.name for x in ['predictions', 'feature_imp', 'strategy'])]
portfolio_files = [f for f in generated_files if any(x in f.name for x in ['frontier', 'weights', 'performance'])]

print("📊 Risk Analysis Charts:")
for f in risk_files:
    print(f"   • {f.name}")

print(f"\n🤖 Machine Learning Charts:")
for f in ml_files:
    print(f"   • {f.name}")

print(f"\n💼 Portfolio Optimization Charts:")
for f in portfolio_files:
    print(f"   • {f.name}")

print("\n" + "=" * 80)
print("  🎉 All reports ready for your portfolio presentation!")
print("=" * 80)
print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext steps:")
print("  1. View charts in: reports/figures/")
print("  2. Include in presentations or documents")
print("  3. Run dashboard for interactive analysis: python run_dashboard.py")
