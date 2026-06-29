"""
pdf_generator.py — PDF Report Generation for Dashboard
======================================================
Generates comprehensive PDF reports with charts and analytics.

Author: Data Science Internship Project
"""

import io
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import download_single, TICKERS
from src.feature_engineering import build_features, get_feature_columns
from src.models import XGBoostPredictor, prepare_xy, train_test_split_ts, evaluate
from src.risk_analysis import (
    var_summary, drawdown_series, var_historical, cvar,
    sharpe_ratio, max_drawdown
)
from src.portfolio import (
    max_sharpe_portfolio, min_variance_portfolio,
    monte_carlo_simulation, efficient_frontier
)


def create_pdf_report(tickers: list = None) -> bytes:
    """
    Generate a comprehensive PDF report with all analytics.
    
    Returns:
        bytes: PDF file content
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM']
    
    # Use ASCII-safe logging for Windows compatibility
    print(f"\n{'='*60}")
    print(f"[START] PDF REPORT GENERATION STARTED")
    print(f"{'='*60}")
    print(f"[INFO] Tickers: {', '.join(tickers)}")
    print(f"[INFO] Estimated time: 60-90 seconds")
    print(f"{'='*60}\n")
    
    # Create PDF in memory
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        print("[1/7] Creating title page...")
        # Page 1: Title Page
        _create_title_page(pdf)
        
        print("[2/7] Generating market overview charts...")
        # Page 2-3: Market Overview
        _create_market_overview_page(pdf, tickers)
        
        print("[3/7] Computing risk analytics...")
        # Page 4-5: Risk Analysis
        _create_risk_analysis_page(pdf, tickers[0])
        
        print("[4/7] Running portfolio optimization...")
        # Page 6: Portfolio Optimization
        _create_portfolio_page(pdf, tickers)
        
        print("[5/7] Training ML model and generating predictions...")
        # Page 7: ML Performance Summary
        _create_ml_summary_page(pdf, tickers[0])
        
        print("[6/7] Finalizing PDF...")
        # Metadata
        d = pdf.infodict()
        d['Title'] = 'Finance Analytics Report'
        d['Author'] = 'FinanceIQ Dashboard'
        d['Subject'] = 'Comprehensive Market Analysis'
        d['Keywords'] = 'Finance, ML, Risk, Portfolio'
        d['CreationDate'] = datetime.now()
    
    buffer.seek(0)
    print(f"\n{'='*60}")
    print(f"[COMPLETE] PDF REPORT GENERATION COMPLETE!")
    print(f"[INFO] Size: {len(buffer.getvalue()) / 1024:.1f} KB")
    print(f"{'='*60}\n")
    return buffer.getvalue()


def _create_title_page(pdf):
    """Create title page."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, 'Finance Analytics Report', 
            ha='center', va='center', fontsize=32, fontweight='bold',
            color='#58a6ff')
    
    # Subtitle
    ax.text(0.5, 0.6, 'Stock Market Intelligence & Portfolio Optimization',
            ha='center', va='center', fontsize=16, color='#8b949e')
    
    # Date
    ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            ha='center', va='center', fontsize=12, color='#8b949e')
    
    # Footer
    ax.text(0.5, 0.1, 'FinanceIQ Dashboard\nPowered by FastAPI + React',
            ha='center', va='center', fontsize=10, color='#8b949e',
            style='italic')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _create_market_overview_page(pdf, tickers):
    """Create market overview with cumulative returns."""
    from src.data_loader import build_price_matrix, compute_returns, RAW_DIR
    
    # Load cached data (faster than download)
    data = {}
    for ticker in tickers:
        try:
            cache_path = RAW_DIR / f"{ticker.replace('^', '')}.csv"
            if cache_path.exists():
                df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                df = df.sort_index()
                # Use recent 2 years of data for speed
                df = df.last('730D')
                if not df.empty:
                    data[ticker] = df
            else:
                # Fallback to download if cache doesn't exist
                df = download_single(ticker, progress=False)
                if not df.empty:
                    data[ticker] = df
        except:
            pass
    
    if not data:
        return
    
    # Build returns
    price_matrix = build_price_matrix(data)
    simple_returns, _ = compute_returns(price_matrix)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Market Overview', fontsize=16, fontweight='bold', color='#f0f6fc')
    
    # Cumulative returns
    cumulative = (1 + simple_returns).cumprod()
    for ticker in cumulative.columns:
        axes[0, 0].plot(cumulative.index, cumulative[ticker], label=ticker, linewidth=2)
    axes[0, 0].set_title('Cumulative Returns', color='#f0f6fc')
    axes[0, 0].set_ylabel('Growth of $1', color='#8b949e')
    axes[0, 0].legend(facecolor='#161b22', labelcolor='#f0f6fc')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr = simple_returns.corr()
    im = axes[0, 1].imshow(corr, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[0, 1].set_xticks(range(len(corr.columns)))
    axes[0, 1].set_yticks(range(len(corr.index)))
    axes[0, 1].set_xticklabels(corr.columns, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(corr.index)
    axes[0, 1].set_title('Correlation Matrix', color='#f0f6fc')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Performance metrics
    ann_ret = simple_returns.mean() * 252 * 100
    ann_vol = simple_returns.std() * np.sqrt(252) * 100
    sharpe = (ann_ret - 5) / ann_vol
    
    metrics_df = pd.DataFrame({
        'Return (%)': ann_ret,
        'Vol (%)': ann_vol,
        'Sharpe': sharpe
    }).sort_values('Sharpe', ascending=False)
    
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table = axes[1, 0].table(cellText=metrics_df.round(2).values,
                             colLabels=metrics_df.columns,
                             rowLabels=metrics_df.index,
                             cellLoc='right',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 0].set_title('Performance Metrics', color='#f0f6fc')
    
    # Return distribution
    for ticker in simple_returns.columns:
        axes[1, 1].hist(simple_returns[ticker] * 100, bins=50, alpha=0.5, label=ticker)
    axes[1, 1].set_title('Return Distributions', color='#f0f6fc')
    axes[1, 1].set_xlabel('Daily Return (%)', color='#8b949e')
    axes[1, 1].set_ylabel('Frequency', color='#8b949e')
    axes[1, 1].legend(facecolor='#161b22', labelcolor='#f0f6fc')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _create_risk_analysis_page(pdf, ticker):
    """Create risk analysis page."""
    from src.data_loader import RAW_DIR
    
    try:
        # Try to load from cache first (faster)
        cache_path = RAW_DIR / f"{ticker.replace('^', '')}.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            df = df.sort_index()
            # Use recent 2 years for speed
            df = df.last('730D')
        else:
            df = download_single(ticker, progress=False)
            
        if df.empty:
            return
        
        returns = df['Close'].pct_change().dropna()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'Risk Analysis — {ticker}', fontsize=16, fontweight='bold', color='#f0f6fc')
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        dd = drawdown_series(returns) * 100
        
        axes[0, 0].plot(cumulative.index, cumulative.values, color='#58a6ff', linewidth=2)
        axes[0, 0].set_title('Cumulative Return', color='#f0f6fc')
        axes[0, 0].set_ylabel('Growth of $1', color='#8b949e')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].fill_between(dd.index, 0, dd.values, color='#f85149', alpha=0.6)
        axes[0, 1].plot(dd.index, dd.values, color='#f85149', linewidth=1)
        axes[0, 1].set_title('Drawdown', color='#f0f6fc')
        axes[0, 1].set_ylabel('Drawdown (%)', color='#8b949e')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Return distribution with VaR
        axes[1, 0].hist(returns * 100, bins=80, density=True, color='#1f6feb', alpha=0.7)
        var95 = var_historical(returns, 0.95) * 100
        var99 = var_historical(returns, 0.99) * 100
        cvar95 = cvar(returns, 0.95) * 100
        
        axes[1, 0].axvline(var95, color='#f85149', linestyle='--', linewidth=2, label=f'VaR 95%: {var95:.2f}%')
        axes[1, 0].axvline(cvar95, color='#ff7b72', linestyle=':', linewidth=2, label=f'CVaR 95%: {cvar95:.2f}%')
        axes[1, 0].set_title('Return Distribution', color='#f0f6fc')
        axes[1, 0].set_xlabel('Daily Return (%)', color='#8b949e')
        axes[1, 0].legend(facecolor='#161b22', labelcolor='#f0f6fc')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk metrics table
        metrics = {
            'Metric': ['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR 95%', 'CVaR 95%'],
            'Value': [
                f"{returns.mean() * 252 * 100:.2f}%",
                f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                f"{sharpe_ratio(returns):.3f}",
                f"{max_drawdown(returns) * 100:.2f}%",
                f"{var95:.2f}%",
                f"{cvar95:.2f}%"
            ]
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=list(zip(metrics['Metric'], metrics['Value'])),
                                colLabels=['Metric', 'Value'],
                                cellLoc='left',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        axes[1, 1].set_title('Risk Metrics', color='#f0f6fc')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating risk page: {e}")


def _create_portfolio_page(pdf, tickers):
    """Create portfolio optimization page."""
    from src.data_loader import build_price_matrix, compute_returns, RAW_DIR
    
    try:
        # Load cached data (faster)
        data = {}
        for ticker in tickers:
            try:
                cache_path = RAW_DIR / f"{ticker.replace('^', '')}.csv"
                if cache_path.exists():
                    df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                    df = df.sort_index()
                    # Use recent 2 years for speed
                    df = df.last('730D')
                    if not df.empty:
                        data[ticker] = df
                else:
                    df = download_single(ticker, progress=False)
                    if not df.empty:
                        data[ticker] = df
            except:
                pass
        
        if len(data) < 2:
            return
        
        price_matrix = build_price_matrix(data)
        simple_returns, _ = compute_returns(price_matrix)
        recent_returns = simple_returns[simple_returns.index >= '2022-01-01']
        
        mu = recent_returns.mean().values
        cov = recent_returns.cov().values
        
        # Optimize
        max_sharpe = max_sharpe_portfolio(mu, cov, rf=0.05)
        min_var = min_variance_portfolio(mu, cov)
        
        # Reduced Monte Carlo for speed (500 instead of 1000)
        mc_df = monte_carlo_simulation(mu, cov, n_portfolios=500, rf=0.05)
        ef_df = efficient_frontier(mu, cov, n_points=30, rf=0.05)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Portfolio Optimization', fontsize=16, fontweight='bold', color='#f0f6fc')
        
        # Efficient Frontier
        sc = axes[0].scatter(mc_df['Volatility']*100, mc_df['Return']*100,
                            c=mc_df['Sharpe'], cmap='plasma', alpha=0.6, s=10)
        axes[0].plot(ef_df['Volatility']*100, ef_df['Return']*100,
                    color='white', linewidth=2, label='Efficient Frontier')
        axes[0].scatter(max_sharpe['volatility']*100, max_sharpe['return']*100,
                       color='#f1c40f', s=200, marker='*', edgecolors='white', linewidths=2,
                       label=f"Max Sharpe (SR={max_sharpe['sharpe']:.2f})")
        axes[0].scatter(min_var['volatility']*100, min_var['return']*100,
                       color='#58a6ff', s=150, marker='o', edgecolors='white', linewidths=2,
                       label=f"Min Var")
        axes[0].set_xlabel('Volatility (%)', color='#8b949e')
        axes[0].set_ylabel('Return (%)', color='#8b949e')
        axes[0].set_title('Efficient Frontier', color='#f0f6fc')
        axes[0].legend(facecolor='#161b22', labelcolor='#f0f6fc')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[0], label='Sharpe Ratio')
        
        # Portfolio weights
        width = 0.35
        x = np.arange(len(tickers))
        axes[1].bar(x - width/2, max_sharpe['weights']*100, width, label='Max Sharpe', color='#f1c40f')
        axes[1].bar(x + width/2, min_var['weights']*100, width, label='Min Variance', color='#58a6ff')
        axes[1].set_xlabel('Assets', color='#8b949e')
        axes[1].set_ylabel('Weight (%)', color='#8b949e')
        axes[1].set_title('Portfolio Allocations', color='#f0f6fc')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tickers, rotation=45)
        axes[1].legend(facecolor='#161b22', labelcolor='#f0f6fc')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating portfolio page: {e}")


def _create_ml_summary_page(pdf, ticker):
    """Create ML performance summary page."""
    from src.data_loader import RAW_DIR
    
    try:
        # Try to load from cache first (faster)
        cache_path = RAW_DIR / f"{ticker.replace('^', '')}.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            df = df.sort_index()
            # Use only 2 years of recent data for faster ML training
            df = df.last('730D')
        else:
            df = download_single(ticker, progress=False)
            
        if df.empty:
            return
        
        feature_df = build_features(df, forward_days=1, verbose=False)
        feature_cols = get_feature_columns(feature_df)
        
        X, y, dates = prepare_xy(feature_df, feature_cols, 'Target_Return')
        
        # Use larger test split for faster training (30% instead of 20%)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split_ts(X, y, dates, test_ratio=0.3)
        
        # Train XGBoost with reduced complexity for speed
        fast_params = {
            "n_estimators": 50,        # Reduced from 500
            "max_depth": 4,            # Reduced from 6
            "learning_rate": 0.1,      # Increased from 0.03 for faster convergence
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        xgb_model = XGBoostPredictor(params=fast_params, ticker=ticker)
        xgb_model.fit(X_train, y_train, feature_names=feature_cols)
        y_pred = xgb_model.predict(X_test)
        
        metrics = evaluate(y_test, y_pred, 'XGBoost')
        importance_df = xgb_model.feature_importance(top_n=10)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'ML Performance — {ticker}', fontsize=16, fontweight='bold', color='#f0f6fc')
        
        # Predictions
        axes[0, 0].plot(dates_test, y_test*100, color='#8b949e', label='Actual', linewidth=1, alpha=0.8)
        axes[0, 0].plot(dates_test, y_pred*100, color='#58a6ff', label='Predicted', linewidth=2)
        axes[0, 0].set_title('Predictions vs Actual', color='#f0f6fc')
        axes[0, 0].set_ylabel('Return (%)', color='#8b949e')
        axes[0, 0].legend(facecolor='#161b22', labelcolor='#f0f6fc')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter
        axes[0, 1].scatter(y_test*100, y_pred*100, alpha=0.4, color='#58a6ff', s=10)
        lim = max(abs(y_test).max(), abs(y_pred).max()) * 100 * 1.1
        axes[0, 1].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        axes[0, 1].set_xlim(-lim, lim)
        axes[0, 1].set_ylim(-lim, lim)
        axes[0, 1].set_xlabel('Actual (%)', color='#8b949e')
        axes[0, 1].set_ylabel('Predicted (%)', color='#8b949e')
        axes[0, 1].set_title('Scatter Plot', color='#f0f6fc')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance
        axes[1, 0].barh(importance_df['Feature'], importance_df['Importance'], color='#3fb950')
        axes[1, 0].set_xlabel('Importance', color='#8b949e')
        axes[1, 0].set_title('Top 10 Features', color='#f0f6fc')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Metrics table
        metrics_data = [
            ['RMSE', f"{metrics['RMSE']:.6f}"],
            ['MAE', f"{metrics['MAE']:.6f}"],
            ['MAPE', f"{metrics['MAPE (%)']}%"],
            ['R²', f"{metrics['R²']:.4f}"],
            ['Dir. Accuracy', f"{metrics['Dir. Acc. (%)']}%"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=metrics_data,
                                colLabels=['Metric', 'Value'],
                                cellLoc='left',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        axes[1, 1].set_title('Model Performance', color='#f0f6fc')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating ML page: {e}")
