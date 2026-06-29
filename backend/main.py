import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ─── Path Setup ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import download_single, TICKERS, BENCHMARKS, ALL_TICKERS

def load_data(ticker: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Loads ticker close/OHLCV data from raw CSV cache, downloading if necessary."""
    from src.data_loader import RAW_DIR
    cache_path = RAW_DIR / f"{ticker.replace('^', '')}.csv"
    
    df = pd.DataFrame()
    try:
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notnull()]
            df = df.sort_index()
            if df.empty or df.index.max() < pd.to_datetime(datetime.today() - timedelta(days=5)):
                df = pd.DataFrame()
        
        if df.empty:
            df = download_single(ticker)
            if not df.empty:
                df.to_csv(cache_path)
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
    return df

from src.feature_engineering import (
    add_sma, add_ema, add_macd, add_rsi, add_bollinger_bands,
    build_features, get_feature_columns
)
from src.risk_analysis import (
    drawdown_series, var_historical, var_parametric, var_montecarlo, cvar,
    sharpe_ratio, sortino_ratio, calmar_ratio
)
from src.portfolio import (
    max_sharpe_portfolio, min_variance_portfolio, efficient_frontier,
    monte_carlo_simulation
)
from src.models import (
    XGBoostPredictor, prepare_xy, train_test_split_ts, evaluate
)

app = FastAPI(title="FinanceIQ API", description="FastAPI Backend for Finance Analytics Dashboard")

# Enable CORS for React Frontend on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Real-Time Price Ticking Simulator & Synchronizer
# ─────────────────────────────────────────────
GLOBAL_DATA_CACHE = {}
GLOBAL_TICK_LATEST = {}

def period_to_start(period: str) -> str:
    offsets = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}
    days = offsets.get(period, 365)
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

def get_live_simulated_data(ticker: str, period: str = "1Y", n_intervals: int = 0) -> pd.DataFrame:
    """Gets cached data and appends a real-time simulated price tick to it, synchronized by n_intervals."""
    start = period_to_start(period)
    cache_key = f"{ticker}_{period}"
    
    # 1. Initialize base data if not cached
    if cache_key not in GLOBAL_DATA_CACHE:
        df = load_data(ticker, start)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        GLOBAL_DATA_CACHE[cache_key] = df.copy()
        
    df = GLOBAL_DATA_CACHE[cache_key].copy()
    
    # 2. Check if we need to apply a new shock for this interval
    tick_key = f"{ticker}_{n_intervals}"
    if tick_key not in GLOBAL_TICK_LATEST:
        if not df.empty:
            last_idx = df.index[-1]
            last_close = df.loc[last_idx, "Close"]
            
            # Apply a random walk shock: mean = 0, std = 0.06% change
            shock = 1 + np.random.normal(0, 0.0006)
            new_close = last_close * shock
            
            df.loc[last_idx, "Close"] = new_close
            if new_close > df.loc[last_idx, "High"]:
                df.loc[last_idx, "High"] = new_close
            if new_close < df.loc[last_idx, "Low"]:
                df.loc[last_idx, "Low"] = new_close
                
            GLOBAL_TICK_LATEST[tick_key] = df
            
    # If we have the ticked DataFrame for this interval, return it
    if tick_key in GLOBAL_TICK_LATEST:
        return GLOBAL_TICK_LATEST[tick_key]
        
    return df

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/api/tickers")
def get_tickers_list():
    """Get metadata of all supported tickers."""
    return {
        "tickers": [{"symbol": k, "name": v} for k, v in TICKERS.items()],
        "benchmarks": [{"symbol": k, "name": v} for k, v in BENCHMARKS.items()]
    }

@app.get("/api/live-tickers")
def get_live_tickers(n_intervals: int = 0):
    """Retrieve simulated ticking price updates for the top scrolling ribbon."""
    results = []
    for ticker in list(TICKERS.keys())[:8]:
        try:
            df = get_live_simulated_data(ticker, "2D", n_intervals)
            if len(df) >= 2:
                prev_close = float(df["Close"].iloc[-2])
                curr_price = float(df["Close"].iloc[-1])
                change = curr_price - prev_close
                pct_change = (change / prev_close) * 100
                results.append({
                    "ticker": ticker,
                    "price": round(curr_price, 2),
                    "change": round(change, 2),
                    "pctChange": round(pct_change, 2)
                })
        except Exception:
            pass
    return {"tickers": results, "timestamp": datetime.now().strftime("%H:%M:%S")}

@app.get("/api/overview")
def get_overview(period: str = "1Y", n_intervals: int = 0):
    """Calculate cumulative returns, sector returns, correlations, and KPIs for the Overview dashboard."""
    # List of tickers to track on Overview
    tickers_list = list(TICKERS.keys())[:8]
    dfs = {}
    for t in tickers_list:
        try:
            df = get_live_simulated_data(t, period, n_intervals)
            if not df.empty:
                dfs[t] = df
        except Exception:
            pass
            
    if not dfs:
        raise HTTPException(status_code=404, detail="No data available")
        
    # 1. Cumulative Return Chart
    cumulative_returns = {}
    final_returns = {}
    dates_list = []
    
    for ticker, df in dfs.items():
        if "Close" not in df.columns or df.empty:
            continue
        ret = df["Close"].pct_change().dropna()
        cum_ret = (1 + ret).cumprod()
        final_returns[ticker] = float((cum_ret.iloc[-1] - 1) * 100)
        cumulative_returns[ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in cum_ret.index],
            "values": [float(v) for v in cum_ret.values]
        }
        if not dates_list:
            dates_list = [d.strftime("%Y-%m-%d") for d in cum_ret.index]

    # 2. Sector Performance
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Technology", "META": "Technology", "JPM": "Financials",
        "V": "Financials", "BAC": "Financials", "AMZN": "Consumer",
        "TSLA": "Consumer", "JNJ": "Healthcare", "UNH": "Healthcare"
    }
    sector_rets = {}
    for t, fr in final_returns.items():
        sect = sector_map.get(t, "Other")
        sector_rets.setdefault(sect, []).append(fr)
    
    sector_performance = [
        {"sector": s, "return": float(np.mean(v))} for s, v in sector_rets.items()
    ]

    # 3. Correlation Heatmap
    ret_df = pd.DataFrame({t: dfs[t]["Close"].pct_change() for t in dfs}).dropna()
    corr = ret_df.corr()
    correlation_matrix = {
        "columns": list(corr.columns),
        "index": list(corr.index),
        "data": corr.values.tolist()
    }

    # 4. Daily Return Distributions
    distributions = {}
    for ticker, df in list(dfs.items())[:4]:
        ret = (df["Close"].pct_change().dropna() * 100).values.tolist()
        distributions[ticker] = [float(v) for v in ret]

    # 5. KPIs
    best_ticker = max(final_returns, key=final_returns.get) if final_returns else "N/A"
    worst_ticker = min(final_returns, key=final_returns.get) if final_returns else "N/A"
    avg_vol = float(np.mean([dfs[t]["Close"].pct_change().std() * np.sqrt(252) * 100 for t in dfs if not dfs[t].empty]))

    return {
        "cumulative": cumulative_returns,
        "sectors": sector_performance,
        "correlation": correlation_matrix,
        "distributions": distributions,
        "kpi": {
            "assetsTracked": len(dfs),
            "bestPerformer": best_ticker,
            "bestReturn": round(final_returns.get(best_ticker, 0), 1),
            "worstPerformer": worst_ticker,
            "worstReturn": round(final_returns.get(worst_ticker, 0), 1),
            "avgVolatility": round(avg_vol, 1)
        }
    }

@app.get("/api/technical")
def get_technical(ticker: str = "AAPL", period: str = "1Y", n_intervals: int = 0):
    """Retrieve candlestick data overlaid with technical indicators (SMA, EMA, BB, RSI, MACD)."""
    df = get_live_simulated_data(ticker, period, n_intervals).copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
    close = df["Close"]
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_MID"] = mid
    df["BB_UP"] = mid + 2 * std
    df["BB_LOW"] = mid - 2 * std
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    df.dropna(inplace=True)
    
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    
    return {
        "dates": dates,
        "ohlc": {
            "open": df["Open"].values.tolist(),
            "high": df["High"].values.tolist(),
            "low": df["Low"].values.tolist(),
            "close": df["Close"].values.tolist(),
            "volume": df["Volume"].values.tolist()
        },
        "indicators": {
            "sma20": df["SMA20"].values.tolist(),
            "sma50": df["SMA50"].values.tolist(),
            "ema9": df["EMA9"].values.tolist(),
            "bb_mid": df["BB_MID"].values.tolist(),
            "bb_up": df["BB_UP"].values.tolist(),
            "bb_low": df["BB_LOW"].values.tolist()
        },
        "rsi": df["RSI"].values.tolist(),
        "macd": {
            "macd": df["MACD"].values.tolist(),
            "signal": df["MACD_Signal"].values.tolist(),
            "hist": df["MACD_Hist"].values.tolist()
        }
    }

@app.get("/api/risk")
def get_risk(ticker: str = "AAPL", confidence: float = 95, portfolio_value: float = 1000000, n_intervals: int = 0):
    """Expose institutional-grade risk metrics, VaR distribution and rolling drawdowns/volatility."""
    conf = confidence / 100.0
    df = get_live_simulated_data(ticker, "2Y", n_intervals)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
        
    ret = df["Close"].pct_change().dropna()
    
    # Cumulative & Drawdown series
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max * 100
    
    # Value at Risk calculations
    h_var = var_historical(ret, conf)
    p_var = var_parametric(ret, conf)
    mc_var = var_montecarlo(ret, conf)
    cv = cvar(ret, conf)
    
    # Rolling volatility (63-day)
    roll_vol = ret.rolling(63).std() * np.sqrt(252) * 100
    roll_vol.dropna(inplace=True)
    
    # Basic Risk KPIs
    ann_ret = ret.mean() * 252 * 100
    ann_vol = ret.std() * np.sqrt(252) * 100
    sr = (ret.mean() * 252 - 0.05) / (ret.std() * np.sqrt(252) + 1e-9)
    max_dd = dd.min()
    cvar_val = ret[ret <= h_var].mean()
    
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
        "cumulative": cum.values.tolist(),
        "drawdown": dd.values.tolist(),
        "rollingVolatility": {
            "dates": [d.strftime("%Y-%m-%d") for d in roll_vol.index],
            "values": roll_vol.values.tolist()
        },
        "returnsDistribution": (ret * 100).values.tolist(),
        "kpis": {
            "annualizedReturn": round(ann_ret, 2),
            "annualizedVolatility": round(ann_vol, 2),
            "sharpeRatio": round(sr, 2),
            "maxDrawdown": round(max_dd, 2),
            "varHistorical": round(h_var * 100, 2),
            "varParametric": round(p_var * 100, 2),
            "varMonteCarlo": round(mc_var * 100, 2),
            "cvar": round(cv * 100, 2),
            "varDollar": round(abs(h_var) * portfolio_value, 0),
            "cvarDollar": round(abs(cv) * portfolio_value, 0)
        }
    }

@app.get("/api/portfolio")
def get_portfolio_optimization(tickers: str = "AAPL,MSFT,NVDA,JPM,AMZN,JNJ", rf: float = 5):
    """Run Markowitz portfolio optimization and return Efficient Frontier points and allocations."""
    sel_tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    if len(sel_tickers) < 2:
        raise HTTPException(status_code=400, detail="Require at least 2 tickers for optimization")
        
    rf_rate = rf / 100.0
    
    # Load historical data
    ret_dict = {}
    for t in sel_tickers:
        df = load_data(t, "2021-01-01")
        if not df.empty and "Close" in df.columns:
            ret_dict[t] = df["Close"].pct_change()
            
    if len(ret_dict) < 2:
        raise HTTPException(status_code=400, detail="Insufficient stock historical data for optimization")
        
    ret_df = pd.DataFrame(ret_dict).dropna()
    mean_ret = ret_df.mean().values
    cov_mat = ret_df.cov().values
    
    # 1. Run Optimizers
    opt_max_sharpe = max_sharpe_portfolio(mean_ret, cov_mat, rf_rate)
    opt_min_var = min_variance_portfolio(mean_ret, cov_mat)
    
    # 2. Monte Carlo Simulation (500 portfolios for faster web execution)
    sim_df = monte_carlo_simulation(mean_ret, cov_mat, n_portfolios=500, rf=rf_rate)
    sim_returns = sim_df["Return"].values
    sim_vols = sim_df["Volatility"].values
    sim_sharpe = sim_df["Sharpe"].values
    
    # 3. Efficient Frontier curve (20 portfolios for speed)
    ef_df = efficient_frontier(mean_ret, cov_mat, n_points=20, rf=rf_rate)
    ef_returns = ef_df["Return"].values
    ef_vols = ef_df["Volatility"].values
    
    # 4. Performance Backtest
    # Weights for Max Sharpe, Min Vol, and Equal weight
    w_max_sharpe = opt_max_sharpe["weights"]
    w_min_var = opt_min_var["weights"]
    w_equal = np.ones(len(sel_tickers)) / len(sel_tickers)
    
    ret_max_sharpe = (ret_df * w_max_sharpe).sum(axis=1)
    ret_min_var = (ret_df * w_min_var).sum(axis=1)
    ret_equal = (ret_df * w_equal).sum(axis=1)
    
    # Benchmark SPY
    df_spy = load_data("SPY", "2021-01-01")
    ret_spy = df_spy["Close"].pct_change().loc[ret_df.index].fillna(0)
    
    # Cumulative Performance
    cum_max_sharpe = (1 + ret_max_sharpe).cumprod()
    cum_min_var = (1 + ret_min_var).cumprod()
    cum_equal = (1 + ret_equal).cumprod()
    cum_spy = (1 + ret_spy).cumprod()
    
    dates = [d.strftime("%Y-%m-%d") for d in ret_df.index]
    
    return {
        "tickers": sel_tickers,
        "monteCarlo": {
            "returns": sim_returns.tolist(),
            "volatilities": sim_vols.tolist(),
            "sharpeRatios": sim_sharpe.tolist()
        },
        "efficientFrontier": {
            "returns": (ef_returns * 100).tolist(),
            "volatilities": (ef_vols * 100).tolist()
        },
        "maxSharpe": {
            "weights": opt_max_sharpe["weights"].tolist(),
            "return": round(opt_max_sharpe["return"] * 100, 2),
            "volatility": round(opt_max_sharpe["volatility"] * 100, 2),
            "sharpe": round(opt_max_sharpe["sharpe"], 2)
        },
        "minVariance": {
            "weights": opt_min_var["weights"].tolist(),
            "return": round(opt_min_var["return"] * 100, 2),
            "volatility": round(opt_min_var["volatility"] * 100, 2),
            "sharpe": round(opt_min_var["sharpe"], 2)
        },
        "backtest": {
            "dates": dates,
            "maxSharpe": cum_max_sharpe.values.tolist(),
            "minVariance": cum_min_var.values.tolist(),
            "equalWeight": cum_equal.values.tolist(),
            "spy": cum_spy.values.tolist()
        }
    }

@app.get("/api/ml")
def get_ml_tab_predictions(ticker: str = "AAPL"):
    """Train XGBoost model on-the-fly, returning performance metrics, predictions, and feature importances."""
    try:
        # Load historical stock data
        df = load_data(ticker, "2020-01-01")
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data found")
            
        feat_df = build_features(df)
        feat_cols = get_feature_columns(feat_df)
        
        # Prepare X, y arrays
        X, y, dates = prepare_xy(feat_df, feat_cols, "Target_Return")
        if len(X) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data to train machine learning model")
            
        # Chronological train/test split (80/20)
        X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split_ts(X, y, dates)
        
        # Train XGBoost Regressor
        model = XGBoostPredictor(ticker=ticker)
        model.fit(X_tr, y_tr, feature_names=feat_cols)
        
        # Predict on Test Set
        y_pred = model.predict(X_te)
        
        # Evaluate performance metrics
        metrics = evaluate(y_te, y_pred, "XGBoost")
        
        # Top 12 Feature Importances
        imp_df = model.feature_importance(top_n=12)
        importances = imp_df.to_dict(orient="records")
        
        # Backtest strategy comparison
        # Buy/Hold: cumulative return of close prices
        test_close = feat_df.loc[d_te, "Close"]
        bh_ret = test_close.pct_change().fillna(0)
        
        # ML Strategy: long if positive return prediction, short if negative
        signals = np.sign(y_pred)
        strat_ret = bh_ret * pd.Series(signals, index=d_te).shift(1).fillna(0)
        
        bh_cum = (1 + bh_ret).cumprod()
        strat_cum = (1 + strat_ret).cumprod()
        
        return {
            "metrics": metrics,
            "importances": importances,
            "dates": [d.strftime("%Y-%m-%d") for d in d_te],
            "actual": bh_cum.values.tolist(),
            "predicted": strat_cum.values.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ML pipeline: {str(e)}")

@app.get("/api/download-report")
def download_pdf_report(tickers: str = "AAPL,MSFT,NVDA,JPM"):
    """Generate and download a comprehensive PDF report."""
    from fastapi.responses import Response
    
    try:
        # Import PDF generator
        import sys
        from pathlib import Path
        backend_dir = Path(__file__).parent
        sys.path.insert(0, str(backend_dir))
        from pdf_generator import create_pdf_report
        
        # Parse tickers
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        
        # Generate PDF
        try:
            pdf_bytes = create_pdf_report(ticker_list)
        except UnicodeEncodeError as e:
            # This should not happen anymore after removing emojis from print statements
            print(f"[ERROR] Unicode encoding error: {e}")
            raise HTTPException(status_code=500, detail=f"Encoding error during PDF generation. Please check server logs.")
        
        # Return as downloadable file
        filename = f"FinanceIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
