"""
data_loader.py — Financial Data Acquisition Pipeline
=====================================================
Downloads, cleans, and stores OHLCV data for a diversified
portfolio of S&P 500 stocks + market benchmarks using yfinance.

Author  : Finance DS Internship Project
Version : 1.0
"""

import os
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Universe of stocks — diversified across sectors
TICKERS = {
    # Technology
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "NVDA":  "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "META":  "Meta Platforms",
    # Financials
    "JPM":   "JPMorgan Chase",
    "V":     "Visa Inc.",
    "BAC":   "Bank of America",
    # Consumer / E-commerce
    "AMZN":  "Amazon.com",
    "TSLA":  "Tesla Inc.",
    # Healthcare
    "JNJ":   "Johnson & Johnson",
    "UNH":   "UnitedHealth Group",
}

# Market benchmarks
BENCHMARKS = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "NASDAQ-100 ETF",
    "^VIX": "CBOE Volatility Index",
    "^TNX": "10-Year Treasury Yield",
}

ALL_TICKERS  = list(TICKERS.keys()) + list(BENCHMARKS.keys())
START_DATE   = "2019-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# Download Functions
# ─────────────────────────────────────────────

def download_single(ticker: str, start: str = START_DATE, end: str = END_DATE,
                    progress: bool = False) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and return a clean DataFrame."""
    logger.info(f"Downloading {ticker} ...")
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=progress)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df[["Open", "Close", "High", "Low", "Volume"]].copy()
        df.dropna(how="all", inplace=True)

        # Forward-fill any isolated NaNs (weekends / holidays)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        logger.info(f"  [OK]  {ticker}: {len(df)} rows | {df.index.min().date()} -> {df.index.max().date()}")
        return df

    except Exception as e:
        logger.error(f"  [ERROR]  Failed to download {ticker}: {e}")
        return pd.DataFrame()


def download_all(tickers: List[str] = None, start: str = START_DATE,
                 end: str = END_DATE, force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download data for all tickers and save raw CSVs.

    Parameters
    ----------
    tickers : list of ticker symbols (default: all configured tickers)
    start   : start date string (YYYY-MM-DD)
    end     : end date string   (YYYY-MM-DD)
    force   : re-download even if cached file exists

    Returns
    -------
    dict of {ticker: DataFrame}
    """
    if tickers is None:
        tickers = ALL_TICKERS

    data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_path = RAW_DIR / f"{ticker.replace('^','')}.csv"

        if cache_path.exists() and not force:
            logger.info(f"Loading {ticker} from cache: {cache_path.name}")
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        else:
            df = download_single(ticker, start=start, end=end)
            if not df.empty:
                df.to_csv(cache_path)

        if not df.empty:
            data[ticker] = df

    logger.info(f"\n[SUCCESS] Downloaded {len(data)}/{len(tickers)} tickers successfully.")
    return data


# ─────────────────────────────────────────────
# Aggregation & Processing
# ─────────────────────────────────────────────

def build_price_matrix(data: Dict[str, pd.DataFrame],
                        price_col: str = "Close") -> pd.DataFrame:
    """
    Combine individual ticker data into a single wide-format price matrix.
    Rows = trading days, Columns = tickers.
    """
    frames = {ticker: df[price_col] for ticker, df in data.items()}
    price_df = pd.DataFrame(frames)
    price_df.index.name = "Date"

    # Align to common trading calendar (inner join — only days all exist)
    price_df.dropna(how="all", inplace=True)
    price_df.ffill(inplace=True)
    price_df.bfill(inplace=True)
    return price_df


def compute_returns(price_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute simple and log returns from a price matrix.

    Returns
    -------
    simple_returns, log_returns
    """
    simple_returns = price_matrix.pct_change().dropna()
    log_returns    = np.log(price_matrix / price_matrix.shift(1)).dropna()
    return simple_returns, log_returns


def get_market_cap_weights(tickers: List[str]) -> Dict[str, float]:
    """Fetch approximate market-cap weights for portfolio construction."""
    caps = {}
    for t in tickers:
        try:
            info   = yf.Ticker(t).fast_info
            cap    = getattr(info, "market_cap", None) or 1e9
            caps[t] = cap
        except Exception:
            caps[t] = 1e9

    total = sum(caps.values())
    return {t: v / total for t, v in caps.items()}


def load_processed(name: str = "price_matrix") -> pd.DataFrame:
    """Load a previously saved processed CSV from data/processed/."""
    path = PROC_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def save_processed(df: pd.DataFrame, name: str):
    """Save a DataFrame to data/processed/."""
    path = PROC_DIR / f"{name}.csv"
    df.to_csv(path)
    logger.info(f"Saved processed data -> {path}")


# ─────────────────────────────────────────────
# Summary Stats
# ─────────────────────────────────────────────

def dataset_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print a summary table of downloaded datasets."""
    rows = []
    for ticker, df in data.items():
        name = {**TICKERS, **BENCHMARKS}.get(ticker, ticker)
        rows.append({
            "Ticker":      ticker,
            "Name":        name,
            "Start":       df.index.min().date(),
            "End":         df.index.max().date(),
            "Trading Days": len(df),
            "Avg Close":   f"${df['Close'].mean():.2f}",
            "Missing (%)": f"{df.isnull().mean().mean() * 100:.2f}%",
        })
    summary = pd.DataFrame(rows).set_index("Ticker")
    return summary


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  📥  Financial Data Acquisition Pipeline")
    print("=" * 60)

    # Download all tickers
    data = download_all(force=False)

    # Build price matrix
    stock_tickers = list(TICKERS.keys())
    stock_data    = {t: data[t] for t in stock_tickers if t in data}
    price_matrix  = build_price_matrix(stock_data)

    # Compute returns
    simple_ret, log_ret = compute_returns(price_matrix)

    # Save processed
    save_processed(price_matrix, "price_matrix")
    save_processed(simple_ret,   "simple_returns")
    save_processed(log_ret,      "log_returns")

    # Print summary
    print("\n📋 Dataset Summary:")
    print(dataset_summary(data).to_string())
    print(f"\n💰 Price Matrix Shape : {price_matrix.shape}")
    print(f"📈 Returns Shape      : {simple_ret.shape}")
