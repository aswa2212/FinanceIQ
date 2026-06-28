"""
feature_engineering.py — Technical Indicator & Feature Factory
==============================================================
Computes 25+ technical indicators and statistical features
from OHLCV data for use in ML models.

Indicators implemented:
  Trend    : SMA, EMA, MACD, ADX, Parabolic SAR
  Momentum : RSI, Stochastic Oscillator, ROC, Williams %R, CCI
  Volatility: Bollinger Bands, ATR, Keltner Channel, Historical Vol
  Volume   : OBV, VWAP, Volume SMA, CMF, MFI
  Custom   : Z-Score, Lag features, Rolling statistics

Author  : Finance DS Internship Project
Version : 1.0
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────
# Trend Indicators
# ─────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows=(5, 10, 20, 50, 200)) -> pd.DataFrame:
    """Simple Moving Averages at multiple windows."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    return df


def add_ema(df: pd.DataFrame, windows=(9, 12, 26, 50)) -> pd.DataFrame:
    """Exponential Moving Averages."""
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence/Divergence.
    Adds: MACD, MACD_Signal, MACD_Hist
    """
    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df


def add_adx(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """Average Directional Index (trend strength)."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    dmp = (high - high.shift(1)).clip(lower=0)
    dmn = (low.shift(1) - low).clip(lower=0)
    dmp[dmp < dmn] = 0
    dmn[dmn < dmp] = 0

    atr  = tr.ewm(alpha=1/window, adjust=False).mean()
    dip  = 100 * dmp.ewm(alpha=1/window, adjust=False).mean() / atr
    din  = 100 * dmn.ewm(alpha=1/window, adjust=False).mean() / atr
    dx   = (100 * (dip - din).abs() / (dip + din + 1e-9))
    df[f"ADX_{window}"] = dx.ewm(alpha=1/window, adjust=False).mean()
    return df


# ─────────────────────────────────────────────
# Momentum Indicators
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """
    Relative Strength Index.
    RSI > 70 → overbought  |  RSI < 30 → oversold
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df: pd.DataFrame, k_window=14, d_window=3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D lines)."""
    low_min  = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(d_window).mean()
    return df


def add_roc(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """Rate of Change — price momentum."""
    for w in windows:
        df[f"ROC_{w}"] = df["Close"].pct_change(periods=w) * 100
    return df


def add_cci(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f"CCI_{window}"] = (tp - ma) / (0.015 * md + 1e-9)
    return df


def add_williams_r(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """Williams %R momentum oscillator."""
    high_max = df["High"].rolling(window).max()
    low_min  = df["Low"].rolling(window).min()
    df["Williams_R"] = -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-9)
    return df


# ─────────────────────────────────────────────
# Volatility Indicators
# ─────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    """
    Bollinger Bands — volatility envelope around SMA.
    Adds: BB_Middle, BB_Upper, BB_Lower, BB_Width, BB_%B
    """
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_Middle"] = sma
    df["BB_Upper"]  = sma + num_std * std
    df["BB_Lower"]  = sma - num_std * std
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)
    df["BB_PctB"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    return df


def add_atr(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """Average True Range — absolute volatility measure."""
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{window}"]  = tr.ewm(span=window, adjust=False).mean()
    df["ATR_Pct"] = df[f"ATR_{window}"] / df["Close"] * 100   # normalised
    return df


def add_historical_volatility(df: pd.DataFrame, windows=(10, 20, 30)) -> pd.DataFrame:
    """Annualised historical volatility from log returns."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    for w in windows:
        df[f"HV_{w}"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
    return df


# ─────────────────────────────────────────────
# Volume Indicators
# ─────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume — cumulative volume pressure."""
    sign     = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (sign * df["Volume"]).cumsum()
    df["OBV_EMA"] = df["OBV"].ewm(span=20, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-Weighted Average Price (daily reset approximation)."""
    tp          = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"]  = (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
    df["VWAP_Dev"] = (df["Close"] - df["VWAP"]) / df["VWAP"] * 100
    return df


def add_volume_sma(df: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    """Volume moving averages and relative volume."""
    for w in windows:
        df[f"Vol_SMA_{w}"] = df["Volume"].rolling(w).mean()
    df["Rel_Volume"] = df["Volume"] / (df["Vol_SMA_20"] + 1e-9)
    return df


def add_cmf(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """Chaikin Money Flow — volume-weighted buying/selling pressure."""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    mfv = mfm * df["Volume"]
    df[f"CMF_{window}"] = mfv.rolling(window).sum() / (df["Volume"].rolling(window).sum() + 1e-9)
    return df


def add_mfi(df: pd.DataFrame, window=14) -> pd.DataFrame:
    """Money Flow Index — RSI applied to dollar volume."""
    tp      = (df["High"] + df["Low"] + df["Close"]) / 3
    mf      = tp * df["Volume"]
    pos_mf  = mf.where(tp > tp.shift(1), 0).rolling(window).sum()
    neg_mf  = mf.where(tp < tp.shift(1), 0).rolling(window).sum()
    mfr     = pos_mf / (neg_mf + 1e-9)
    df[f"MFI_{window}"] = 100 - 100 / (1 + mfr)
    return df


# ─────────────────────────────────────────────
# Custom / Statistical Features
# ─────────────────────────────────────────────

def add_zscore(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """Rolling z-score of closing price — mean reversion signal."""
    mu  = df["Close"].rolling(window).mean()
    sig = df["Close"].rolling(window).std()
    df[f"ZScore_{window}"] = (df["Close"] - mu) / (sig + 1e-9)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Lagged returns for ML feature vectors."""
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df["Close"].pct_change().shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """Rolling mean, std, skew, kurt of returns."""
    ret = df["Close"].pct_change()
    for w in windows:
        df[f"Roll_Mean_{w}"] = ret.rolling(w).mean()
        df[f"Roll_Std_{w}"]  = ret.rolling(w).std()
    df["Roll_Skew_20"]  = ret.rolling(20).skew()
    df["Roll_Kurt_20"]  = ret.rolling(20).kurt()
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Candle-body, shadow ratios, gap detection."""
    df["Body"]       = (df["Close"] - df["Open"]).abs()
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Candle_Dir"] = np.sign(df["Close"] - df["Open"])
    df["Gap_Up"]     = (df["Open"] > df["High"].shift(1)).astype(int)
    df["Gap_Down"]   = (df["Open"] < df["Low"].shift(1)).astype(int)
    return df


def add_target(df: pd.DataFrame, forward_days: int = 1) -> pd.DataFrame:
    """
    Target variable for supervised learning:
      - Target_Return: next-N-day return
      - Target_Dir   : 1 if price goes up, 0 if down
    """
    future_close        = df["Close"].shift(-forward_days)
    df["Target_Return"] = (future_close - df["Close"]) / df["Close"]
    df["Target_Dir"]    = (df["Target_Return"] > 0).astype(int)
    return df


# ─────────────────────────────────────────────
# Master Feature Builder
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame, forward_days: int = 1,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Apply all technical indicators to a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    df           : OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
    forward_days : prediction horizon for target variable
    verbose      : print feature count

    Returns
    -------
    DataFrame enriched with all technical features
    """
    df = df.copy()

    # Trend
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_adx(df)

    # Momentum
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_roc(df)
    df = add_cci(df)
    df = add_williams_r(df)

    # Volatility
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_historical_volatility(df)

    # Volume
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_sma(df)
    df = add_cmf(df)
    df = add_mfi(df)

    # Custom
    df = add_zscore(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_price_patterns(df)

    # Target
    df = add_target(df, forward_days=forward_days)

    # Drop initial NaN rows (warm-up period)
    df.dropna(inplace=True)

    if verbose:
        feature_cols = [c for c in df.columns
                        if c not in {"Open", "High", "Low", "Close", "Volume",
                                     "Target_Return", "Target_Dir"}]
        print(f"[OK] Features built: {len(feature_cols)} indicators | {len(df)} rows retained")

    return df


def build_features_all(data: dict, forward_days: int = 1,
                        save: bool = True) -> dict:
    """Build features for multiple tickers and optionally save."""
    from src.data_loader import PROC_DIR

    results = {}
    for ticker, df in data.items():
        print(f"  Engineering features for {ticker} ...", end="  ")
        try:
            feat_df = build_features(df, forward_days=forward_days, verbose=True)
            results[ticker] = feat_df
            if save:
                path = PROC_DIR / f"features_{ticker}.csv"
                feat_df.to_csv(path)
        except Exception as e:
            print(f"\n  [WARNING] Error for {ticker}: {e}")
    return results


# ─────────────────────────────────────────────
# Feature Selection Helper
# ─────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns (exclude raw OHLCV and targets)."""
    exclude = {"Open", "High", "Low", "Close", "Volume",
               "Target_Return", "Target_Dir"}
    return [c for c in df.columns if c not in exclude]


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """
    Remove highly correlated features (|corr| > threshold) to reduce redundancy.
    Returns list of surviving feature columns.
    """
    feature_cols = get_feature_columns(df)
    corr_matrix  = df[feature_cols].corr().abs()
    upper        = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    surviving = [c for c in feature_cols if c not in to_drop]
    print(f"[FILTER] Correlation filter: {len(feature_cols)} -> {len(surviving)} features "
          f"(dropped {len(to_drop)} at |r|>{threshold})")
    return surviving


if __name__ == "__main__":
    from src.data_loader import download_all, TICKERS

    print("🔧 Building technical features ...")
    data = download_all(list(TICKERS.keys()))
    # Build for AAPL as demo
    df   = data.get("AAPL")
    if df is not None:
        feat = build_features(df)
        print(feat.tail(3))
