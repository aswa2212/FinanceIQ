"""
models.py — Machine Learning Prediction Suite
==============================================
Four complementary models for stock price direction & magnitude prediction:

  1. XGBoost Regressor   — gradient-boosted trees (primary workhorse)
  2. Random Forest       — ensemble baseline
  3. ARIMA/SARIMA        — classical time-series benchmark
  4. LSTM                — deep learning sequence model

Includes:
  - Walk-forward cross-validation (backtesting)
  - Unified evaluation metrics (RMSE, MAE, MAPE, Directional Accuracy)
  - Feature importance analysis
  - Signal generation (Buy/Sell/Hold)

Author  : Finance DS Internship Project
Version : 1.0
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
FIG_DIR      = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             model_name: str = "") -> Dict[str, float]:
    """
    Compute comprehensive regression + directional metrics.

    Returns
    -------
    dict with RMSE, MAE, MAPE, R², Directional Accuracy
    """
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true + 1e-9, y_pred + 1e-9) * 100
    r2    = r2_score(y_true, y_pred)

    # Directional accuracy (did we predict up/down correctly?)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    metrics = {
        "Model":          model_name,
        "RMSE":           round(rmse, 6),
        "MAE":            round(mae, 6),
        "MAPE (%)":       round(mape, 4),
        "R²":             round(r2, 4),
        "Dir. Acc. (%)":  round(dir_acc, 2),
    }
    return metrics


def metrics_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """Format all model metrics into a comparison table."""
    df = pd.DataFrame(all_metrics).set_index("Model")
    return df


# ─────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame, feature_cols: list,
               target_col: str = "Target_Return") -> Tuple:
    """
    Split feature DataFrame into X (features) and y (target),
    removing NaN rows and aligning indices.
    """
    clean   = df[feature_cols + [target_col]].dropna()
    X       = clean[feature_cols].values
    y       = clean[target_col].values
    dates   = clean.index
    return X, y, dates


def train_test_split_ts(X: np.ndarray, y: np.ndarray, dates,
                         test_ratio: float = 0.2):
    """
    Chronological train/test split (no random shuffle — essential for time series).
    Returns X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    n     = len(X)
    split = int(n * (1 - test_ratio))
    return (X[:split], X[split:], y[:split], y[split:],
            dates[:split], dates[split:])


# ─────────────────────────────────────────────
# 1. XGBoost Model
# ─────────────────────────────────────────────

class XGBoostPredictor:
    """
    Gradient-boosted tree model for return prediction.
    Includes feature importance ranking and SHAP-ready output.
    """

    DEFAULT_PARAMS = {
        "n_estimators":      500,
        "max_depth":         6,
        "learning_rate":     0.03,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "min_child_weight":  5,
        "gamma":             0.1,
        "objective":         "reg:squarederror",
        "tree_method":       "hist",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }

    def __init__(self, params: dict = None, ticker: str = ""):
        self.params  = params or self.DEFAULT_PARAMS
        self.ticker  = ticker
        self.model   = xgb.XGBRegressor(**self.params)
        self.scaler  = RobustScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feature_names: list = None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X_train)

        eval_set = [(X_scaled, y_train)]
        if X_val is not None:
            eval_set.append((self.scaler.transform(X_val), y_val))

        self.model.fit(
            X_scaled, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if not self.is_fitted or self.feature_names is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "Feature":   self.feature_names,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(top_n)
        return imp

    def save(self, path: str = None):
        path = path or str(MODEL_DIR / f"xgb_{self.ticker}.pkl")
        joblib.dump(self, path)
        print(f"💾 XGBoost model saved: {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ─────────────────────────────────────────────
# 2. Random Forest Model
# ─────────────────────────────────────────────

class RandomForestPredictor:
    """Random Forest ensemble — strong baseline model."""

    DEFAULT_PARAMS = {
        "n_estimators":   300,
        "max_depth":      10,
        "min_samples_split": 10,
        "min_samples_leaf":  5,
        "max_features":  "sqrt",
        "n_jobs":        -1,
        "random_state":  RANDOM_STATE,
    }

    def __init__(self, params: dict = None, ticker: str = ""):
        self.params  = params or self.DEFAULT_PARAMS
        self.ticker  = ticker
        self.model   = RandomForestRegressor(**self.params)
        self.scaler  = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list = None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if not self.is_fitted or self.feature_names is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "Feature":    self.feature_names,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(top_n)
        return imp


# ─────────────────────────────────────────────
# 3. ARIMA Model
# ─────────────────────────────────────────────

class ARIMAPredictor:
    """
    Auto-ARIMA model for classical time-series forecasting.
    Uses pmdarima for automatic (p,d,q) selection.
    """

    def __init__(self, ticker: str = ""):
        self.ticker  = ticker
        self.model   = None
        self.is_fitted = False

    def fit(self, price_series: pd.Series, seasonal: bool = False,
            m: int = 5):
        """
        Fit Auto-ARIMA model to a price series.

        Parameters
        ----------
        price_series : closing prices (not returns)
        seasonal     : whether to fit seasonal ARIMA
        m            : seasonal period (5 = weekly)
        """
        try:
            import pmdarima as pm
            print(f"  🔧 Fitting Auto-ARIMA for {self.ticker} ...", end=" ")
            self.model = pm.auto_arima(
                price_series,
                start_p=1, start_q=1,
                max_p=5,   max_q=5,
                d=1,
                seasonal=seasonal, m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
                n_fits=30,
            )
            order = self.model.order
            print(f"Order {order} | AIC={self.model.aic():.2f}")
            self.is_fitted = True
        except ImportError:
            print("⚠ pmdarima not installed — using statsmodels ARIMA(2,1,2)")
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(price_series, order=(2, 1, 2)).fit()
            self.is_fitted = True
        return self

    def forecast(self, steps: int = 30) -> np.ndarray:
        """Forecast next N steps."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        try:
            preds, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
            return preds, conf_int
        except Exception:
            return np.array(self.model.forecast(steps=steps)), None

    def rolling_forecast(self, price_series: pd.Series,
                          train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walk-forward (rolling) 1-step ahead forecast on held-out test set.
        Refits model incrementally.
        """
        try:
            import pmdarima as pm
        except ImportError:
            # Fallback: naive random walk
            n       = len(price_series)
            split   = int(n * train_size)
            y_test  = price_series.values[split:]
            y_pred  = price_series.values[split-1:-1]   # yesterday's price
            return y_test, y_pred

        n     = len(price_series)
        split = int(n * train_size)
        train = price_series.values[:split]
        test  = price_series.values[split:]
        preds = []

        # Initial fit
        mdl = pm.ARIMA(order=self.model.order if self.is_fitted else (2, 1, 2))
        mdl.fit(train)

        # Rolling forecast
        history = list(train)
        for obs in test:
            fc = mdl.predict(n_periods=1)[0]
            preds.append(fc)
            history.append(obs)
            mdl.update([obs])

        return test, np.array(preds)


# ─────────────────────────────────────────────
# 4. LSTM Model
# ─────────────────────────────────────────────

class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) deep learning model.
    Handles sequence input for multi-step temporal pattern recognition.
    """

    def __init__(self, seq_len: int = 60, hidden_units: int = 128,
                 dropout: float = 0.2, ticker: str = ""):
        self.seq_len      = seq_len
        self.hidden_units = hidden_units
        self.dropout      = dropout
        self.ticker       = ticker
        self.model        = None
        self.scaler       = StandardScaler()
        self.history      = None
        self.is_fitted    = False

    def _build_model(self, n_features: int):
        """Build Keras LSTM model architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                                  BatchNormalization, Bidirectional)
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            model = Sequential([
                Bidirectional(LSTM(self.hidden_units, return_sequences=True,
                                   input_shape=(self.seq_len, n_features))),
                BatchNormalization(),
                Dropout(self.dropout),
                Bidirectional(LSTM(self.hidden_units // 2, return_sequences=False)),
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(64, activation="relu"),
                Dropout(self.dropout / 2),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber",
                          metrics=["mae"])
            return model
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """Convert flat feature arrays into (samples, seq_len, n_features) sequences."""
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 80, batch_size: int = 32, verbose: int = 0):
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            X_scaled = self.scaler.fit_transform(X_train)
            Xs, ys   = self._make_sequences(X_scaled, y_train)

            self.model = self._build_model(X_train.shape[1])

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=15,
                               restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=7, min_lr=1e-6, verbose=0),
            ]

            val_data = None
            if X_val is not None and y_val is not None:
                Xv_scaled = self.scaler.transform(X_val)
                Xv_seq, yv_seq = self._make_sequences(
                    np.vstack([X_scaled[-self.seq_len:], Xv_scaled]),
                    np.concatenate([y_train[-self.seq_len:], y_val])
                )
                val_data = (Xv_seq, yv_seq)

            self.history = self.model.fit(
                Xs, ys,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=False,
            )
            self.is_fitted = True
            print(f"  ✅ LSTM training complete | "
                  f"Epochs: {len(self.history.history['loss'])} | "
                  f"Final Loss: {self.history.history['loss'][-1]:.6f}")
        except ImportError as e:
            print(f"  ⚠ LSTM skipped: {e}")

        return self

    def predict(self, X: np.ndarray, seed_sequence: np.ndarray = None) -> np.ndarray:
        if self.model is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        if seed_sequence is not None:
            seed_scaled = self.scaler.transform(seed_sequence)
            combined    = np.vstack([seed_scaled, X_scaled])
        else:
            combined = X_scaled

        Xs = []
        for i in range(self.seq_len, len(combined)):
            Xs.append(combined[i - self.seq_len:i])
        if not Xs:
            return np.array([])
        return self.model.predict(np.array(Xs), verbose=0).flatten()

    def plot_training(self, save: bool = True):
        """Plot training vs validation loss curves."""
        if self.history is None:
            return
        hist = self.history.history
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e")

        ax.plot(hist["loss"],     color="#58a6ff", label="Train Loss", linewidth=1.5)
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], color="#f85149", label="Val Loss",   linewidth=1.5)
        ax.set_title(f"LSTM Training History — {self.ticker}", color="#f0f6fc",
                      fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.set_ylabel("Loss",  color="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#f0f6fc")
        plt.tight_layout()
        if save:
            path = FIG_DIR / f"lstm_training_{self.ticker}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
            plt.close()


# ─────────────────────────────────────────────
# Walk-Forward Backtesting
# ─────────────────────────────────────────────

def walk_forward_backtest(model_class, X: np.ndarray, y: np.ndarray,
                           dates, n_splits: int = 5,
                           feature_names: list = None,
                           model_kwargs: dict = None,
                           verbose: bool = True) -> Dict:
    """
    Time-series cross-validation using walk-forward splits.

    Each fold:
      - Train on all data up to fold boundary
      - Test on next N days
      - Report metrics per fold + aggregate

    Returns
    -------
    dict with per-fold metrics, predictions, and aggregate stats
    """
    model_kwargs = model_kwargs or {}
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    all_metrics  = []
    all_preds    = np.full(len(y), np.nan)
    fold         = 0

    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mdl = model_class(**model_kwargs)

        if model_class == XGBoostPredictor:
            mdl.fit(X_train, y_train, feature_names=feature_names)
        elif model_class == RandomForestPredictor:
            mdl.fit(X_train, y_train, feature_names=feature_names)
        else:
            mdl.fit(X_train, y_train)

        y_pred = mdl.predict(X_test)
        all_preds[test_idx] = y_pred

        m = evaluate(y_test, y_pred, f"Fold {fold}")
        all_metrics.append(m)

        if verbose:
            print(f"  Fold {fold}/{n_splits} | "
                  f"RMSE={m['RMSE']:.6f} | "
                  f"Dir.Acc={m['Dir. Acc. (%)']:.1f}%")

    # Aggregate metrics
    agg_df  = pd.DataFrame(all_metrics).set_index("Model")
    agg_mean = agg_df.mean().rename("Mean")
    agg_std  = agg_df.std().rename("Std")

    return {
        "fold_metrics": agg_df,
        "mean":         agg_mean,
        "std":          agg_std,
        "predictions":  all_preds,
        "actuals":      y,
        "dates":        dates,
    }


# ─────────────────────────────────────────────
# Signal Generation
# ─────────────────────────────────────────────

def generate_signals(predicted_returns: np.ndarray,
                     threshold_buy:  float = 0.005,
                     threshold_sell: float = -0.005) -> np.ndarray:
    """
    Convert predicted return magnitudes into trading signals.
      +1 = Buy  (predicted return > threshold_buy)
      -1 = Sell (predicted return < threshold_sell)
       0 = Hold (otherwise)
    """
    signals = np.zeros(len(predicted_returns))
    signals[predicted_returns >  threshold_buy]  = 1
    signals[predicted_returns < threshold_sell] = -1
    return signals


def simulate_strategy(signals: np.ndarray, actual_returns: np.ndarray,
                       transaction_cost: float = 0.001) -> pd.Series:
    """
    Simulate a simple long/short strategy from generated signals.

    Parameters
    ----------
    signals           : array of +1, -1, 0
    actual_returns    : next-day returns (aligned with signals)
    transaction_cost  : round-trip cost per trade (default: 10 bps)

    Returns
    -------
    pd.Series of strategy daily returns
    """
    strategy_returns = signals * actual_returns
    # Subtract transaction cost on trades
    trade_mask        = signals != 0
    strategy_returns[trade_mask] -= transaction_cost
    return pd.Series(strategy_returns)


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     dates, model_name: str = "", ticker: str = "",
                     save: bool = True):
    """Plot actual vs predicted returns on test set."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e")

    # Line plot
    axes[0].plot(dates, y_true * 100, color="#8b949e",  label="Actual",    linewidth=1, alpha=0.8)
    axes[0].plot(dates, y_pred * 100, color="#58a6ff",  label="Predicted", linewidth=1.2)
    axes[0].axhline(0, color="#30363d", linewidth=0.8)
    axes[0].set_title(f"{model_name} — {ticker}: Actual vs Predicted Returns",
                       color="#f0f6fc", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Return (%)", color="#8b949e")
    axes[0].legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=9)

    # Scatter plot
    axes[1].scatter(y_true * 100, y_pred * 100, alpha=0.4,
                    color="#58a6ff", s=8, edgecolors="none")
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 100 * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], color="#f85149", linewidth=1,
                  linestyle="--", label="Perfect fit")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("Actual (%)",    color="#8b949e")
    axes[1].set_ylabel("Predicted (%)", color="#8b949e")
    axes[1].set_title("Scatter: Actual vs Predicted", color="#f0f6fc",
                       fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=9)

    plt.tight_layout(pad=2)
    if save:
        path = FIG_DIR / f"predictions_{model_name}_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"💾 Saved: {path.name}")
    else:
        plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                             model_name: str = "", ticker: str = "",
                             save: bool = True):
    """Horizontal bar chart of feature importances."""
    if importance_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    top = importance_df.head(20)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))

    ax.barh(top["Feature"][::-1], top["Importance"][::-1],
            color=colors[::-1], edgecolor="none")
    ax.set_title(f"{model_name} — {ticker}: Feature Importance (Top 20)",
                  color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score", color="#8b949e")

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"feature_imp_{model_name}_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"💾 Saved: {path.name}")
    else:
        plt.show()


def plot_strategy_comparison(strategy_returns: pd.Series,
                              buy_hold_returns: pd.Series,
                              ticker: str = "", model_name: str = "",
                              save: bool = True):
    """Compare ML strategy cumulative return vs buy-and-hold."""
    strat_cum  = (1 + strategy_returns).cumprod()
    bh_cum     = (1 + buy_hold_returns).cumprod()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    ax.plot(strat_cum.index if hasattr(strat_cum, "index") else range(len(strat_cum)),
            strat_cum.values, color="#3fb950", linewidth=2, label=f"ML Strategy ({model_name})")
    ax.plot(bh_cum.index if hasattr(bh_cum, "index") else range(len(bh_cum)),
            bh_cum.values,  color="#58a6ff", linewidth=2, label="Buy & Hold")
    ax.axhline(1, color="#30363d", linewidth=0.8, linestyle="--")

    ax.fill_between(range(len(strat_cum)), 1, strat_cum.values,
                     where=strat_cum.values > bh_cum.values, alpha=0.15, color="#3fb950")

    ax.set_title(f"{ticker} — ML Strategy vs Buy & Hold",
                  color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($1 initial)", color="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#f0f6fc")
    plt.tight_layout()

    if save:
        path = FIG_DIR / f"strategy_{model_name}_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"💾 Saved: {path.name}")
    else:
        plt.show()


if __name__ == "__main__":
    from src.feature_engineering import build_features, get_feature_columns
    from src.data_loader import download_single

    print("🤖 Running ML model demo on AAPL ...")
    df       = download_single("AAPL")
    feat_df  = build_features(df)
    feat_cols = get_feature_columns(feat_df)

    X, y, dates = prepare_xy(feat_df, feat_cols, "Target_Return")
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split_ts(X, y, dates)

    # XGBoost
    xgb_model = XGBoostPredictor(ticker="AAPL")
    xgb_model.fit(X_tr, y_tr, X_te, y_te, feature_names=feat_cols)
    y_pred = xgb_model.predict(X_te)
    m = evaluate(y_te, y_pred, "XGBoost")
    print(f"\n📊 XGBoost: RMSE={m['RMSE']:.6f} | Dir.Acc={m['Dir. Acc. (%)']:.1f}%")
