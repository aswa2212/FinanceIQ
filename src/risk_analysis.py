"""
risk_analysis.py — Financial Risk & Performance Metrics
=======================================================
Computes institutional-grade risk metrics:
  - Value at Risk (Historical, Parametric, Monte Carlo)
  - Conditional VaR / Expected Shortfall
  - Sharpe, Sortino, Calmar, Treynor Ratios
  - Beta, Alpha vs benchmark
  - Maximum Drawdown & recovery analytics
  - Rolling risk metrics
  - Stress testing

Author  : Finance DS Internship Project
Version : 1.0
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR      = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE    = 0.05          # 5% annualised risk-free rate (US 1-yr T-bill approx)


# ─────────────────────────────────────────────
# Core Risk Metrics
# ─────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR — non-parametric quantile of return distribution."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR — assumes normally distributed returns."""
    mu  = returns.mean()
    sig = returns.std()
    z   = stats.norm.ppf(1 - confidence)
    return float(mu + z * sig)


def var_montecarlo(returns: pd.Series, confidence: float = 0.95,
                   n_simulations: int = 100_000) -> float:
    """Monte Carlo VaR — simulate return distribution via fitted normal."""
    mu  = returns.mean()
    sig = returns.std()
    sim = np.random.normal(mu, sig, n_simulations)
    return float(np.percentile(sim, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — expected loss beyond VaR threshold.
    More robust than VaR for tail-risk estimation.
    """
    var_level = var_historical(returns, confidence)
    return float(returns[returns <= var_level].mean())


def var_summary(returns: pd.Series, portfolio_value: float = 1_000_000,
                confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)) -> pd.DataFrame:
    """
    Compute all VaR/CVaR variants at multiple confidence levels.

    Returns
    -------
    DataFrame with dollar amounts and percentages
    """
    rows = []
    for cl in confidence_levels:
        h_var  = var_historical(returns, cl)
        p_var  = var_parametric(returns, cl)
        mc_var = var_montecarlo(returns, cl)
        cv     = cvar(returns, cl)
        rows.append({
            "Confidence":          f"{cl*100:.0f}%",
            "Historical VaR (%)":  f"{h_var*100:.3f}%",
            "Parametric VaR (%)":  f"{p_var*100:.3f}%",
            "Monte Carlo VaR (%)": f"{mc_var*100:.3f}%",
            "CVaR / ES (%)":       f"{cv*100:.3f}%",
            "Historical VaR ($)":  f"${abs(h_var)*portfolio_value:,.0f}",
            "CVaR ($)":            f"${abs(cv)*portfolio_value:,.0f}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Performance Ratios
# ─────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE,
                 annualise: bool = True) -> float:
    """Sharpe Ratio — excess return per unit of total risk."""
    daily_rf = rf / TRADING_DAYS
    excess   = returns - daily_rf
    ratio    = excess.mean() / (returns.std() + 1e-9)
    return float(ratio * np.sqrt(TRADING_DAYS) if annualise else ratio)


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE,
                  annualise: bool = True) -> float:
    """Sortino Ratio — excess return per unit of downside risk only."""
    daily_rf     = rf / TRADING_DAYS
    excess       = returns - daily_rf
    downside_ret = returns[returns < 0]
    downside_std = downside_ret.std() + 1e-9
    ratio        = excess.mean() / downside_std
    return float(ratio * np.sqrt(TRADING_DAYS) if annualise else ratio)


def calmar_ratio(returns: pd.Series, annualise: bool = True) -> float:
    """Calmar Ratio — annualised return / maximum drawdown."""
    ann_return = returns.mean() * TRADING_DAYS
    max_dd     = max_drawdown(returns)
    return float(ann_return / (abs(max_dd) + 1e-9))


def treynor_ratio(returns: pd.Series, market_returns: pd.Series,
                  rf: float = RISK_FREE) -> float:
    """Treynor Ratio — excess return per unit of systematic (market) risk."""
    daily_rf = rf / TRADING_DAYS
    b        = beta(returns, market_returns)
    excess   = (returns.mean() - daily_rf) * TRADING_DAYS
    return float(excess / (abs(b) + 1e-9))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Information Ratio — active return over tracking error."""
    active    = returns - benchmark_returns
    tracking  = active.std() + 1e-9
    return float(active.mean() / tracking * np.sqrt(TRADING_DAYS))


# ─────────────────────────────────────────────
# Drawdown Analysis
# ─────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the rolling drawdown time series from a return stream."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    dd          = (cumulative - rolling_max) / rolling_max
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    return float(drawdown_series(returns).min())


def drawdown_analytics(returns: pd.Series) -> Dict:
    """
    Full drawdown analytics:
    - Max drawdown
    - Drawdown start/end dates
    - Recovery date
    - Average drawdown
    - Duration statistics
    """
    dd_series    = drawdown_series(returns)
    cumulative   = (1 + returns).cumprod()
    rolling_max  = cumulative.cummax()

    max_dd       = dd_series.min()
    max_dd_end   = dd_series.idxmin()

    # Find the peak before max drawdown
    peak_idx = cumulative[:max_dd_end].idxmax()

    # Find recovery (first day after trough that hits previous peak)
    recovery_series = cumulative[max_dd_end:]
    peak_value      = cumulative[peak_idx]
    recovered       = recovery_series[recovery_series >= peak_value]
    recovery_date   = recovered.index[0] if not recovered.empty else None

    avg_dd = dd_series[dd_series < 0].mean()

    return {
        "Max Drawdown":       f"{max_dd*100:.2f}%",
        "Max DD Peak Date":   str(peak_idx.date()),
        "Max DD Trough Date": str(max_dd_end.date()),
        "Recovery Date":      str(recovery_date.date()) if recovery_date else "Not recovered",
        "DD Duration (days)": (max_dd_end - peak_idx).days,
        "Recovery Time (days)": (recovery_date - max_dd_end).days if recovery_date else "N/A",
        "Avg Drawdown":       f"{avg_dd*100:.2f}%",
        "Time in Drawdown (%)": f"{(dd_series < 0).mean()*100:.1f}%",
    }


# ─────────────────────────────────────────────
# Beta & Alpha
# ─────────────────────────────────────────────

def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Market Beta — sensitivity of asset to market movements."""
    aligned     = pd.concat([returns, market_returns], axis=1).dropna()
    asset_r     = aligned.iloc[:, 0]
    mkt_r       = aligned.iloc[:, 1]
    covariance  = np.cov(asset_r, mkt_r)[0, 1]
    mkt_var     = np.var(mkt_r, ddof=1)
    return float(covariance / (mkt_var + 1e-12))


def alpha(returns: pd.Series, market_returns: pd.Series,
          rf: float = RISK_FREE) -> float:
    """
    Jensen's Alpha — abnormal return vs CAPM prediction.
    α = R_asset - [R_f + β × (R_market - R_f)]
    """
    daily_rf   = rf / TRADING_DAYS
    b          = beta(returns, market_returns)
    ann_asset  = returns.mean() * TRADING_DAYS
    ann_mkt    = market_returns.mean() * TRADING_DAYS
    return float(ann_asset - (daily_rf * TRADING_DAYS + b * (ann_mkt - rf)))


def capm_decomposition(returns: pd.Series,
                        market_returns: pd.Series,
                        rf: float = RISK_FREE) -> pd.Series:
    """Full CAPM analytics for a single asset."""
    b   = beta(returns, market_returns)
    a   = alpha(returns, market_returns, rf)
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    corr    = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    r2      = corr ** 2

    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sr      = sharpe_ratio(returns, rf)
    so      = sortino_ratio(returns, rf)
    mdd     = max_drawdown(returns)

    return pd.Series({
        "Annualised Return":  f"{ann_ret*100:.2f}%",
        "Annualised Vol":     f"{ann_vol*100:.2f}%",
        "Sharpe Ratio":       f"{sr:.3f}",
        "Sortino Ratio":      f"{so:.3f}",
        "Calmar Ratio":       f"{calmar_ratio(returns):.3f}",
        "Beta (β)":           f"{b:.3f}",
        "Alpha (α)":          f"{a*100:.2f}%",
        "R² vs Market":       f"{r2:.3f}",
        "Max Drawdown":       f"{mdd*100:.2f}%",
        "VaR 95% (daily)":    f"{var_historical(returns, 0.95)*100:.3f}%",
        "CVaR 95% (daily)":   f"{cvar(returns, 0.95)*100:.3f}%",
    })


# ─────────────────────────────────────────────
# Portfolio Risk Metrics
# ─────────────────────────────────────────────

def portfolio_metrics(weights: np.ndarray, returns_matrix: pd.DataFrame,
                       rf: float = RISK_FREE) -> Dict:
    """
    Compute annualised return, volatility, Sharpe for a weighted portfolio.

    Parameters
    ----------
    weights        : array of portfolio weights (must sum to 1)
    returns_matrix : DataFrame of daily returns, one column per asset
    rf             : annualised risk-free rate

    Returns
    -------
    dict with ret, vol, sharpe, var, cvar
    """
    W         = np.array(weights)
    mu        = returns_matrix.mean().values
    cov       = returns_matrix.cov().values

    port_ret  = W @ mu * TRADING_DAYS
    port_var  = W @ cov @ W
    port_vol  = np.sqrt(port_var) * np.sqrt(TRADING_DAYS)
    port_sr   = (port_ret - rf) / (port_vol + 1e-9)

    port_daily_returns = returns_matrix @ W
    port_var95  = var_historical(port_daily_returns, 0.95)
    port_cvar95 = cvar(port_daily_returns, 0.95)

    return {
        "Return":     port_ret,
        "Volatility": port_vol,
        "Sharpe":     port_sr,
        "VaR_95":     port_var95,
        "CVaR_95":    port_cvar95,
    }


# ─────────────────────────────────────────────
# Rolling Risk Analytics
# ─────────────────────────────────────────────

def rolling_risk(returns: pd.Series, window: int = 63) -> pd.DataFrame:
    """
    Compute rolling risk metrics over a sliding window (default: ~1 quarter).

    Returns
    -------
    DataFrame with rolling Sharpe, vol, VaR, drawdown
    """
    results = pd.DataFrame(index=returns.index)
    results["Rolling_Volatility"] = (
        returns.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    )
    results["Rolling_Sharpe"] = returns.rolling(window).apply(
        lambda x: sharpe_ratio(pd.Series(x)), raw=False
    )
    results["Rolling_VaR_95"] = returns.rolling(window).apply(
        lambda x: var_historical(pd.Series(x), 0.95) * 100, raw=False
    )
    results["Rolling_Drawdown"] = drawdown_series(returns) * 100
    return results.dropna()


# ─────────────────────────────────────────────
# Stress Testing
# ─────────────────────────────────────────────

STRESS_SCENARIOS = {
    "Covid Crash (Mar 2020)":   ("2020-02-19", "2020-03-23"),
    "Dot-Com Crash":            ("2000-03-10", "2002-10-09"),
    "GFC 2008-09":              ("2007-10-09", "2009-03-09"),
    "2022 Bear Market":         ("2021-12-29", "2022-10-12"),
    "SVB Crisis (Mar 2023)":    ("2023-03-08", "2023-03-14"),
    "Ukraine War Spike":        ("2022-02-24", "2022-03-08"),
}


def stress_test(returns: pd.Series, scenarios: Dict = None) -> pd.DataFrame:
    """
    Evaluate portfolio/asset return during historical stress scenarios.

    Returns
    -------
    DataFrame with scenario returns, Sharpe, and max drawdown
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    rows = []
    for name, (start, end) in scenarios.items():
        try:
            period_ret  = returns.loc[start:end]
            total_ret   = (1 + period_ret).prod() - 1
            period_vol  = period_ret.std() * np.sqrt(TRADING_DAYS)
            period_mdd  = max_drawdown(period_ret)
            rows.append({
                "Scenario":        name,
                "Period":          f"{start} → {end}",
                "Total Return":    f"{total_ret*100:.2f}%",
                "Annualised Vol":  f"{period_vol*100:.2f}%",
                "Max Drawdown":    f"{period_mdd*100:.2f}%",
            })
        except Exception:
            rows.append({
                "Scenario": name, "Period": f"{start} → {end}",
                "Total Return": "N/A", "Annualised Vol": "N/A", "Max Drawdown": "N/A",
            })
    return pd.DataFrame(rows).set_index("Scenario")


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────

def plot_drawdown(returns: pd.Series, ticker: str = "", save: bool = True):
    """Plot the drawdown time series with shaded regions."""
    dd = drawdown_series(returns) * 100
    cumret = (1 + returns).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.xaxis.label.set_color("#8b949e")

    # Cumulative return
    axes[0].plot(cumret.index, cumret.values, color="#58a6ff", linewidth=1.5)
    axes[0].fill_between(cumret.index, 1, cumret.values,
                          where=cumret.values >= 1, alpha=0.2, color="#3fb950")
    axes[0].fill_between(cumret.index, 1, cumret.values,
                          where=cumret.values < 1,  alpha=0.2, color="#f85149")
    axes[0].set_title(f"{ticker} — Cumulative Return", color="#f0f6fc",
                       fontsize=13, pad=10, fontweight="bold")
    axes[0].set_ylabel("Growth of $1")
    axes[0].axhline(1, color="#30363d", linewidth=0.8, linestyle="--")

    # Drawdown
    axes[1].fill_between(dd.index, 0, dd.values, color="#f85149", alpha=0.6)
    axes[1].plot(dd.index, dd.values, color="#f85149", linewidth=1)
    axes[1].set_title("Drawdown (%)", color="#f0f6fc", fontsize=13, pad=10, fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2)
    if save:
        path = FIG_DIR / f"drawdown_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def plot_return_distribution(returns: pd.Series, ticker: str = "", save: bool = True):
    """Return distribution with VaR/CVaR annotations and normal overlay."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    returns_pct = returns * 100
    mu, sig = returns_pct.mean(), returns_pct.std()

    ax.hist(returns_pct, bins=80, density=True, color="#1f6feb",
            alpha=0.7, edgecolor="#0d1117", linewidth=0.3, label="Empirical")

    x = np.linspace(returns_pct.min(), returns_pct.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sig), color="#e3b341",
            linewidth=2, linestyle="--", label="Normal Fit")

    # VaR / CVaR lines
    v95  = var_historical(returns, 0.95) * 100
    cv95 = cvar(returns, 0.95) * 100
    v99  = var_historical(returns, 0.99) * 100

    ax.axvline(v95,  color="#f85149", linewidth=1.5, linestyle="--", label=f"VaR 95% ({v95:.2f}%)")
    ax.axvline(cv95, color="#ff7b72", linewidth=1.5, linestyle=":",  label=f"CVaR 95% ({cv95:.2f}%)")
    ax.axvline(v99,  color="#ffa657", linewidth=1.5, linestyle="--", label=f"VaR 99% ({v99:.2f}%)")
    ax.axvline(0,    color="#8b949e", linewidth=0.8, linestyle="-")

    # Skew / Kurt annotation
    sk = stats.skew(returns_pct.dropna())
    kt = stats.kurtosis(returns_pct.dropna())
    ax.text(0.02, 0.97, f"μ={mu:.3f}%  σ={sig:.3f}%\nSkew={sk:.3f}  Kurt={kt:.3f}",
            transform=ax.transAxes, color="#8b949e", fontsize=9,
            verticalalignment="top", fontfamily="monospace")

    ax.set_title(f"{ticker} — Daily Return Distribution", color="#f0f6fc",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Daily Return (%)", color="#8b949e")
    ax.set_ylabel("Density",          color="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=9)

    plt.tight_layout()
    if save:
        path = FIG_DIR / f"return_dist_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def plot_rolling_risk(returns: pd.Series, ticker: str = "", save: bool = True):
    """Visualise rolling volatility, Sharpe ratio, and VaR."""
    rr = rolling_risk(returns)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, facecolor="#0d1117")
    colors = ["#58a6ff", "#3fb950", "#f85149"]
    labels = ["Rolling Volatility (%, ann.)", "Rolling Sharpe Ratio", "Rolling VaR 95% (%)"]
    cols   = ["Rolling_Volatility", "Rolling_Sharpe", "Rolling_VaR_95"]

    for ax, col, color, label in zip(axes, cols, colors, labels):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.plot(rr.index, rr[col], color=color, linewidth=1.2)
        ax.fill_between(rr.index, rr[col].min(), rr[col], alpha=0.15, color=color)
        ax.set_ylabel(label, color="#8b949e", fontsize=9)
        ax.axhline(rr[col].mean(), color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    axes[0].set_title(f"{ticker} — Rolling Risk Metrics (63-day window)",
                       color="#f0f6fc", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Date", color="#8b949e")
    plt.tight_layout(pad=1.5)

    if save:
        path = FIG_DIR / f"rolling_risk_{ticker}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def plot_correlation_heatmap(returns_matrix: pd.DataFrame, save: bool = True):
    """Heatmap of pairwise correlations among all assets."""
    corr = returns_matrix.corr()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9, "color": "white"})

    ax.set_title("Asset Return Correlations", color="#f0f6fc",
                  fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#8b949e", labelsize=10)
    plt.tight_layout()

    if save:
        path = FIG_DIR / "correlation_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def full_risk_report(returns: pd.Series, market_returns: pd.Series,
                      ticker: str = "Asset") -> pd.DataFrame:
    """
    Generate a comprehensive risk report for a single asset vs benchmark.

    Returns
    -------
    pd.Series with all metrics
    """
    metrics = capm_decomposition(returns, market_returns)
    dd_info = drawdown_analytics(returns)

    extra = pd.Series({
        "Treynor Ratio":         f"{treynor_ratio(returns, market_returns):.3f}",
        "Information Ratio":     f"{information_ratio(returns, market_returns):.3f}",
        "VaR 99% (daily)":       f"{var_historical(returns, 0.99)*100:.3f}%",
        "CVaR 99% (daily)":      f"{cvar(returns, 0.99)*100:.3f}%",
        "Skewness":              f"{returns.skew():.3f}",
        "Excess Kurtosis":       f"{returns.kurtosis():.3f}",
        "Positive Days (%)":     f"{(returns > 0).mean()*100:.1f}%",
        **dd_info
    })

    report = pd.concat([metrics, extra])
    report.name = ticker
    return report


if __name__ == "__main__":
    print("[RUNNING] Running risk analysis demo (synthetic data) ...")
    np.random.seed(42)
    n    = 1260
    ret  = pd.Series(np.random.normal(0.0005, 0.015, n),
                     index=pd.date_range("2020-01-01", periods=n, freq="B"))
    mkt  = pd.Series(np.random.normal(0.0004, 0.012, n),
                     index=ret.index)

    report = full_risk_report(ret, mkt, "Demo Asset")
    print("\n[REPORT] Full Risk Report:")
    print(report.to_string())

    print("\n[REPORT] VaR Summary:")
    print(var_summary(ret).to_string(index=False))

    print("\n[REPORT] Stress Test:")
    print(stress_test(ret).to_string())
