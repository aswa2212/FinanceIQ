"""
portfolio.py — Markowitz Portfolio Optimization & Simulation
============================================================
Implements:
  - Markowitz Mean-Variance Optimization
  - Efficient Frontier construction (500+ portfolios)
  - Monte Carlo simulation (10,000+ random portfolios)
  - Maximum Sharpe Ratio portfolio
  - Minimum Variance portfolio
  - Risk Parity portfolio
  - Equal Weight benchmark
  - Cumulative performance comparison

Author  : Finance DS Internship Project
Version : 1.0
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR      = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────
# Core Portfolio Math
# ─────────────────────────────────────────────

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Annualised expected portfolio return."""
    return float(weights @ mean_returns * TRADING_DAYS)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualised portfolio volatility (std deviation)."""
    return float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS))


def portfolio_sharpe(weights: np.ndarray, mean_returns: np.ndarray,
                      cov_matrix: np.ndarray, rf: float = 0.05) -> float:
    """Annualised Sharpe Ratio for a given weight vector."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - rf) / (vol + 1e-9)


def negative_sharpe(weights, mean_returns, cov_matrix, rf=0.05):
    """Negative Sharpe (for minimisation)."""
    return -portfolio_sharpe(weights, mean_returns, cov_matrix, rf)


def portfolio_variance(weights, cov_matrix):
    """Portfolio variance (for minimum variance optimisation)."""
    return float(weights @ cov_matrix @ weights)


# ─────────────────────────────────────────────
# Optimisation Functions
# ─────────────────────────────────────────────

def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                          rf: float = 0.05, allow_short: bool = False) -> Dict:
    """
    Find portfolio weights that maximise the Sharpe Ratio.

    Parameters
    ----------
    mean_returns : array of daily mean returns per asset
    cov_matrix   : covariance matrix of returns
    rf           : annualised risk-free rate
    allow_short  : whether to allow short positions

    Returns
    -------
    dict with weights, return, volatility, sharpe
    """
    n = len(mean_returns)
    init_w = np.ones(n) / n

    bounds = ((-1.0, 1.0),) * n if allow_short else ((0.0, 1.0),) * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        negative_sharpe,
        init_w,
        args=(mean_returns, cov_matrix, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w    = result.x
    ret  = portfolio_return(w, mean_returns)
    vol  = portfolio_volatility(w, cov_matrix)
    sr   = (ret - rf) / (vol + 1e-9)

    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sr}


def min_variance_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                            allow_short: bool = False) -> Dict:
    """Find portfolio weights that minimise total variance."""
    n = len(mean_returns)
    init_w = np.ones(n) / n

    bounds      = ((-1.0, 1.0),) * n if allow_short else ((0.0, 1.0),) * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        portfolio_variance,
        init_w,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w   = result.x
    ret = portfolio_return(w, mean_returns)
    vol = portfolio_volatility(w, cov_matrix)
    rf  = 0.05
    sr  = (ret - rf) / (vol + 1e-9)

    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sr}


def risk_parity_portfolio(mean_returns: np.ndarray,
                           cov_matrix: np.ndarray) -> Dict:
    """
    Risk Parity — allocate such that each asset contributes equally to portfolio risk.
    Uses iterative 'equal risk contribution' (ERC) method.
    """
    n      = len(mean_returns)
    init_w = np.ones(n) / n

    def risk_contribution_diff(weights):
        """Minimise sum of squared differences in risk contributions."""
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        mrc      = cov_matrix @ weights / (port_vol + 1e-9)
        rc       = weights * mrc
        rc_norm  = rc / (rc.sum() + 1e-9)
        target   = 1.0 / n
        return np.sum((rc_norm - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = ((0.0, 1.0),) * n

    result = minimize(
        risk_contribution_diff,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    w   = result.x
    ret = portfolio_return(w, mean_returns)
    vol = portfolio_volatility(w, cov_matrix)
    rf  = 0.05
    sr  = (ret - rf) / (vol + 1e-9)

    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sr}


def equal_weight_portfolio(mean_returns: np.ndarray,
                            cov_matrix: np.ndarray) -> Dict:
    """1/N equal-weight benchmark portfolio."""
    n   = len(mean_returns)
    w   = np.ones(n) / n
    ret = portfolio_return(w, mean_returns)
    vol = portfolio_volatility(w, cov_matrix)
    rf  = 0.05
    sr  = (ret - rf) / (vol + 1e-9)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sr}


# ─────────────────────────────────────────────
# Efficient Frontier
# ─────────────────────────────────────────────

def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        n_points: int = 200, rf: float = 0.05) -> pd.DataFrame:
    """
    Trace the Efficient Frontier by optimising portfolios at target return levels.

    Returns
    -------
    DataFrame with columns [Return, Volatility, Sharpe, Weights...]
    """
    n           = len(mean_returns)
    min_ret     = mean_returns.min() * TRADING_DAYS
    max_ret     = mean_returns.max() * TRADING_DAYS
    target_rets = np.linspace(min_ret, max_ret, n_points)

    frontier_records = []

    for target_ret in target_rets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=target_ret:
             portfolio_return(w, mean_returns) - r},
        ]
        bounds = ((0.0, 1.0),) * n

        result = minimize(
            portfolio_variance,
            np.ones(n) / n,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if result.success:
            w   = result.x
            vol = portfolio_volatility(w, cov_matrix)
            sr  = (target_ret - rf) / (vol + 1e-9)
            frontier_records.append({
                "Return":     target_ret,
                "Volatility": vol,
                "Sharpe":     sr,
            })

    return pd.DataFrame(frontier_records)


# ─────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────

def monte_carlo_simulation(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             n_portfolios: int = 10_000,
                             rf: float = 0.05) -> pd.DataFrame:
    """
    Randomly generate portfolios to map the risk-return space.

    Returns
    -------
    DataFrame with [Return, Volatility, Sharpe, ...weights]
    """
    n         = len(mean_returns)
    results   = np.zeros((n_portfolios, n + 3))  # ret, vol, sharpe + weights

    for i in range(n_portfolios):
        w          = np.random.dirichlet(np.ones(n))   # random weights on simplex
        ret        = portfolio_return(w, mean_returns)
        vol        = portfolio_volatility(w, cov_matrix)
        sr         = (ret - rf) / (vol + 1e-9)
        results[i] = np.concatenate([[ret, vol, sr], w])

    cols = ["Return", "Volatility", "Sharpe"] + [f"w_{i}" for i in range(n)]
    return pd.DataFrame(results, columns=cols)


# ─────────────────────────────────────────────
# Portfolio Backtesting
# ─────────────────────────────────────────────

def backtest_portfolio(weights: np.ndarray, returns_matrix: pd.DataFrame,
                        rebalance_freq: str = "QE",
                        rf: float = 0.05) -> pd.DataFrame:
    """
    Simulate portfolio performance with periodic rebalancing.

    Parameters
    ----------
    weights         : initial weight array
    returns_matrix  : daily returns DataFrame
    rebalance_freq  : pandas offset alias (e.g. 'QE'=quarterly, 'ME'=monthly)
    rf              : risk-free rate for Sharpe calculation

    Returns
    -------
    DataFrame with portfolio daily return, cumulative return, drawdown
    """
    W         = np.array(weights)
    port_ret  = pd.Series(
        returns_matrix.values @ W,
        index=returns_matrix.index,
        name="Portfolio_Return"
    )
    port_cum  = (1 + port_ret).cumprod()
    roll_max  = port_cum.cummax()
    drawdown  = (port_cum - roll_max) / roll_max

    return pd.DataFrame({
        "Daily_Return":      port_ret,
        "Cumulative_Return": port_cum,
        "Drawdown":          drawdown,
    })


def summary_table(portfolios: Dict[str, Dict], tickers: List[str]) -> pd.DataFrame:
    """
    Compare multiple portfolio strategies in a single table.

    portfolios = {
        "Max Sharpe": {"weights": ..., "return": ..., "volatility": ..., "sharpe": ...},
        ...
    }
    """
    rows = []
    for name, pf in portfolios.items():
        w = pf["weights"]
        rows.append({
            "Strategy":           name,
            "Ann. Return (%)":    f"{pf['return']*100:.2f}%",
            "Ann. Volatility (%)":f"{pf['volatility']*100:.2f}%",
            "Sharpe Ratio":       f"{pf['sharpe']:.3f}",
            **{f"{t} (%)": f"{w[i]*100:.1f}%" for i, t in enumerate(tickers)},
        })
    return pd.DataFrame(rows).set_index("Strategy")


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────

def plot_efficient_frontier(mc_df: pd.DataFrame,
                             frontier_df: pd.DataFrame,
                             special_portfolios: Dict[str, Dict] = None,
                             save: bool = True):
    """
    Master Efficient Frontier plot:
    - Monte Carlo scatter (coloured by Sharpe)
    - Efficient Frontier line
    - Special portfolio markers (Max Sharpe, Min Var, Risk Parity, EW)
    """
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    # Monte Carlo scatter
    sc = ax.scatter(
        mc_df["Volatility"] * 100,
        mc_df["Return"]     * 100,
        c=mc_df["Sharpe"],
        cmap="plasma",
        alpha=0.4, s=8, linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Sharpe Ratio", color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Efficient Frontier
    if not frontier_df.empty:
        ax.plot(
            frontier_df["Volatility"] * 100,
            frontier_df["Return"]     * 100,
            color="#f0f6fc", linewidth=2.5, label="Efficient Frontier", zorder=5
        )

    # Special portfolios
    markers = {"Max Sharpe": ("*", "#f1c40f", 250),
               "Min Variance": ("o", "#58a6ff", 200),
               "Risk Parity":  ("D", "#3fb950", 200),
               "Equal Weight": ("s", "#ffa657", 180)}

    if special_portfolios:
        for name, pf in special_portfolios.items():
            color  = markers.get(name, ("o", "#f85149", 150))[1]
            size   = markers.get(name, ("o", "#f85149", 150))[2]
            label  = f"{name}  (SR={pf['sharpe']:.2f})"
            ax.scatter(
                pf["volatility"] * 100,
                pf["return"]     * 100,
                color=color, s=size, zorder=10, edgecolors="white",
                linewidths=1.5, label=label
            )

    ax.set_title("Markowitz Efficient Frontier + Monte Carlo Simulation\n"
                  f"(10,000 random portfolios)",
                  color="#f0f6fc", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Annualised Volatility (%)", color="#8b949e", fontsize=11)
    ax.set_ylabel("Annualised Return (%)",     color="#8b949e", fontsize=11)
    ax.legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=10,
               loc="upper left", framealpha=0.8)
    ax.grid(True, color="#30363d", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    if save:
        path = FIG_DIR / "efficient_frontier.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def plot_portfolio_weights(portfolios: Dict[str, Dict], tickers: List[str],
                            save: bool = True):
    """Side-by-side weight bar charts for each portfolio strategy."""
    n_port = len(portfolios)
    fig, axes = plt.subplots(1, n_port, figsize=(4 * n_port, 6), facecolor="#0d1117")

    if n_port == 1:
        axes = [axes]

    color_palette = ["#58a6ff", "#3fb950", "#f85149", "#ffa657",
                      "#bc8cff", "#39d353", "#ff7b72", "#79c0ff"]

    for ax, (name, pf) in zip(axes, portfolios.items()):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

        weights = pf["weights"] * 100
        colors  = [color_palette[i % len(color_palette)] for i in range(len(tickers))]

        bars = ax.bar(tickers, weights, color=colors, edgecolor="none")
        for bar, val in zip(bars, weights):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom",
                        color="#f0f6fc", fontsize=7)

        ax.set_title(name, color="#f0f6fc", fontsize=10, fontweight="bold")
        ax.set_ylabel("Weight (%)", color="#8b949e", fontsize=8)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Portfolio Weight Allocations", color="#f0f6fc",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = FIG_DIR / "portfolio_weights.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


def plot_portfolio_performance(perf_dict: Dict[str, pd.DataFrame],
                                save: bool = True):
    """Cumulative return comparison across portfolio strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0d1117")
    colors = ["#f1c40f", "#58a6ff", "#3fb950", "#ffa657", "#bc8cff"]

    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e")

    # Cumulative return
    for (name, perf), color in zip(perf_dict.items(), colors):
        axes[0].plot(perf.index, perf["Cumulative_Return"],
                     color=color, linewidth=1.8, label=name)

    axes[0].axhline(1, color="#30363d", linewidth=0.8, linestyle="--")
    axes[0].set_title("Portfolio Cumulative Performance (Out-of-Sample)",
                       color="#f0f6fc", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Growth of $1", color="#8b949e")
    axes[0].legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=9)

    # Drawdown
    for (name, perf), color in zip(perf_dict.items(), colors):
        axes[1].plot(perf.index, perf["Drawdown"] * 100,
                     color=color, linewidth=1.2, label=name, alpha=0.8)

    axes[1].axhline(0, color="#30363d", linewidth=0.8)
    axes[1].set_title("Portfolio Drawdowns (%)", color="#f0f6fc",
                       fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)", color="#8b949e")
    axes[1].set_xlabel("Date",         color="#8b949e")
    axes[1].legend(facecolor="#161b22", labelcolor="#f0f6fc", fontsize=9)

    plt.tight_layout(pad=2)
    if save:
        path = FIG_DIR / "portfolio_performance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[SAVED] {path.name}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import download_all, build_price_matrix, compute_returns, TICKERS

    print("[RUNNING] Running Portfolio Optimization ...")
    tickers = list(TICKERS.keys())[:8]   # Use first 8 stocks
    data    = download_all(tickers)
    pm      = build_price_matrix({t: data[t] for t in tickers if t in data})
    simple_ret, _ = compute_returns(pm)

    # Only keep last 3 years
    simple_ret = simple_ret[simple_ret.index >= "2022-01-01"]

    mu  = simple_ret.mean().values
    cov = simple_ret.cov().values

    print("\n[OPTIMIZING] Optimising portfolios ...")
    ms  = max_sharpe_portfolio(mu, cov)
    mv  = min_variance_portfolio(mu, cov)
    rp  = risk_parity_portfolio(mu, cov)
    ew  = equal_weight_portfolio(mu, cov)

    portfolios = {
        "Max Sharpe":   ms,
        "Min Variance": mv,
        "Risk Parity":  rp,
        "Equal Weight": ew,
    }

    print("\n[REPORT] Portfolio Comparison:")
    print(summary_table(portfolios, tickers).to_string())

    print("\n[RUNNING] Running Monte Carlo Simulation (10k portfolios) ...")
    mc_df = monte_carlo_simulation(mu, cov, n_portfolios=10_000)

    print("[RUNNING] Computing Efficient Frontier ...")
    ef_df = efficient_frontier(mu, cov, n_points=200)

    plot_efficient_frontier(mc_df, ef_df,
                             special_portfolios={k: {"return": v["return"],
                                                      "volatility": v["volatility"],
                                                      "sharpe": v["sharpe"]}
                                                   for k, v in portfolios.items()})
    print("[SUCCESS] Portfolio optimization complete!")
