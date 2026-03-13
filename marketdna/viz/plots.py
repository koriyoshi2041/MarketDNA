"""
MarketDNA 可视化模块 — 把数字变成直觉

每一张图都对应一个关键的量化概念：
  - QQ-Plot → 正态性假设是否成立
  - 收益率分布 → 厚尾有多厚
  - 波动率时序 → 聚类现象
  - 价差图 → 配对交易机会
  - Regime 时间线 → 市场状态切换
  - RMT 特征值 → 信号 vs 噪声
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats


# 全局样式设置
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "grid.color": "#222",
    "grid.alpha": 0.5,
    "figure.dpi": 120,
    "font.size": 10,
})

# 配色方案
COLORS = {
    "primary": "#00d4aa",
    "secondary": "#ff6b6b",
    "accent": "#4ecdc4",
    "warn": "#ffe66d",
    "neutral": "#95a5a6",
    "bg_panel": "#1a1a2e",
}


def _save_or_show(fig: plt.Figure, save_path: str | None, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_qq(returns: pd.Series, ticker: str = "", save_path: str | None = None) -> None:
    """QQ-Plot: 收益率 vs 正态分布

    如果点落在对角线上 → 完美正态
    如果尾部弯曲 → 厚尾（真实市场几乎都这样）
    """
    r = returns.dropna().values
    fig, ax = plt.subplots(figsize=(7, 7))

    (theoretical, sample), (slope, intercept, _) = stats.probplot(r, dist="norm")

    # Size points by distance from the reference line — tails get bigger
    expected = slope * theoretical + intercept
    deviation = np.abs(sample - expected)
    sizes = 6 + 80 * (deviation / (deviation.max() + 1e-9))

    ax.scatter(theoretical, sample, s=sizes, alpha=0.6, color=COLORS["primary"],
               edgecolors="white", linewidths=0.3, zorder=2)

    # Reference line
    x_line = np.array([theoretical.min(), theoretical.max()])
    ax.plot(x_line, slope * x_line + intercept, "--", color=COLORS["secondary"], linewidth=1.5, label="Normal ref")

    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"{ticker} QQ-Plot vs Normal Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_distribution(
    returns: pd.Series,
    ticker: str = "",
    t_df: float | None = None,
    save_path: str | None = None,
) -> None:
    """收益率分布直方图 + 正态叠加 + Student-t 叠加"""
    r = returns.dropna().values
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(r, bins=100, density=True, alpha=0.6, color=COLORS["primary"], label="Actual")

    # 正态拟合
    x = np.linspace(r.min(), r.max(), 300)
    mu, sigma = np.mean(r), np.std(r)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "--",
            color=COLORS["secondary"], linewidth=2, label=f"Normal(mu={mu:.4f})")

    # Student-t 拟合
    if t_df is None:
        t_df, t_loc, t_scale = stats.t.fit(r)
    else:
        t_loc, t_scale = mu, sigma * np.sqrt((t_df - 2) / t_df) if t_df > 2 else sigma
    ax.plot(x, stats.t.pdf(x, t_df, loc=t_loc, scale=t_scale), "-",
            color=COLORS["warn"], linewidth=2, label=f"Student-t(df={t_df:.1f})")

    # 标注 3-sigma 区域
    for sign in [-1, 1]:
        ax.axvline(mu + sign * 3 * sigma, color=COLORS["neutral"], linestyle=":", alpha=0.7)
    ax.text(mu + 3 * sigma, ax.get_ylim()[1] * 0.9, "3σ", color=COLORS["neutral"], fontsize=9)

    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.set_title(f"{ticker} Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_volatility_cluster(
    returns: pd.Series,
    ticker: str = "",
    window: int = 20,
    save_path: str | None = None,
) -> None:
    """波动率聚类可视化：收益率 + 滚动波动率"""
    r = returns.dropna()
    rolling_vol = r.rolling(window).std() * np.sqrt(252)  # 年化

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # 上图：收益率
    ax1.bar(r.index, r.values, width=1, color=np.where(r.values >= 0, COLORS["primary"], COLORS["secondary"]),
            alpha=0.6)
    ax1.set_ylabel("Daily Return")
    ax1.set_title(f"{ticker} Returns & Volatility Clustering")
    ax1.grid(True, alpha=0.3)

    # 下图：滚动波动率
    ax2.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.4, color=COLORS["accent"])
    ax2.plot(rolling_vol.index, rolling_vol.values, color=COLORS["accent"], linewidth=0.8)
    ax2.set_ylabel(f"Rolling {window}d Ann. Vol")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    _save_or_show(fig, save_path)


def plot_spread(
    spread: pd.Series,
    zscore: pd.Series,
    positions: pd.Series | None = None,
    ticker_a: str = "A",
    ticker_b: str = "B",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    save_path: str | None = None,
) -> None:
    """配对交易价差与Z-Score图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2]})

    # 上图：价差
    ax1.plot(spread.index, spread.values, color=COLORS["primary"], linewidth=0.8)
    ax1.axhline(spread.mean(), color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax1.set_ylabel("Spread")
    ax1.set_title(f"{ticker_a}/{ticker_b} Pair Trading Spread & Z-Score")
    ax1.grid(True, alpha=0.3)

    # Lower panel: Z-Score with thresholds (simplified — no position shading to reduce clutter)
    ax2.fill_between(zscore.index, 0, zscore.values,
                     where=zscore.values > entry_z, alpha=0.3,
                     color=COLORS["secondary"], label=f"Short zone (z>{entry_z})")
    ax2.fill_between(zscore.index, 0, zscore.values,
                     where=zscore.values < -entry_z, alpha=0.3,
                     color=COLORS["primary"], label=f"Long zone (z<-{entry_z})")
    ax2.plot(zscore.index, zscore.values, color=COLORS["accent"], linewidth=0.6, alpha=0.8)
    ax2.axhline(entry_z, color=COLORS["secondary"], linestyle="--", alpha=0.5)
    ax2.axhline(-entry_z, color=COLORS["secondary"], linestyle="--", alpha=0.5)
    ax2.axhline(0, color=COLORS["neutral"], linestyle="-", alpha=0.3)

    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_regime_timeline(
    returns: pd.Series,
    regime_labels: pd.Series,
    regime_names: list[str] | None = None,
    ticker: str = "",
    save_path: str | None = None,
) -> None:
    """Regime 时间线：用颜色标注不同市场状态"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    n_regimes = int(regime_labels.max()) + 1
    regime_colors = [COLORS["primary"], COLORS["warn"], COLORS["secondary"], "#9b59b6"][:n_regimes]

    if regime_names is None:
        regime_names = [f"Regime {i}" for i in range(n_regimes)]

    # 上图：累计收益 + regime 背景色
    cum_ret = (1 + returns.loc[regime_labels.index]).cumprod()
    ax1.plot(cum_ret.index, cum_ret.values, color="white", linewidth=0.8)

    for k in range(n_regimes):
        mask = regime_labels == k
        ax1.fill_between(cum_ret.index, cum_ret.min(), cum_ret.max(),
                        where=mask.reindex(cum_ret.index, fill_value=False),
                        alpha=0.15, color=regime_colors[k], label=regime_names[k])

    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(f"{ticker} Market Regime Timeline")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下图：regime 标签条
    for k in range(n_regimes):
        mask = regime_labels == k
        ax2.fill_between(regime_labels.index, 0, 1,
                        where=mask.values, color=regime_colors[k], alpha=0.8)

    ax2.set_yticks([])
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    _save_or_show(fig, save_path)


def plot_rmt_eigenvalues(
    eigenvalues: np.ndarray,
    mp_lambda_max: float,
    mp_lambda_min: float,
    n_assets: int,
    n_observations: int,
    save_path: str | None = None,
) -> None:
    """RMT 特征值分布 vs Marchenko-Pastur 理论分布"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 特征值直方图
    ax.hist(eigenvalues, bins=min(50, len(eigenvalues) // 2), density=True,
            alpha=0.6, color=COLORS["primary"], label="Sample eigenvalues")

    # Marchenko-Pastur 理论分布
    q = n_assets / n_observations
    x = np.linspace(mp_lambda_min * 0.9, mp_lambda_max * 1.1, 500)
    mp_pdf = np.zeros_like(x)
    valid = (x >= mp_lambda_min) & (x <= mp_lambda_max)
    if valid.any():
        mp_pdf[valid] = (1 / (2 * np.pi * q * x[valid])) * np.sqrt(
            (mp_lambda_max - x[valid]) * (x[valid] - mp_lambda_min)
        )
    ax.plot(x, mp_pdf, color=COLORS["secondary"], linewidth=2, label="Marchenko-Pastur (noise)")

    # 噪声边界
    ax.axvline(mp_lambda_max, color=COLORS["warn"], linestyle="--", linewidth=1.5,
              label=f"lambda_max = {mp_lambda_max:.2f}")

    # 标注信号特征值
    signal_eigs = eigenvalues[eigenvalues > mp_lambda_max]
    if len(signal_eigs) > 0:
        for i, ev in enumerate(sorted(signal_eigs, reverse=True)[:5]):
            ax.annotate(f"Signal {i+1}\nlambda={ev:.1f}",
                       xy=(ev, 0), xytext=(ev, ax.get_ylim()[1] * 0.3 * (1 - i * 0.15)),
                       arrowprops={"arrowstyle": "->", "color": COLORS["warn"]},
                       color=COLORS["warn"], fontsize=8, ha="center")

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(f"RMT Eigenvalue Distribution ({n_assets} assets x {n_observations} days)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_vol_timing(
    returns: pd.Series,
    weights: pd.Series,
    strategy_returns: pd.Series,
    ticker: str = "",
    save_path: str | None = None,
) -> None:
    """波动率择时策略：买入持有 vs Vol Timing"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    common = returns.index.intersection(strategy_returns.index)
    raw_cum = (1 + returns.loc[common]).cumprod()
    strat_cum = (1 + strategy_returns.loc[common]).cumprod()

    ax1.plot(raw_cum.index, raw_cum.values, color=COLORS["neutral"], linewidth=1, label="Buy & Hold", alpha=0.7)
    ax1.plot(strat_cum.index, strat_cum.values, color=COLORS["primary"], linewidth=1.2, label="Vol Timing")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(f"{ticker} Volatility Timing Strategy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：权重
    w = weights.loc[common]
    ax2.fill_between(w.index, 0, w.values, alpha=0.4, color=COLORS["accent"])
    ax2.plot(w.index, w.values, color=COLORS["accent"], linewidth=0.6)
    ax2.axhline(1.0, color=COLORS["neutral"], linestyle="--", alpha=0.5, label="1x (no leverage)")
    ax2.set_ylabel("Position Weight")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    _save_or_show(fig, save_path)


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: list[str],
    title: str = "Correlation Matrix",
    save_path: str | None = None,
) -> None:
    """相关矩阵热力图（用于 RMT 去噪前后对比）"""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)

    _save_or_show(fig, save_path)


def plot_full_dashboard(
    returns: pd.Series,
    ticker: str = "",
    save_path: str | None = None,
) -> None:
    """单只股票的四合一 Dashboard"""
    r = returns.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) 收益率时序
    ax = axes[0, 0]
    ax.bar(r.index, r.values, width=1,
           color=np.where(r.values >= 0, COLORS["primary"], COLORS["secondary"]), alpha=0.6)
    ax.set_title(f"{ticker} Daily Returns")
    ax.grid(True, alpha=0.3)

    # (0,1) QQ-Plot
    ax = axes[0, 1]
    (theoretical, sample), (slope, intercept, _) = stats.probplot(r.values, dist="norm")
    expected = slope * theoretical + intercept
    dev = np.abs(sample - expected)
    sizes = 4 + 60 * (dev / (dev.max() + 1e-9))
    ax.scatter(theoretical, sample, s=sizes, alpha=0.5, color=COLORS["primary"],
               edgecolors="white", linewidths=0.2)
    x_line = np.array([theoretical.min(), theoretical.max()])
    ax.plot(x_line, slope * x_line + intercept, "--", color=COLORS["secondary"], linewidth=1.5)
    ax.set_title("QQ-Plot vs Normal")
    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Sample")
    ax.grid(True, alpha=0.3)

    # (1,0) 分布直方图
    ax = axes[1, 0]
    ax.hist(r.values, bins=80, density=True, alpha=0.6, color=COLORS["primary"])
    x = np.linspace(r.min(), r.max(), 200)
    mu, sigma = r.mean(), r.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "--", color=COLORS["secondary"], linewidth=1.5, label="Normal")
    ax.set_title("Return Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) 滚动波动率
    ax = axes[1, 1]
    rolling_vol = r.rolling(20).std() * np.sqrt(252)
    ax.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.4, color=COLORS["accent"])
    ax.plot(rolling_vol.index, rolling_vol.values, color=COLORS["accent"], linewidth=0.6)
    ax.set_title("Rolling 20d Annualized Vol")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ticker} MarketDNA Dashboard", fontsize=14, fontweight="bold", y=1.02)
    _save_or_show(fig, save_path)
