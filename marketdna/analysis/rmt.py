"""
Random Matrix Theory (RMT) 相关矩阵去噪

★ 为什么需要去噪？
  假设你有 500 只股票、3 年日度数据（~750天）。
  相关矩阵是 500×500 = 250,000 个参数，
  但你只有 750×500 = 375,000 个数据点来估计它们。
  结果：大部分"相关性"其实只是噪声！

★ Marchenko-Pastur 定律
  如果收益率完全随机（没有任何真实相关性），
  样本相关矩阵的特征值应该服从 Marchenko-Pastur 分布：
    λ_max = σ²(1 + √(N/T))²
    λ_min = σ²(1 - √(N/T))²
  其中 N=股票数, T=观测天数, σ²=方差。
  任何超过 λ_max 的特征值才是"真实信号"，其余都是噪声。

★ 这有什么用？
  去噪后的相关矩阵用于：
  1. 更稳健的投资组合优化（最小方差组合不再把噪声放大）
  2. 更准确的风险评估
  3. 发现真实的市场因子结构（超过噪声阈值的特征向量 = 真实因子）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RMTFingerprint:
    """相关矩阵去噪结果"""

    n_assets: int
    n_observations: int
    q_ratio: float                 # N/T 比率（越大噪声越多）

    # Marchenko-Pastur 边界
    mp_lambda_max: float           # 噪声特征值的理论上界
    mp_lambda_min: float           # 噪声特征值的理论下界

    # 特征值分析
    n_signal_eigenvalues: int      # 超过噪声阈值的特征值个数（= 真实因子数）
    n_noise_eigenvalues: int       # 噪声特征值个数
    signal_ratio: float            # 信号特征值占总方差的比例

    # 前几个真实因子
    top_eigenvalues: list[float]   # 最大的几个信号特征值
    variance_explained_top1: float  # 第一因子解释的方差比例（通常=市场因子）

    # 去噪前后对比
    raw_condition_number: float    # 原始矩阵条件数（越大越不稳定）
    denoised_condition_number: float


def analyze_rmt(
    returns_df: pd.DataFrame,
    n_top: int = 5,
) -> tuple[RMTFingerprint, np.ndarray, np.ndarray]:
    """对多只股票的收益率矩阵做 RMT 去噪

    Parameters
    ----------
    returns_df : pd.DataFrame
        每列是一只股票的日度收益率
    n_top : int
        展示前几个信号特征值

    Returns
    -------
    fingerprint : RMTFingerprint
    raw_corr : np.ndarray
        原始相关矩阵
    denoised_corr : np.ndarray
        去噪后的相关矩阵
    """
    # 丢弃缺失值（只保留所有股票都有数据的天）
    clean = returns_df.dropna()
    T, N = clean.shape
    q = N / T

    # 样本相关矩阵
    raw_corr = clean.corr().values

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(raw_corr)
    # 从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur 边界
    sigma2 = 1.0  # 相关矩阵的方差=1
    mp_max = sigma2 * (1 + np.sqrt(q)) ** 2
    mp_min = sigma2 * (1 - np.sqrt(q)) ** 2

    # 分类：信号 vs 噪声
    signal_mask = eigenvalues > mp_max
    n_signal = int(np.sum(signal_mask))
    n_noise = N - n_signal

    signal_var_ratio = float(np.sum(eigenvalues[signal_mask]) / np.sum(eigenvalues))

    # 去噪：把噪声特征值替换为均值（保持矩阵的trace不变）
    denoised_eigenvalues = eigenvalues.copy()
    noise_eigenvalues = denoised_eigenvalues[~signal_mask]
    if len(noise_eigenvalues) > 0:
        noise_mean = noise_eigenvalues.mean()
        denoised_eigenvalues[~signal_mask] = noise_mean

    # 重构去噪后的相关矩阵
    denoised_corr = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
    # 强制对角线为1（相关矩阵性质）
    d = np.sqrt(np.diag(denoised_corr))
    d[d == 0] = 1
    denoised_corr = denoised_corr / np.outer(d, d)
    np.fill_diagonal(denoised_corr, 1.0)

    # 条件数
    raw_cond = float(eigenvalues[0] / max(eigenvalues[-1], 1e-10))
    denoised_cond = float(
        denoised_eigenvalues[0] / max(denoised_eigenvalues[-1], 1e-10)
    )

    top_eigs = eigenvalues[:n_top].tolist()
    var_top1 = float(eigenvalues[0] / np.sum(eigenvalues))

    fp = RMTFingerprint(
        n_assets=N,
        n_observations=T,
        q_ratio=q,
        mp_lambda_max=float(mp_max),
        mp_lambda_min=float(mp_min),
        n_signal_eigenvalues=n_signal,
        n_noise_eigenvalues=n_noise,
        signal_ratio=signal_var_ratio,
        top_eigenvalues=top_eigs,
        variance_explained_top1=var_top1,
        raw_condition_number=raw_cond,
        denoised_condition_number=denoised_cond,
    )

    return fp, raw_corr, denoised_corr


def print_rmt(fp: RMTFingerprint) -> None:
    """人话版 RMT 报告"""

    print(f"\n{'='*60}")
    print(f"  🔬 RMT 相关矩阵去噪 ({fp.n_assets}只股票 × {fp.n_observations}天)")
    print(f"{'='*60}")
    print()
    print(f"  N/T 比率:  {fp.q_ratio:.3f}  ← {'噪声多（N接近T）' if fp.q_ratio > 0.5 else '数据充足' if fp.q_ratio < 0.1 else '中等'}")
    print()
    print(f"  Marchenko-Pastur 噪声边界:")
    print(f"    λ_max = {fp.mp_lambda_max:.3f}  ← 超过这个的特征值才是真信号")
    print(f"    λ_min = {fp.mp_lambda_min:.3f}")
    print()
    print(f"  特征值分类:")
    print(f"    信号特征值: {fp.n_signal_eigenvalues} 个  ← 真实的市场因子数量")
    print(f"    噪声特征值: {fp.n_noise_eigenvalues} 个  ← 全是随机噪声")
    print(f"    信号占比:   {fp.signal_ratio:.1%}")
    print(f"    噪声占比:   {1-fp.signal_ratio:.1%}  ← 相关矩阵中{1-fp.signal_ratio:.0%}是噪声！")
    print()
    print(f"  最大特征值（真实因子）:")
    for i, ev in enumerate(fp.top_eigenvalues):
        label = "（市场因子）" if i == 0 else ""
        marker = "★" if ev > fp.mp_lambda_max else " "
        print(f"    {marker} λ_{i+1} = {ev:.2f}  {label}")
    print(f"    第一因子解释方差: {fp.variance_explained_top1:.1%}  ← {'强市场因子' if fp.variance_explained_top1 > 0.3 else '中等'}")
    print()
    print(f"  矩阵稳定性（条件数，越小越好）:")
    print(f"    去噪前: {fp.raw_condition_number:.0f}")
    print(f"    去噪后: {fp.denoised_condition_number:.0f}  ← 改善{fp.raw_condition_number/max(fp.denoised_condition_number,1):.1f}倍")
    print(f"{'='*60}\n")
