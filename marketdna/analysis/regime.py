"""
Regime 检测 — 市场在不同"状态"之间切换

★ 什么是 Regime？
  市场不是永远一样的——有时候是"安静低波"的牛市，
  有时候是"恐慌高波"的熊市，有时候是"方向不明"的震荡市。
  这些不同的市场状态叫做 regime。

  为什么这对量化很重要？
  1. 同一个策略在不同 regime 下表现天差地别
     （动量策略在趋势市赚钱，在震荡市亏钱）
  2. 如果你能检测到 regime 切换，就可以动态调整策略
  3. 很多"回测里赚钱但实盘亏钱"的策略，
     就是因为训练数据和测试数据处于不同 regime

★ Hidden Markov Model (HMM) 是什么？
  假设市场有 K 个隐藏状态（你看不见），
  每个状态下收益率服从不同的正态分布（均值和方差不同）。
  市场在状态之间随机切换，切换概率是一个转移矩阵。
  HMM 的任务：从你能观察到的收益率序列，反推每个时刻最可能处于哪个状态。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeFingerprint:
    """市场状态指纹"""

    ticker: str
    n_regimes: int

    # 每个 regime 的统计特性
    regime_means: list[float]        # 各 regime 的年化均值收益
    regime_vols: list[float]         # 各 regime 的年化波动率
    regime_durations: list[float]    # 各 regime 的平均持续天数

    # 转移矩阵
    transition_matrix: np.ndarray    # K×K 状态转移概率

    # 当前状态
    current_regime: int
    current_regime_prob: float       # 当前状态的置信度

    # 状态序列（用于画图）
    regime_labels: pd.Series         # 每天的 regime 标签
    regime_probs: pd.DataFrame       # 每天每个 regime 的概率

    @property
    def regime_names(self) -> list[str]:
        """Name each regime by volatility rank"""
        vol_order = np.argsort(self.regime_vols)
        names = [""] * self.n_regimes
        labels = ["Calm", "Choppy", "Panic", "Crisis"]
        for rank, idx in enumerate(vol_order):
            name_idx = min(rank, len(labels) - 1)
            names[idx] = labels[name_idx]
        return names


def _smooth_labels(labels: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """Smooth regime labels by removing short-lived segments.

    Single-pass: identify all segments, then replace any segment
    shorter than min_duration with its left neighbor's label.
    """
    smoothed = labels.copy()
    n = len(smoothed)

    # Step 1: identify segment boundaries on the original labels
    segments: list[tuple[int, int, int]] = []  # (start, end, label)
    i = 0
    while i < n:
        j = i + 1
        while j < n and smoothed[j] == smoothed[i]:
            j += 1
        segments.append((i, j, int(smoothed[i])))
        i = j

    # Step 2: merge short segments into the previous segment
    for idx in range(1, len(segments)):
        start, end, val = segments[idx]
        if (end - start) < min_duration:
            prev_val = segments[idx - 1][2]
            smoothed[start:end] = prev_val
            # Update this segment's label so subsequent merges chain correctly
            segments[idx] = (start, end, prev_val)

    return smoothed


def analyze_regime(
    returns: pd.Series,
    ticker: str = "",
    n_regimes: int = 2,
    min_regime_days: int = 3,
) -> RegimeFingerprint:
    """Detect market regimes using Hidden Markov Model

    Parameters
    ----------
    returns : pd.Series
        Daily log returns
    n_regimes : int
        Number of hidden states (2=bull/bear, 3=bull/choppy/bear)
    min_regime_days : int
        Minimum regime duration in days. Shorter segments are merged
        into the previous regime to prevent unrealistic oscillations.
    """
    from hmmlearn.hmm import GaussianHMM

    r = returns.dropna()
    X = r.values.reshape(-1, 1)

    # Train HMM
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    # Predict states
    labels = model.predict(X)
    probs = model.predict_proba(X)

    # Smooth out short-lived regime flickers
    if min_regime_days > 1:
        labels = _smooth_labels(labels, min_duration=min_regime_days)

    # 提取每个 regime 的参数
    means_daily = model.means_.flatten()
    vols_daily = np.sqrt(model.covars_.flatten())

    means_annual = (means_daily * 252).tolist()
    vols_annual = (vols_daily * np.sqrt(252)).tolist()

    # 计算每个 regime 的平均持续天数
    durations = []
    for k in range(n_regimes):
        # 找到连续处于该 regime 的片段
        mask = labels == k
        runs = []
        count = 0
        for v in mask:
            if v:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        durations.append(float(np.mean(runs)) if runs else 0.0)

    # 构建 regime_labels Series（与原始 index 对齐）
    regime_series = pd.Series(labels, index=r.index, name="regime")
    probs_df = pd.DataFrame(
        probs,
        index=r.index,
        columns=[f"regime_{i}" for i in range(n_regimes)],
    )

    return RegimeFingerprint(
        ticker=ticker,
        n_regimes=n_regimes,
        regime_means=means_annual,
        regime_vols=vols_annual,
        regime_durations=durations,
        transition_matrix=model.transmat_,
        current_regime=int(labels[-1]),
        current_regime_prob=float(probs[-1, labels[-1]]),
        regime_labels=regime_series,
        regime_probs=probs_df,
    )


def print_regime(rf: RegimeFingerprint) -> None:
    """Human-readable regime report"""

    print(f"\n{'='*60}")
    print(f"  Regime Fingerprint: {rf.ticker} ({rf.n_regimes} states)")
    print(f"{'='*60}")

    names = rf.regime_names
    for i in range(rf.n_regimes):
        pct_time = float((rf.regime_labels == i).mean() * 100)
        print(f"\n  Regime {i}: {names[i]}")
        print(f"    Ann. Return:   {rf.regime_means[i]:+.1%}")
        print(f"    Ann. Vol:      {rf.regime_vols[i]:.1%}")
        print(f"    Avg Duration:  {rf.regime_durations[i]:.0f} days")
        print(f"    Time Share:    {pct_time:.1f}%")

    print(f"\n  Transition Matrix:")
    for i in range(rf.n_regimes):
        row = " -> ".join(
            f"{rf.transition_matrix[i, j]:.2%}" for j in range(rf.n_regimes)
        )
        print(f"    {names[i]:>8s}: {row}")

    cur = rf.current_regime
    print(f"\n  Current State: Regime {cur} ({names[cur]})")
    print(f"  Confidence:    {rf.current_regime_prob:.1%}")
    print(f"{'='*60}\n")
