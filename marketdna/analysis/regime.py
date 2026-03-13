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
        """根据波动率大小给 regime 起名字"""
        vol_order = np.argsort(self.regime_vols)
        names = [""] * self.n_regimes
        labels = ["低波平静", "中波震荡", "高波恐慌", "极端危机"]
        for rank, idx in enumerate(vol_order):
            name_idx = min(rank, len(labels) - 1)
            names[idx] = labels[name_idx]
        return names


def analyze_regime(
    returns: pd.Series,
    ticker: str = "",
    n_regimes: int = 3,
) -> RegimeFingerprint:
    """用 HMM 检测市场 regime

    Parameters
    ----------
    returns : pd.Series
        日度 log 收益率
    n_regimes : int
        假设的状态数量（2=牛/熊，3=牛/震荡/熊，推荐3）
    """
    from hmmlearn.hmm import GaussianHMM

    r = returns.dropna()
    X = r.values.reshape(-1, 1)

    # 训练 HMM
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    # 预测每天的状态
    labels = model.predict(X)
    probs = model.predict_proba(X)

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
    """人话版 regime 报告"""

    print(f"\n{'='*60}")
    print(f"  🎭 {rf.ticker} Regime 指纹 ({rf.n_regimes}个状态)")
    print(f"{'='*60}")

    names = rf.regime_names
    for i in range(rf.n_regimes):
        pct_time = float((rf.regime_labels == i).mean() * 100)
        print(f"\n  Regime {i}: {names[i]}")
        print(f"    年化收益:  {rf.regime_means[i]:+.1%}")
        print(f"    年化波动:  {rf.regime_vols[i]:.1%}")
        print(f"    平均持续:  {rf.regime_durations[i]:.0f} 天")
        print(f"    时间占比:  {pct_time:.1f}%")

    print(f"\n  转移概率矩阵:")
    for i in range(rf.n_regimes):
        row = "    " + " → ".join(
            f"{rf.transition_matrix[i, j]:.2%}" for j in range(rf.n_regimes)
        )
        print(f"    从 {names[i]}: {row}")

    cur = rf.current_regime
    print(f"\n  📍 当前状态: Regime {cur} ({names[cur]})")
    print(f"     置信度:   {rf.current_regime_prob:.1%}")
    print(f"{'='*60}\n")
