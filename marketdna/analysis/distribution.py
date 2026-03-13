"""
分布分析 — 打破"收益率是正态分布"的幻觉

★ 为什么这很重要？
  金融世界的大部分风险管理模型（VaR、Black-Scholes）都假设正态分布。
  但真实市场的日收益率尾部比正态分布厚 3-10 倍。
  2008年金融危机中，"六个标准差"事件每天都在发生——
  如果真是正态分布，这种事情要几百万年才会发生一次。
  理解这一点，就理解了为什么VaR模型会失败。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DistributionFingerprint:
    """一只股票收益率分布的"指纹" """

    ticker: str
    n_obs: int

    # 基本矩
    mean_annual: float        # 年化均值
    vol_annual: float         # 年化波动率
    skewness: float           # 偏度：<0 表示左尾更厚（跌得猛）
    excess_kurtosis: float    # 超额峰度：>0 表示尾部比正态厚

    # 正态性检验
    jarque_bera_stat: float
    jarque_bera_pval: float   # <0.05 → 拒绝正态假设
    shapiro_stat: float
    shapiro_pval: float

    # 尾部分析
    left_tail_5pct: float     # 5%分位数（最差情况）
    right_tail_95pct: float   # 95%分位数（最好情况）
    normal_left_5pct: float   # 如果是正态分布，5%分位数应该是多少
    tail_ratio: float         # 真实左尾 / 正态左尾，>1说明尾部更厚

    # Student-t 拟合
    t_df: float               # 自由度：越小尾巴越厚，3-5很常见
    t_loc: float
    t_scale: float

    # 极端事件
    n_beyond_3sigma: int      # 超过3σ的天数
    pct_beyond_3sigma: float  # 正态分布下应该只有0.27%
    max_daily_loss: float
    max_daily_gain: float

    @property
    def is_normal(self) -> bool:
        """JB检验是否拒绝正态假设"""
        return self.jarque_bera_pval > 0.05

    @property
    def tail_fatness_verdict(self) -> str:
        """尾部厚度的人话翻译"""
        k = self.excess_kurtosis
        if k < 1:
            return "接近正态（罕见！）"
        if k < 3:
            return "轻度厚尾（普通股票）"
        if k < 10:
            return "中度厚尾（波动大的股票）"
        return "极度厚尾（极端波动，小心！）"


def analyze_distribution(returns: pd.Series, ticker: str = "") -> DistributionFingerprint:
    """对收益率序列做完整的分布指纹分析"""

    r = returns.dropna().values
    n = len(r)

    # 基本统计量
    mean_daily = float(np.mean(r))
    std_daily = float(np.std(r, ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))  # scipy默认就是excess kurtosis

    # 正态性检验
    jb_stat, jb_pval = stats.jarque_bera(r)

    # Shapiro-Wilk（样本量大时取子集，因为它限制5000个样本）
    sample_for_shapiro = r if n <= 5000 else np.random.default_rng(42).choice(r, 5000, replace=False)
    sw_stat, sw_pval = stats.shapiro(sample_for_shapiro)

    # 尾部分析
    left_5 = float(np.percentile(r, 5))
    right_95 = float(np.percentile(r, 95))
    normal_left_5 = mean_daily + stats.norm.ppf(0.05) * std_daily
    tail_ratio = abs(left_5 / normal_left_5) if normal_left_5 != 0 else float("inf")

    # Student-t 拟合 (MLE)
    t_df, t_loc, t_scale = stats.t.fit(r)

    # 极端事件统计
    threshold_3sigma = 3 * std_daily
    beyond_3sigma = int(np.sum(np.abs(r - mean_daily) > threshold_3sigma))
    pct_beyond = beyond_3sigma / n * 100

    return DistributionFingerprint(
        ticker=ticker,
        n_obs=n,
        mean_annual=mean_daily * 252,
        vol_annual=std_daily * np.sqrt(252),
        skewness=skew,
        excess_kurtosis=kurt,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pval=float(jb_pval),
        shapiro_stat=float(sw_stat),
        shapiro_pval=float(sw_pval),
        left_tail_5pct=left_5,
        right_tail_95pct=right_95,
        normal_left_5pct=normal_left_5,
        tail_ratio=tail_ratio,
        t_df=float(t_df),
        t_loc=float(t_loc),
        t_scale=float(t_scale),
        n_beyond_3sigma=beyond_3sigma,
        pct_beyond_3sigma=pct_beyond,
        max_daily_loss=float(np.min(r)),
        max_daily_gain=float(np.max(r)),
    )


def print_fingerprint(fp: DistributionFingerprint) -> None:
    """把指纹翻译成人话"""

    normal_expected_3sigma = 0.27
    tail_multiple = fp.pct_beyond_3sigma / normal_expected_3sigma if normal_expected_3sigma > 0 else 0

    print(f"\n{'='*60}")
    print(f"  📊 {fp.ticker} 分布指纹  ({fp.n_obs} 个交易日)")
    print(f"{'='*60}")
    print(f"\n  年化收益率:  {fp.mean_annual:+.1%}")
    print(f"  年化波动率:  {fp.vol_annual:.1%}")
    print(f"  偏度:        {fp.skewness:+.3f}  {'← 左偏（跌得更猛）' if fp.skewness < -0.1 else '← 右偏' if fp.skewness > 0.1 else '← 对称'}")
    print(f"  超额峰度:    {fp.excess_kurtosis:.2f}  ← {fp.tail_fatness_verdict}")
    print()
    print(f"  🔬 正态性检验:")
    print(f"     Jarque-Bera p值: {fp.jarque_bera_pval:.2e}  {'✅ 正态' if fp.is_normal else '❌ 拒绝正态'}")
    print(f"     Shapiro-Wilk p值: {fp.shapiro_pval:.2e}")
    print()
    print(f"  📉 尾部分析:")
    print(f"     最差5%的日子:    {fp.left_tail_5pct:+.2%}")
    print(f"     正态分布预测:    {fp.normal_left_5pct:+.2%}")
    print(f"     尾部厚度倍数:    {fp.tail_ratio:.2f}x  ← 真实风险是正态假设的{fp.tail_ratio:.1f}倍")
    print()
    print(f"  ⚡ 极端事件:")
    print(f"     超过3σ的天数:    {fp.n_beyond_3sigma} 天 ({fp.pct_beyond_3sigma:.2f}%)")
    print(f"     正态预测:        {normal_expected_3sigma:.2f}% → 实际是{tail_multiple:.1f}倍")
    print(f"     史上最大单日跌幅: {fp.max_daily_loss:+.2%}")
    print(f"     史上最大单日涨幅: {fp.max_daily_gain:+.2%}")
    print()
    print(f"  📐 Student-t 拟合:")
    print(f"     自由度 ν = {fp.t_df:.1f}  ← {'尾巴很厚（<5）' if fp.t_df < 5 else '中等' if fp.t_df < 10 else '接近正态'}")
    print(f"{'='*60}\n")
