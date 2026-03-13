"""
相关性与协整分析 — 发现"假朋友"和"真CP"

★ 相关性 vs 协整：金融里最重要的区分之一
  - 相关性：两只股票的收益率方向一致（"今天一起涨一起跌"）
  - 协整：两只股票的价格差维持在一个均值附近（"价差会回归"）

  两个高相关但不协整的股票做配对交易 → 可能爆仓
  两个低相关但协整的股票 → 反而可以赚价差回归的钱

  这就是为什么"相关性 ≠ 可交易关系"

★ Random Matrix Theory（RMT）去噪
  500只股票的相关矩阵有 500×500 = 250,000 个元素
  但其中 80%+ 的特征值只是噪声！
  用 Marchenko-Pastur 定律可以识别噪声特征值并去除
  这是顶级量化基金（Capital Fund Management等）的核心技术
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class PairFingerprint:
    """两只股票之间关系的指纹"""

    ticker_a: str
    ticker_b: str

    # 相关性
    return_corr: float         # 收益率相关系数
    rolling_corr_std: float    # 滚动相关的标准差（越大说明关系越不稳定）
    corr_regime_range: float   # 滚动相关的极差（最大-最小）

    # 协整检验 (Engle-Granger)
    is_cointegrated: bool
    coint_pval: float
    coint_stat: float

    # 价差（Spread）统计
    spread_mean: float
    spread_std: float
    spread_half_life: float    # 均值回复半衰期（天数）
    spread_current_zscore: float  # 当前价差偏离均值几个标准差

    @property
    def pair_verdict(self) -> str:
        if self.is_cointegrated and self.spread_half_life < 30:
            return "优质配对（协整+快速回归）"
        if self.is_cointegrated:
            return "协整但回归慢"
        if self.return_corr > 0.8:
            return "高相关但不协整（假CP）"
        return "无显著关系"


def analyze_pair(
    returns_a: pd.Series,
    returns_b: pd.Series,
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
) -> PairFingerprint:
    """分析两只股票之间的关系"""

    # 对齐索引
    common_idx = returns_a.index.intersection(returns_b.index)
    ra = returns_a.loc[common_idx].dropna()
    rb = returns_b.loc[common_idx].dropna()
    common_idx = ra.index.intersection(rb.index)
    ra = ra.loc[common_idx]
    rb = rb.loc[common_idx]

    pa = prices_a.loc[common_idx]
    pb = prices_b.loc[common_idx]

    # --- 相关性分析 ---
    ret_corr = float(ra.corr(rb))

    # 滚动相关（60日窗口）
    rolling_corr = ra.rolling(60).corr(rb).dropna()
    rolling_corr_std = float(rolling_corr.std()) if len(rolling_corr) > 0 else 0.0
    corr_range = float(rolling_corr.max() - rolling_corr.min()) if len(rolling_corr) > 0 else 0.0

    # --- 协整检验 ---
    # Engle-Granger两步法
    try:
        coint_stat_val, coint_pval_val, _ = coint(pa, pb)
        is_coint = coint_pval_val < 0.05
    except Exception:
        coint_stat_val, coint_pval_val, is_coint = 0.0, 1.0, False

    # --- 价差分析 ---
    # OLS回归得到hedge ratio
    if len(pa) > 0 and len(pb) > 0:
        beta = float(np.polyfit(pb.values, pa.values, 1)[0])
        spread = pa - beta * pb
    else:
        spread = pd.Series(dtype=float)
        beta = 1.0

    spread_mean = float(spread.mean()) if len(spread) > 0 else 0.0
    spread_std = float(spread.std()) if len(spread) > 0 else 1.0

    # 均值回复半衰期：用AR(1)系数估计
    if len(spread) > 2:
        spread_lag = spread.shift(1).dropna()
        spread_now = spread.iloc[1:]
        common = spread_lag.index.intersection(spread_now.index)
        if len(common) > 10:
            slope, _, _, _, _ = stats.linregress(spread_lag.loc[common], spread_now.loc[common])
            if 0 < slope < 1:
                half_life = -np.log(2) / np.log(slope)
            else:
                half_life = float("inf")
        else:
            half_life = float("inf")
    else:
        half_life = float("inf")

    # 当前z-score
    current_zscore = float((spread.iloc[-1] - spread_mean) / spread_std) if len(spread) > 0 and spread_std > 0 else 0.0

    return PairFingerprint(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        return_corr=ret_corr,
        rolling_corr_std=rolling_corr_std,
        corr_regime_range=corr_range,
        is_cointegrated=is_coint,
        coint_pval=float(coint_pval_val),
        coint_stat=float(coint_stat_val),
        spread_mean=spread_mean,
        spread_std=spread_std,
        spread_half_life=half_life,
        spread_current_zscore=current_zscore,
    )


def print_pair(pf: PairFingerprint) -> None:
    """人话版配对报告"""

    print(f"\n{'='*60}")
    print(f"  🔗 {pf.ticker_a} × {pf.ticker_b} 配对指纹")
    print(f"{'='*60}")
    print()
    print(f"  相关性:")
    print(f"    收益率相关:     {pf.return_corr:+.3f}  ← {'高' if abs(pf.return_corr) > 0.7 else '中' if abs(pf.return_corr) > 0.4 else '低'}相关")
    print(f"    相关稳定性:     σ={pf.rolling_corr_std:.3f}  ← {'不稳定' if pf.rolling_corr_std > 0.15 else '相对稳定'}")
    print(f"    相关极差:       {pf.corr_regime_range:.3f}")
    print()
    print(f"  协整检验 (Engle-Granger):")
    print(f"    p值:  {pf.coint_pval:.4f}  {'✅ 协整！可以做配对交易' if pf.is_cointegrated else '❌ 不协整'}")
    print()

    if pf.is_cointegrated:
        print(f"  价差 (Spread) 分析:")
        print(f"    均值回复半衰期: {pf.spread_half_life:.1f} 天  ← {'快（适合交易）' if pf.spread_half_life < 20 else '中等' if pf.spread_half_life < 60 else '慢（风险高）'}")
        print(f"    当前Z-Score:    {pf.spread_current_zscore:+.2f}  ← {'偏离大，可能有机会！' if abs(pf.spread_current_zscore) > 2 else '正常范围'}")
    print()
    print(f"  综合判断: {pf.pair_verdict}")
    print(f"{'='*60}\n")


def find_cointegrated_pairs(
    data_dict: dict[str, pd.Series],
    price_dict: dict[str, pd.Series],
    max_pval: float = 0.05,
) -> list[PairFingerprint]:
    """在一组股票中搜索所有协整对

    注意：这里存在多重检验问题！
    如果有N只股票，就有 N*(N-1)/2 个配对
    5%显著性下会有 ~5% 的假阳性
    """
    tickers = list(data_dict.keys())
    pairs = []

    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            pf = analyze_pair(
                data_dict[t1], data_dict[t2],
                price_dict[t1], price_dict[t2],
                t1, t2,
            )
            if pf.coint_pval < max_pval:
                pairs.append(pf)

    # 按p值排序
    return sorted(pairs, key=lambda p: p.coint_pval)
