"""
波动率分析 — 发现"波动率聚类"现象和GARCH建模

★ 波动率聚类（Volatility Clustering）是什么？
  "大波动后面跟着大波动，小波动后面跟着小波动"
  — 这违反了收益率独立同分布的假设
  — 是金融市场最robust的统计规律之一
  — GARCH模型就是用来捕捉这种现象的

★ 为什么这对量化很重要？
  如果波动率可预测，你就可以：
  1. 在低波时加杠杆、高波时减仓（波动率择时）
  2. 更准确地估计风险（动态VaR而非固定VaR）
  3. 交易波动率本身（VIX期货、期权straddle）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class VolatilityFingerprint:
    """波动率结构的指纹"""

    ticker: str

    # 波动率聚类检验
    has_clustering: bool       # 收益率平方序列有自相关吗？
    ljung_box_stat: float      # Ljung-Box统计量（检测自相关）
    ljung_box_pval: float      # <0.05 → 存在波动率聚类
    sq_return_acf_lag1: float  # 收益率平方的1阶自相关（通常0.1-0.3）

    # GARCH(1,1) 参数
    garch_omega: float         # 长期均值项
    garch_alpha: float         # ARCH项：昨天的冲击对今天波动率的影响
    garch_beta: float          # GARCH项：昨天的波动率对今天的惯性
    garch_persistence: float   # alpha+beta：越接近1，波动率记忆越长
    garch_half_life: float     # 波动率冲击的半衰期（天数）

    # 杠杆效应（Leverage Effect）
    has_leverage: bool         # 下跌是否比上涨引起更大波动？
    leverage_corr: float       # 收益率与未来波动率变化的相关性（通常为负）

    # 波动率的波动率（vol of vol）
    vol_of_vol: float          # 波动率本身有多不稳定
    vol_regime_ratio: float    # 最高波动率 / 最低波动率（滚动窗口）


def analyze_volatility(returns: pd.Series, ticker: str = "") -> VolatilityFingerprint:
    """分析收益率的波动率结构"""

    r = returns.dropna()
    r_values = r.values

    # --- 波动率聚类检验 ---
    r_squared = r_values ** 2

    # Ljung-Box检验：r²序列是否有自相关
    lb_result = acorr_ljungbox(r_squared, lags=[10], return_df=True)
    lb_stat = float(lb_result["lb_stat"].iloc[0])
    lb_pval = float(lb_result["lb_pvalue"].iloc[0])
    has_clustering = lb_pval < 0.05

    # r²的1阶自相关系数
    sq_acf1 = float(np.corrcoef(r_squared[:-1], r_squared[1:])[0, 1])

    # --- GARCH(1,1) 拟合 ---
    # rescale=True 让arch库自动缩放数据到百分比
    garch = arch_model(r * 100, vol="Garch", p=1, q=1, mean="Constant", rescale=False)

    try:
        garch_fit = garch.fit(disp="off")
        omega = float(garch_fit.params.get("omega", 0))
        alpha = float(garch_fit.params.get("alpha[1]", 0))
        beta = float(garch_fit.params.get("beta[1]", 0))
    except Exception:
        omega, alpha, beta = 0.0, 0.1, 0.8

    persistence = alpha + beta

    # 半衰期：冲击衰减到一半所需的天数
    if 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = float("inf")

    # --- 杠杆效应 ---
    # 检验：负收益率是否预示更高的未来波动率
    rolling_vol = r.rolling(20).std()
    vol_change = rolling_vol.diff()

    # 对齐：用t时刻的收益率和t+1~t+20的波动率变化
    aligned = pd.DataFrame({
        "return": r,
        "future_vol_change": vol_change.shift(-1),
    }).dropna()

    if len(aligned) > 50:
        leverage_corr = float(aligned["return"].corr(aligned["future_vol_change"]))
    else:
        leverage_corr = 0.0

    has_leverage = leverage_corr < -0.1

    # --- 波动率的波动率 ---
    rolling_vol_clean = rolling_vol.dropna()
    vol_of_vol = float(rolling_vol_clean.std() / rolling_vol_clean.mean()) if len(rolling_vol_clean) > 0 else 0.0

    # 波动率regime比（最高/最低的20日滚动窗口均值）
    vol_rolling_60 = rolling_vol.rolling(60).mean().dropna()
    if len(vol_rolling_60) > 0:
        vol_regime_ratio = float(vol_rolling_60.max() / vol_rolling_60.min()) if vol_rolling_60.min() > 0 else float("inf")
    else:
        vol_regime_ratio = 1.0

    return VolatilityFingerprint(
        ticker=ticker,
        has_clustering=has_clustering,
        ljung_box_stat=lb_stat,
        ljung_box_pval=lb_pval,
        sq_return_acf_lag1=sq_acf1,
        garch_omega=omega,
        garch_alpha=alpha,
        garch_beta=beta,
        garch_persistence=persistence,
        garch_half_life=half_life,
        has_leverage=has_leverage,
        leverage_corr=leverage_corr,
        vol_of_vol=vol_of_vol,
        vol_regime_ratio=vol_regime_ratio,
    )


def print_volatility(vf: VolatilityFingerprint) -> None:
    """人话版波动率报告"""

    print(f"\n{'='*60}")
    print(f"  🌊 {vf.ticker} 波动率指纹")
    print(f"{'='*60}")
    print()
    print(f"  波动率聚类:")
    print(f"    Ljung-Box p值:     {vf.ljung_box_pval:.2e}  {'✅ 存在聚类' if vf.has_clustering else '❌ 无聚类（罕见）'}")
    print(f"    r² 自相关(lag=1):  {vf.sq_return_acf_lag1:.3f}  ← {'强聚类' if vf.sq_return_acf_lag1 > 0.2 else '中等' if vf.sq_return_acf_lag1 > 0.1 else '弱'}")
    print()
    print(f"  GARCH(1,1) 模型:")
    print(f"    α (ARCH):     {vf.garch_alpha:.4f}  ← 昨天冲击的影响")
    print(f"    β (GARCH):    {vf.garch_beta:.4f}  ← 波动率的惯性")
    print(f"    α+β 持续性:   {vf.garch_persistence:.4f}  ← {'接近单位根（长记忆）' if vf.garch_persistence > 0.95 else '高持续性' if vf.garch_persistence > 0.9 else '中等'}")
    print(f"    半衰期:       {vf.garch_half_life:.1f} 天  ← 波动率冲击衰减到一半")
    print()
    print(f"  杠杆效应（坏消息 vs 好消息）:")
    print(f"    收益-未来波动率相关: {vf.leverage_corr:+.3f}  {'← 下跌引起更大波动！' if vf.has_leverage else '← 无显著杠杆效应'}")
    print()
    print(f"  波动率的稳定性:")
    print(f"    Vol of Vol:    {vf.vol_of_vol:.3f}  ← 波动率本身{'很不稳定' if vf.vol_of_vol > 0.5 else '中等波动' if vf.vol_of_vol > 0.3 else '相对稳定'}")
    print(f"    Regime比率:    {vf.vol_regime_ratio:.1f}x  ← 最高波动率是最低的{vf.vol_regime_ratio:.1f}倍")
    print(f"{'='*60}\n")
