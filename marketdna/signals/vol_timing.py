"""
波动率择时信号 — 用 GARCH 预测波动率来调整仓位

★ 核心逻辑：
  GARCH 可以预测明天的波动率。
  波动率高 → 降低仓位（承担1单位风险能获得的回报更不确定）
  波动率低 → 增加仓位（同样的风险预算可以放更多头寸）

  具体做法：目标波动率策略（Volatility Targeting）
  每天调整仓位 w_t = σ_target / σ_predicted_t
  → 使得组合的已实现波动率接近恒定

★ 这是量化基金最常用的风险管理方法之一
  桥水的 Risk Parity、AQR 的 Managed Futures 都用类似思路
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class VolTimingSignal:
    """波动率择时信号"""

    ticker: str
    target_vol: float              # 目标年化波动率

    # 信号序列
    predicted_vol: pd.Series       # GARCH 预测的每日波动率
    weights: pd.Series             # 仓位权重 = target / predicted
    strategy_returns: pd.Series    # 策略收益 = 权重 × 原始收益

    # 绩效
    raw_sharpe: float
    strategy_sharpe: float
    raw_annual_vol: float
    strategy_annual_vol: float
    raw_max_drawdown: float
    strategy_max_drawdown: float


def _max_drawdown(cumulative: pd.Series) -> float:
    """计算最大回撤"""
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def generate_vol_timing(
    returns: pd.Series,
    ticker: str = "",
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
) -> VolTimingSignal:
    """生成波动率择时信号

    Parameters
    ----------
    target_vol : float
        目标年化波动率（默认10%）
    max_leverage : float
        最大杠杆倍数（限制极端情况）
    """
    r = returns.dropna()

    # 用 GARCH(1,1) 预测波动率
    garch = arch_model(r * 100, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
    fit = garch.fit(disp="off")

    # 条件波动率（样本内拟合值）
    cond_vol_daily = fit.conditional_volatility / 100  # 转回小数
    cond_vol_annual = cond_vol_daily * np.sqrt(252)

    # 构建信号（对齐索引）
    predicted_vol = pd.Series(cond_vol_annual.values, index=r.index, name="predicted_vol")

    # 仓位 = 目标波动率 / 预测波动率
    target_daily = target_vol / np.sqrt(252)
    weights = target_daily / cond_vol_daily
    weights = weights.clip(upper=max_leverage)  # 限制杠杆
    weights = pd.Series(weights.values, index=r.index, name="weight")

    # 策略收益 = 昨天的权重 × 今天的收益（避免前瞻偏差）
    strategy_ret = (weights.shift(1) * r).dropna()
    strategy_ret.name = "strategy_return"

    # 计算绩效
    raw_annual_vol = float(r.std() * np.sqrt(252))
    strat_annual_vol = float(strategy_ret.std() * np.sqrt(252))

    raw_mean = float(r.mean() * 252)
    strat_mean = float(strategy_ret.mean() * 252)

    raw_sharpe = raw_mean / raw_annual_vol if raw_annual_vol > 0 else 0
    strat_sharpe = strat_mean / strat_annual_vol if strat_annual_vol > 0 else 0

    raw_cumulative = (1 + r).cumprod()
    strat_cumulative = (1 + strategy_ret).cumprod()

    return VolTimingSignal(
        ticker=ticker,
        target_vol=target_vol,
        predicted_vol=predicted_vol,
        weights=weights,
        strategy_returns=strategy_ret,
        raw_sharpe=raw_sharpe,
        strategy_sharpe=strat_sharpe,
        raw_annual_vol=raw_annual_vol,
        strategy_annual_vol=strat_annual_vol,
        raw_max_drawdown=_max_drawdown(raw_cumulative),
        strategy_max_drawdown=_max_drawdown(strat_cumulative),
    )


def print_vol_timing(vt: VolTimingSignal) -> None:
    """人话版波动率择时报告"""

    sharpe_improvement = vt.strategy_sharpe - vt.raw_sharpe
    dd_improvement = vt.strategy_max_drawdown - vt.raw_max_drawdown

    print(f"\n{'='*60}")
    print(f"  ⚡ {vt.ticker} 波动率择时策略 (目标vol={vt.target_vol:.0%})")
    print(f"{'='*60}")
    print()
    print(f"  {'指标':<20} {'买入持有':>10} {'Vol Timing':>12} {'改善':>10}")
    print(f"  {'-'*52}")
    print(f"  {'年化波动率':<20} {vt.raw_annual_vol:>10.1%} {vt.strategy_annual_vol:>12.1%} {'✅' if vt.strategy_annual_vol < vt.raw_annual_vol else ''}")
    print(f"  {'Sharpe Ratio':<20} {vt.raw_sharpe:>10.3f} {vt.strategy_sharpe:>12.3f} {sharpe_improvement:>+10.3f}")
    print(f"  {'最大回撤':<20} {vt.raw_max_drawdown:>10.1%} {vt.strategy_max_drawdown:>12.1%} {dd_improvement:>+10.1%}")
    print()
    print(f"  平均仓位: {vt.weights.mean():.2f}x  "
          f"(最高{vt.weights.max():.2f}x, 最低{vt.weights.min():.2f}x)")
    print(f"{'='*60}\n")
