"""
均值回复配对交易信号 — 从协整关系中赚钱

★ 配对交易的逻辑：
  找到两只协整的股票 A 和 B，它们的价差（spread = A - β×B）
  会围绕一个均值波动。

  当价差偏离均值太远时：
  - 价差过高（z-score > 2）→ 做空 A、做多 B（赌价差会回来）
  - 价差过低（z-score < -2）→ 做多 A、做空 B
  - 价差回到均值附近（|z| < 0.5）→ 平仓

★ 为什么这能赚钱？
  如果两只股票真的协整，价差偏离就是暂时的错误定价。
  你赚的是"错误被修正"的钱——这叫统计套利。

★ 风险在哪？
  协整关系可能破裂！两只股票以前的关系不代表未来也成立。
  这就是为什么我们需要止损和风险管理。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairTradingSignal:
    """配对交易信号"""

    ticker_a: str
    ticker_b: str
    hedge_ratio: float             # 对冲比率 β

    # 信号序列
    spread: pd.Series              # 价差 = A - β×B
    zscore: pd.Series              # 标准化后的价差
    positions: pd.Series           # -1/0/+1（对A的仓位方向）
    strategy_returns: pd.Series    # 策略收益

    # 绩效
    n_trades: int                  # 总交易次数
    win_rate: float                # 胜率
    sharpe: float
    max_drawdown: float
    avg_holding_days: float        # 平均持仓天数


def _max_drawdown(cumulative: pd.Series) -> float:
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def generate_pair_signal(
    prices_a: pd.Series,
    prices_b: pd.Series,
    returns_a: pd.Series,
    returns_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> PairTradingSignal:
    """生成配对交易信号

    Parameters
    ----------
    lookback : int
        计算均值和标准差的回看窗口
    entry_z : float
        开仓 z-score 阈值
    exit_z : float
        平仓 z-score 阈值
    stop_z : float
        止损 z-score 阈值
    """
    # 对齐
    common = prices_a.index.intersection(prices_b.index)
    pa = prices_a.loc[common]
    pb = prices_b.loc[common]

    # OLS 回归得到 hedge ratio
    beta = float(np.polyfit(pb.values, pa.values, 1)[0])

    # 价差
    spread = pa - beta * pb
    spread.name = "spread"

    # 滚动 z-score
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    zscore = ((spread - rolling_mean) / rolling_std).dropna()
    zscore.name = "zscore"

    # 生成仓位信号
    positions = pd.Series(0.0, index=zscore.index)
    in_position = 0  # 当前仓位方向

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        prev_pos = in_position

        if in_position == 0:
            # 不在仓 → 检查是否开仓
            if z > entry_z:
                in_position = -1   # 做空 spread（做空A做多B）
            elif z < -entry_z:
                in_position = 1    # 做多 spread（做多A做空B）
        else:
            # 在仓 → 检查是否平仓或止损
            if abs(z) < exit_z:
                in_position = 0    # 均值回复，平仓获利
            elif (in_position == -1 and z > stop_z) or (in_position == 1 and z < -stop_z):
                in_position = 0    # 止损

        positions.iloc[i] = in_position

    # 计算策略收益
    # spread收益 = Δspread / spread_std（标准化后的变化）
    spread_ret = spread.pct_change().loc[positions.index]
    strategy_ret = (positions.shift(1) * spread_ret).dropna()
    strategy_ret.name = "strategy_return"

    # 统计交易
    pos_changes = positions.diff().abs()
    n_trades = int(pos_changes.sum() / 2)  # 每次开平算一笔

    # 胜率（简化：每次从开仓到平仓的收益是否为正）
    trade_returns = []
    current_entry = 0
    cum_ret = 0.0
    for i in range(1, len(positions)):
        if positions.iloc[i] != 0 and positions.iloc[i - 1] == 0:
            current_entry = i
            cum_ret = 0.0
        elif positions.iloc[i] == 0 and positions.iloc[i - 1] != 0:
            cum_ret = float(strategy_ret.iloc[current_entry:i].sum())
            trade_returns.append(cum_ret)

    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0

    # 平均持仓天数
    holding_lengths = []
    current_len = 0
    for pos in positions:
        if pos != 0:
            current_len += 1
        elif current_len > 0:
            holding_lengths.append(current_len)
            current_len = 0
    avg_holding = float(np.mean(holding_lengths)) if holding_lengths else 0.0

    # Sharpe
    if strategy_ret.std() > 0:
        sharpe = float(strategy_ret.mean() / strategy_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    cum = (1 + strategy_ret).cumprod()
    max_dd = _max_drawdown(cum)

    return PairTradingSignal(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=beta,
        spread=spread,
        zscore=zscore,
        positions=positions,
        strategy_returns=strategy_ret,
        n_trades=n_trades,
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_holding_days=avg_holding,
    )


def print_pair_trading(pt: PairTradingSignal) -> None:
    """人话版配对交易报告"""

    print(f"\n{'='*60}")
    print(f"  💹 {pt.ticker_a} × {pt.ticker_b} 配对交易回测")
    print(f"{'='*60}")
    print()
    print(f"  对冲比率 β = {pt.hedge_ratio:.4f}")
    print(f"  (做多1股{pt.ticker_a}同时做空{pt.hedge_ratio:.2f}股{pt.ticker_b})")
    print()
    print(f"  交易统计:")
    print(f"    总交易次数:    {pt.n_trades}")
    print(f"    胜率:          {pt.win_rate:.1%}")
    print(f"    平均持仓:      {pt.avg_holding_days:.0f} 天")
    print()
    print(f"  绩效:")
    print(f"    Sharpe Ratio:  {pt.sharpe:.3f}")
    print(f"    最大回撤:      {pt.max_drawdown:.1%}")
    print(f"{'='*60}\n")
