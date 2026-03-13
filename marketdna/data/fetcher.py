"""
数据获取层 — 统一接口拉取市场数据并计算基础收益率序列

★ 量化核心概念：
  - log return vs simple return：log return可加性好（多期收益 = 单期之和），
    统计分析中更常用；simple return则是"你实际赚了多少"
  - 复权价格（Adjusted Close）：考虑了股票分拆和股息再投资后的"真实价格"
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketData:
    """不可变的市场数据容器"""

    ticker: str
    prices: pd.DataFrame        # OHLCV + Adj Close
    log_returns: pd.Series      # ln(P_t / P_{t-1})
    simple_returns: pd.Series   # (P_t - P_{t-1}) / P_{t-1}
    start: date
    end: date

    @property
    def n_days(self) -> int:
        return len(self.log_returns.dropna())

    @property
    def n_years(self) -> float:
        return self.n_days / 252  # 252 trading days per year


def fetch(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> MarketData:
    """从Yahoo Finance获取数据并计算收益率序列

    Parameters
    ----------
    ticker : str
        股票代码，如 "SPY", "AAPL", "000300.SS"(沪深300)
    start : str
        起始日期
    end : str | None
        结束日期，默认为今天
    """
    end = end or str(date.today())

    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(f"No data found for {ticker}")

    # yfinance 0.2.40+ 返回 MultiIndex columns: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # 新版 yfinance 没有 "Adj Close"，"Close" 已经是复权价
    close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    adj_close = raw[close_col].squeeze()

    log_ret = np.log(adj_close / adj_close.shift(1)).dropna()
    log_ret.name = f"{ticker}_log_return"

    simple_ret = adj_close.pct_change().dropna()
    simple_ret.name = f"{ticker}_simple_return"

    return MarketData(
        ticker=ticker,
        prices=raw,
        log_returns=log_ret,
        simple_returns=simple_ret,
        start=pd.Timestamp(start).date(),
        end=pd.Timestamp(end).date(),
    )


def fetch_multi(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> dict[str, MarketData]:
    """批量获取多只股票数据"""
    return {t: fetch(t, start, end) for t in tickers}
