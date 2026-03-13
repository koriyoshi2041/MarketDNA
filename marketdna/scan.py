"""
MarketDNA Scanner — 一键生成股票的统计指纹报告

用法:
    from marketdna.scan import scan
    report = scan("AAPL")           # 单只股票
    report = scan("SPY", "QQQ")     # 多只 + 配对分析
"""
from __future__ import annotations

from dataclasses import dataclass

from marketdna.analysis.correlation import (
    PairFingerprint,
    analyze_pair,
    print_pair,
)
from marketdna.analysis.distribution import (
    DistributionFingerprint,
    analyze_distribution,
    print_fingerprint,
)
from marketdna.analysis.volatility import (
    VolatilityFingerprint,
    analyze_volatility,
    print_volatility,
)
from marketdna.data.fetcher import MarketData, fetch
import pandas as pd


@dataclass(frozen=True)
class DNAReport:
    """一只股票的完整DNA报告"""

    data: MarketData
    distribution: DistributionFingerprint
    volatility: VolatilityFingerprint


@dataclass(frozen=True)
class PairReport:
    """两只股票的配对分析报告"""

    report_a: DNAReport
    report_b: DNAReport
    pair: PairFingerprint


def _get_adj_close(data: MarketData) -> pd.Series:
    """从 MarketData 中提取复权价格"""
    close_col = "Adj Close" if "Adj Close" in data.prices.columns else "Close"
    return data.prices[close_col].squeeze()


def scan_single(ticker: str, start: str = "2010-01-01") -> DNAReport:
    """扫描单只股票的DNA"""
    data = fetch(ticker, start=start)

    dist_fp = analyze_distribution(data.log_returns, ticker)
    vol_fp = analyze_volatility(data.log_returns, ticker)

    return DNAReport(data=data, distribution=dist_fp, volatility=vol_fp)


def scan(*tickers: str, start: str = "2010-01-01") -> DNAReport | list[DNAReport] | PairReport:
    """主入口：扫描一只或多只股票

    Examples
    --------
    >>> report = scan("AAPL")                    # 单只
    >>> reports = scan("AAPL", "MSFT", "GOOG")   # 多只
    >>> pair = scan("GLD", "GDX")                # 两只 = 自动做配对分析
    """
    if len(tickers) == 0:
        raise ValueError("At least one ticker required")

    reports = [scan_single(t, start) for t in tickers]

    # 单只：直接返回报告
    if len(reports) == 1:
        r = reports[0]
        print_fingerprint(r.distribution)
        print_volatility(r.volatility)
        return r

    # 多只：打印每只的报告
    for r in reports:
        print_fingerprint(r.distribution)
        print_volatility(r.volatility)

    # 两只：额外做配对分析
    if len(reports) == 2:
        ra, rb = reports[0], reports[1]
        adj_a = _get_adj_close(ra.data)
        adj_b = _get_adj_close(rb.data)

        pair_fp = analyze_pair(
            ra.data.log_returns, rb.data.log_returns,
            adj_a, adj_b,
            ra.data.ticker, rb.data.ticker,
        )
        print_pair(pair_fp)
        return PairReport(report_a=ra, report_b=rb, pair=pair_fp)

    return reports


def scan_deep(
    ticker: str,
    start: str = "2010-01-01",
    n_regimes: int = 2,
    target_vol: float = 0.10,
) -> dict:
    """深度扫描：基础DNA + Regime检测 + Vol Timing信号

    Returns
    -------
    dict with keys: 'report', 'regime', 'vol_timing'
    """
    from marketdna.analysis.regime import analyze_regime, print_regime
    from marketdna.signals.vol_timing import generate_vol_timing, print_vol_timing

    report = scan_single(ticker, start)

    print_fingerprint(report.distribution)
    print_volatility(report.volatility)

    regime = analyze_regime(report.data.log_returns, ticker, n_regimes=n_regimes)
    print_regime(regime)

    vol_timing = generate_vol_timing(report.data.log_returns, ticker, target_vol=target_vol)
    print_vol_timing(vol_timing)

    return {
        "report": report,
        "regime": regime,
        "vol_timing": vol_timing,
    }


def scan_pair_deep(
    ticker_a: str,
    ticker_b: str,
    start: str = "2010-01-01",
    n_regimes: int = 2,
) -> dict:
    """深度配对扫描：基础DNA + 配对分析 + 配对交易信号 + Regime

    Returns
    -------
    dict with keys: 'pair_report', 'regime_a', 'regime_b', 'pair_signal'
    """
    from marketdna.analysis.regime import analyze_regime, print_regime
    from marketdna.signals.mean_reversion import generate_pair_signal, print_pair_trading

    pair_report = scan(ticker_a, ticker_b, start=start)

    if not isinstance(pair_report, PairReport):
        raise ValueError("Expected exactly 2 tickers for pair scan")

    # Regime 检测
    regime_a = analyze_regime(pair_report.report_a.data.log_returns, ticker_a, n_regimes)
    print_regime(regime_a)

    # 配对交易信号
    adj_a = _get_adj_close(pair_report.report_a.data)
    adj_b = _get_adj_close(pair_report.report_b.data)

    pair_signal = generate_pair_signal(
        adj_a, adj_b,
        pair_report.report_a.data.log_returns,
        pair_report.report_b.data.log_returns,
        ticker_a, ticker_b,
    )
    print_pair_trading(pair_signal)

    return {
        "pair_report": pair_report,
        "regime_a": regime_a,
        "pair_signal": pair_signal,
    }
