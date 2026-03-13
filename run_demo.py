#!/usr/bin/env python3
"""
MarketDNA 完整演示脚本

运行：
    cd quantresearch
    source .venv/bin/activate
    python run_demo.py

这个脚本展示了 MarketDNA 的全部功能：
  1. 单只股票 DNA 指纹（分布 + 波动率）
  2. Regime 检测（HMM）
  3. 波动率择时信号（GARCH）
  4. 配对分析（相关性 + 协整）
  5. 配对交易信号（均值回复）
  6. RMT 去噪（多只股票相关矩阵）
  7. 可视化 Dashboard
"""
from __future__ import annotations

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")  # 无头模式，保存到文件

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_single_stock():
    """Demo 1: 单只股票的完整 DNA 分析"""
    print("\n" + "=" * 70)
    print("  DEMO 1: SPY (S&P 500 ETF) DNA 指纹")
    print("=" * 70)

    from marketdna.data.fetcher import fetch
    from marketdna.analysis.distribution import analyze_distribution, print_fingerprint
    from marketdna.analysis.volatility import analyze_volatility, print_volatility
    from marketdna.viz.plots import plot_full_dashboard, plot_qq, plot_distribution

    data = fetch("SPY", start="2015-01-01")
    print(f"\n  获取数据: {data.n_days} 天 ({data.n_years:.1f} 年)")

    dist = analyze_distribution(data.log_returns, "SPY")
    print_fingerprint(dist)

    vol = analyze_volatility(data.log_returns, "SPY")
    print_volatility(vol)

    # 可视化
    plot_full_dashboard(data.log_returns, "SPY",
                       save_path=os.path.join(OUTPUT_DIR, "spy_dashboard.png"))
    plot_qq(data.log_returns, "SPY",
           save_path=os.path.join(OUTPUT_DIR, "spy_qq.png"))
    plot_distribution(data.log_returns, "SPY", t_df=dist.t_df,
                     save_path=os.path.join(OUTPUT_DIR, "spy_distribution.png"))

    print(f"\n  图表已保存到 {OUTPUT_DIR}/")
    return data


def demo_regime(data):
    """Demo 2: Regime 检测"""
    print("\n" + "=" * 70)
    print("  DEMO 2: SPY Regime 检测 (HMM)")
    print("=" * 70)

    from marketdna.analysis.regime import analyze_regime, print_regime
    from marketdna.viz.plots import plot_regime_timeline

    regime = analyze_regime(data.log_returns, "SPY", n_regimes=2)
    print_regime(regime)

    plot_regime_timeline(
        data.log_returns, regime.regime_labels,
        regime.regime_names, "SPY",
        save_path=os.path.join(OUTPUT_DIR, "spy_regime.png"),
    )
    print(f"  Regime 时间线已保存")
    return regime


def demo_vol_timing(data):
    """Demo 3: 波动率择时"""
    print("\n" + "=" * 70)
    print("  DEMO 3: SPY 波动率择时策略 (GARCH)")
    print("=" * 70)

    from marketdna.signals.vol_timing import generate_vol_timing, print_vol_timing
    from marketdna.viz.plots import plot_vol_timing

    vt = generate_vol_timing(data.log_returns, "SPY", target_vol=0.10)
    print_vol_timing(vt)

    plot_vol_timing(
        data.log_returns, vt.weights, vt.strategy_returns, "SPY",
        save_path=os.path.join(OUTPUT_DIR, "spy_vol_timing.png"),
    )
    print(f"  Vol Timing 图已保存")
    return vt


def demo_pair_analysis():
    """Demo 4: 配对分析"""
    print("\n" + "=" * 70)
    print("  DEMO 4: GLD vs GDX 配对分析")
    print("=" * 70)

    from marketdna.data.fetcher import fetch
    from marketdna.analysis.correlation import analyze_pair, print_pair

    data_gld = fetch("GLD", start="2015-01-01")
    data_gdx = fetch("GDX", start="2015-01-01")

    close_col = "Adj Close" if "Adj Close" in data_gld.prices.columns else "Close"

    pair = analyze_pair(
        data_gld.log_returns, data_gdx.log_returns,
        data_gld.prices[close_col].squeeze(),
        data_gdx.prices[close_col].squeeze(),
        "GLD", "GDX",
    )
    print_pair(pair)
    return data_gld, data_gdx, pair


def demo_pair_trading(data_gld, data_gdx):
    """Demo 5: 配对交易信号"""
    print("\n" + "=" * 70)
    print("  DEMO 5: GLD/GDX 配对交易策略")
    print("=" * 70)

    from marketdna.signals.mean_reversion import generate_pair_signal, print_pair_trading
    from marketdna.viz.plots import plot_spread

    close_col = "Adj Close" if "Adj Close" in data_gld.prices.columns else "Close"
    pa = data_gld.prices[close_col].squeeze()
    pb = data_gdx.prices[close_col].squeeze()

    pt = generate_pair_signal(
        pa, pb,
        data_gld.log_returns, data_gdx.log_returns,
        "GLD", "GDX",
    )
    print_pair_trading(pt)

    plot_spread(
        pt.spread, pt.zscore, pt.positions,
        "GLD", "GDX",
        save_path=os.path.join(OUTPUT_DIR, "gld_gdx_spread.png"),
    )
    print(f"  配对交易图已保存")
    return pt


def demo_rmt():
    """Demo 6: RMT 去噪"""
    print("\n" + "=" * 70)
    print("  DEMO 6: 科技股相关矩阵 RMT 去噪")
    print("=" * 70)

    from marketdna.data.fetcher import fetch
    from marketdna.analysis.rmt import analyze_rmt, print_rmt
    from marketdna.viz.plots import plot_rmt_eigenvalues, plot_correlation_heatmap

    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM"]
    print(f"\n  获取 {len(tickers)} 只科技股数据...")

    returns_dict = {}
    for t in tickers:
        try:
            d = fetch(t, start="2020-01-01")
            returns_dict[t] = d.log_returns
        except Exception as e:
            print(f"  跳过 {t}: {e}")

    if len(returns_dict) < 5:
        print("  数据不足，跳过 RMT demo")
        return

    # 构建 returns DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()
    print(f"  共 {returns_df.shape[1]} 只股票, {returns_df.shape[0]} 天数据")

    fp, raw_corr, denoised_corr = analyze_rmt(returns_df)
    print_rmt(fp)

    # 特征值分布图
    eigenvalues = np.linalg.eigvalsh(raw_corr)[::-1]
    plot_rmt_eigenvalues(
        eigenvalues, fp.mp_lambda_max, fp.mp_lambda_min,
        fp.n_assets, fp.n_observations,
        save_path=os.path.join(OUTPUT_DIR, "rmt_eigenvalues.png"),
    )

    # 去噪前后对比
    labels = list(returns_dict.keys())
    plot_correlation_heatmap(
        raw_corr, labels, "Raw Correlation Matrix",
        save_path=os.path.join(OUTPUT_DIR, "corr_raw.png"),
    )
    plot_correlation_heatmap(
        denoised_corr, labels, "RMT Denoised Correlation Matrix",
        save_path=os.path.join(OUTPUT_DIR, "corr_denoised.png"),
    )
    print(f"  RMT 图已保存")


def main():
    print("\n" + "#" * 70)
    print("  MarketDNA — 金融时间序列统计指纹提取器")
    print("  完整功能演示")
    print("#" * 70)

    # Demo 1: 单只股票
    data = demo_single_stock()

    # Demo 2: Regime 检测
    demo_regime(data)

    # Demo 3: 波动率择时
    demo_vol_timing(data)

    # Demo 4: 配对分析
    data_gld, data_gdx, pair = demo_pair_analysis()

    # Demo 5: 配对交易
    demo_pair_trading(data_gld, data_gdx)

    # Demo 6: RMT 去噪
    demo_rmt()

    print("\n" + "#" * 70)
    print("  所有 Demo 完成！")
    print(f"  图表输出目录: {OUTPUT_DIR}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
