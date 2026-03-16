"""
Cointegration Validator — screen pairs before deploying pair trading

★ Why validate cointegration?
  The Engle-Granger test gives you a p-value at one point in time.
  But cointegration can BREAK — what was cointegrated in 2018 might
  not be in 2023 (structural breaks, regime changes, fundamentals shift).

  This validator:
  1. Rolling cointegration test: is the pair still cointegrated?
  2. Stability check: has it been cointegrated for most of the lookback?
  3. Spread stationarity: ADF test on the spread itself
  4. Half-life check: is mean reversion fast enough to be tradeable?

  Only pairs passing ALL checks should be traded.

★ Common failure modes:
  - GLD/GDX: high correlation (0.77) but NOT cointegrated → fake pair
  - TLT/IEF: both Treasury ETFs, structurally linked → good candidate
  - XLF/JPM: sector ETF vs individual stock → cointegration breaks often
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller


@dataclass(frozen=True)
class CointegrationReport:
    """Comprehensive cointegration validation report"""

    ticker_a: str
    ticker_b: str

    # Current state
    is_cointegrated: bool
    coint_pvalue: float
    hedge_ratio: float

    # Rolling stability
    rolling_coint_ratio: float    # % of rolling windows where cointegrated
    rolling_pvalues: pd.Series    # p-value over time

    # Spread stationarity
    spread_adf_pvalue: float
    spread_is_stationary: bool

    # Mean reversion speed
    half_life: float
    is_tradeable_speed: bool      # half_life < 30 days

    # Verdict
    confidence: str               # "HIGH", "MEDIUM", "LOW", "REJECT"
    rejection_reasons: list[str]

    @property
    def is_valid(self) -> bool:
        return self.confidence in ("HIGH", "MEDIUM")


def validate_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    rolling_window: int = 252,    # 1-year rolling window
    min_coint_ratio: float = 0.6, # Must be cointegrated in 60%+ of windows
    max_half_life: float = 30.0,  # Max acceptable half-life in days
    significance: float = 0.05,
) -> CointegrationReport:
    """Validate cointegration relationship for pair trading.

    Parameters
    ----------
    rolling_window : int
        Size of rolling window for stability check (days).
    min_coint_ratio : float
        Minimum fraction of rolling windows showing cointegration.
    max_half_life : float
        Maximum half-life for the spread to be considered tradeable.
    significance : float
        Significance level for cointegration tests.
    """
    # Align
    common = prices_a.index.intersection(prices_b.index)
    pa = prices_a.loc[common].dropna()
    pb = prices_b.loc[common].dropna()
    common = pa.index.intersection(pb.index)
    pa = pa.loc[common]
    pb = pb.loc[common]

    rejection_reasons: list[str] = []

    # --- 1. Current cointegration test ---
    try:
        coint_stat, coint_pval, _ = coint(pa, pb)
        is_coint = coint_pval < significance
    except Exception:
        coint_pval = 1.0
        is_coint = False

    if not is_coint:
        rejection_reasons.append(f"Not cointegrated (p={coint_pval:.4f})")

    # --- 2. Hedge ratio ---
    beta = float(np.polyfit(pb.values, pa.values, 1)[0])

    # --- 3. Spread and its stationarity ---
    spread = pa - beta * pb

    try:
        adf_stat, adf_pval, *_ = adfuller(spread.dropna(), maxlag=20)
        spread_stationary = adf_pval < significance
    except Exception:
        adf_pval = 1.0
        spread_stationary = False

    if not spread_stationary:
        rejection_reasons.append(f"Spread not stationary (ADF p={adf_pval:.4f})")

    # --- 4. Half-life ---
    spread_clean = spread.dropna()
    if len(spread_clean) > 10:
        spread_lag = spread_clean.shift(1).dropna()
        spread_now = spread_clean.iloc[1:]
        common_hl = spread_lag.index.intersection(spread_now.index)
        if len(common_hl) > 10:
            slope, _, _, _, _ = stats.linregress(
                spread_lag.loc[common_hl], spread_now.loc[common_hl]
            )
            if 0 < slope < 1:
                half_life = -np.log(2) / np.log(slope)
            else:
                half_life = float("inf")
        else:
            half_life = float("inf")
    else:
        half_life = float("inf")

    tradeable_speed = half_life < max_half_life
    if not tradeable_speed:
        hl_str = f"{half_life:.0f}" if half_life < 1000 else "inf"
        rejection_reasons.append(f"Half-life too slow ({hl_str} days > {max_half_life:.0f})")

    # --- 5. Rolling cointegration stability ---
    n = len(pa)
    rolling_pvals = []
    rolling_dates = []

    step = max(rolling_window // 4, 20)  # check every ~quarter
    for end_idx in range(rolling_window, n, step):
        start_idx = end_idx - rolling_window
        pa_win = pa.iloc[start_idx:end_idx]
        pb_win = pb.iloc[start_idx:end_idx]
        try:
            _, pval, _ = coint(pa_win, pb_win)
            rolling_pvals.append(pval)
            rolling_dates.append(pa.index[end_idx - 1])
        except Exception:
            rolling_pvals.append(1.0)
            rolling_dates.append(pa.index[end_idx - 1])

    rolling_pvals_series = pd.Series(rolling_pvals, index=rolling_dates, name="coint_pval")
    coint_ratio = float(np.mean([p < significance for p in rolling_pvals])) if rolling_pvals else 0.0

    if coint_ratio < min_coint_ratio:
        rejection_reasons.append(
            f"Unstable cointegration ({coint_ratio:.0%} < {min_coint_ratio:.0%} of windows)"
        )

    # --- Confidence level ---
    if not rejection_reasons:
        if coint_ratio > 0.8 and half_life < 15:
            confidence = "HIGH"
        elif coint_ratio > 0.6 and half_life < 30:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    elif len(rejection_reasons) == 1 and is_coint:
        confidence = "LOW"
    else:
        confidence = "REJECT"

    return CointegrationReport(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        is_cointegrated=is_coint,
        coint_pvalue=float(coint_pval),
        hedge_ratio=beta,
        rolling_coint_ratio=coint_ratio,
        rolling_pvalues=rolling_pvals_series,
        spread_adf_pvalue=float(adf_pval),
        spread_is_stationary=spread_stationary,
        half_life=half_life,
        is_tradeable_speed=tradeable_speed,
        confidence=confidence,
        rejection_reasons=rejection_reasons,
    )


def print_cointegration_report(cr: CointegrationReport) -> None:
    """Human-readable cointegration validation report"""

    status = "PASS" if cr.is_valid else "FAIL"
    print(f"\n{'='*60}")
    print(f"  Cointegration Validator: {cr.ticker_a}/{cr.ticker_b}  [{status}]")
    print(f"{'='*60}")
    print()
    print(f"  Engle-Granger Test:")
    print(f"    p-value:        {cr.coint_pvalue:.4f}  {'PASS' if cr.is_cointegrated else 'FAIL'}")
    print(f"    Hedge ratio:    {cr.hedge_ratio:.4f}")
    print()
    print(f"  Spread Stationarity (ADF):")
    print(f"    p-value:        {cr.spread_adf_pvalue:.4f}  {'PASS' if cr.spread_is_stationary else 'FAIL'}")
    print()
    hl_str = f"{cr.half_life:.1f} days" if cr.half_life < 1000 else "inf"
    print(f"  Mean Reversion Speed:")
    print(f"    Half-life:      {hl_str}  {'PASS' if cr.is_tradeable_speed else 'FAIL'}")
    print()
    print(f"  Rolling Stability ({len(cr.rolling_pvalues)} windows):")
    print(f"    Cointegrated:   {cr.rolling_coint_ratio:.0%} of windows  ", end="")
    print("PASS" if cr.rolling_coint_ratio >= 0.6 else "FAIL")
    print()
    print(f"  Confidence:       {cr.confidence}")

    if cr.rejection_reasons:
        print(f"\n  Rejection Reasons:")
        for reason in cr.rejection_reasons:
            print(f"    - {reason}")

    print(f"{'='*60}\n")
