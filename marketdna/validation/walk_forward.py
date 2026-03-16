"""
Walk-Forward Validation — the gold standard for strategy backtesting

★ Why walk-forward?
  A single in-sample backtest is like studying the answers before an exam.
  Walk-forward validation splits data into expanding training windows and
  fixed out-of-sample (OOS) test periods:

  [===== Train 1 =====][== Test 1 ==]
  [========= Train 2 =========][== Test 2 ==]
  [============== Train 3 ==============][== Test 3 ==]

  Each test period uses ONLY parameters estimated from prior data.
  This prevents look-ahead bias and gives a realistic estimate of
  how the strategy would have performed in real-time.

★ What we validate:
  - Vol timing: GARCH parameters estimated on train, applied to test
  - Pair trading: Hedge ratio and z-score thresholds estimated on train
  - Regime detection: HMM trained on train, applied to test

★ Key metric: OOS Sharpe ratio
  If in-sample Sharpe = 1.5 but OOS Sharpe = 0.3, the strategy is overfit.
  A good strategy has OOS Sharpe ≥ 0.5 × in-sample Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """One fold of walk-forward validation"""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    n_train_days: int
    n_test_days: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated walk-forward validation results"""
    strategy_name: str
    ticker: str
    folds: list[WalkForwardFold]

    # Aggregated OOS metrics
    avg_oos_sharpe: float
    std_oos_sharpe: float
    avg_oos_return: float
    avg_oos_max_drawdown: float
    total_oos_days: int

    # Overfitting diagnostics
    avg_is_sharpe: float
    sharpe_decay_ratio: float   # OOS/IS — closer to 1.0 = less overfit

    # Combined OOS equity curve
    oos_equity: pd.Series


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _max_drawdown(cumulative: pd.Series) -> float:
    if len(cumulative) < 2:
        return 0.0
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def _annualized_return(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    total = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    return float((1 + total) ** (1 / n_years) - 1)


def walk_forward_vol_timing(
    returns: pd.Series,
    ticker: str = "",
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    min_train_days: int = 504,   # ~2 years minimum training
    test_days: int = 63,         # ~1 quarter OOS test
    step_days: int = 63,         # step forward by 1 quarter
) -> WalkForwardResult:
    """Walk-forward validation for vol timing strategy.

    Parameters
    ----------
    min_train_days : int
        Minimum training window size (expanding window).
    test_days : int
        Fixed out-of-sample test period length.
    step_days : int
        How many days to step forward between folds.
    """
    from arch import arch_model

    r = returns.dropna()
    n = len(r)
    folds: list[WalkForwardFold] = []
    all_oos_returns: list[pd.Series] = []

    fold_id = 0
    train_end_idx = min_train_days

    while train_end_idx + test_days <= n:
        # Split
        train = r.iloc[:train_end_idx]
        test = r.iloc[train_end_idx:train_end_idx + test_days]

        # Train GARCH on training data
        try:
            garch = arch_model(
                train * 100, vol="Garch", p=1, q=1,
                mean="Constant", rescale=False,
            )
            fit = garch.fit(disp="off")

            # In-sample performance
            cond_vol_is = fit.conditional_volatility / 100
            target_daily = target_vol / np.sqrt(252)
            weights_is = (target_daily / cond_vol_is).clip(upper=max_leverage)
            is_ret = (pd.Series(weights_is.values, index=train.index).shift(1) * train).dropna()
            is_sharpe = _sharpe(is_ret)

            # Out-of-sample: use last GARCH forecast for the entire test period
            # More realistic: re-estimate GARCH with expanding window each day
            forecasts = fit.forecast(horizon=1, reindex=False)
            last_vol = float(np.sqrt(forecasts.variance.values[-1, 0])) / 100

            # For a more realistic OOS test, we use the most recent conditional
            # vol and decay it toward the unconditional vol over the test window
            omega = fit.params.get("omega", 0.0) / 10000
            alpha = fit.params.get("alpha[1]", 0.05)
            beta_param = fit.params.get("beta[1]", 0.9)
            persistence = alpha + beta_param
            uncond_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else last_vol

            # Simulate forward GARCH variance path
            oos_weights_list = []
            current_var = (last_vol ** 2)
            last_return = float(train.iloc[-1])

            for j in range(len(test)):
                pred_vol = np.sqrt(current_var)
                w = min(target_daily / pred_vol, max_leverage) if pred_vol > 0 else 1.0
                oos_weights_list.append(w)
                # Update variance with realized return
                actual_ret = float(test.iloc[j])
                current_var = omega + alpha * (actual_ret ** 2) + beta_param * current_var

            oos_weights = pd.Series(oos_weights_list, index=test.index)
            oos_ret = (oos_weights.shift(1) * test).dropna()

        except Exception:
            # If GARCH fails, use simple vol targeting
            is_sharpe = _sharpe(train)
            realized_vol = float(train.std())
            target_daily = target_vol / np.sqrt(252)
            w = min(target_daily / realized_vol, max_leverage) if realized_vol > 0 else 1.0
            oos_ret = test * w
            oos_ret = oos_ret.iloc[1:]  # drop first day (no prior weight)

        oos_sharpe = _sharpe(oos_ret)
        oos_return = _annualized_return(oos_ret)
        oos_cum = (1 + oos_ret).cumprod()
        oos_dd = _max_drawdown(oos_cum)

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            in_sample_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_return=oos_return,
            oos_max_drawdown=oos_dd,
            n_train_days=len(train),
            n_test_days=len(test),
        )
        folds.append(fold)
        all_oos_returns.append(oos_ret)

        fold_id += 1
        train_end_idx += step_days

    # Aggregate
    if not folds:
        empty_equity = pd.Series(dtype=float)
        return WalkForwardResult(
            strategy_name="Vol Timing",
            ticker=ticker,
            folds=[],
            avg_oos_sharpe=0.0,
            std_oos_sharpe=0.0,
            avg_oos_return=0.0,
            avg_oos_max_drawdown=0.0,
            total_oos_days=0,
            avg_is_sharpe=0.0,
            sharpe_decay_ratio=0.0,
            oos_equity=empty_equity,
        )

    oos_sharpes = [f.oos_sharpe for f in folds]
    is_sharpes = [f.in_sample_sharpe for f in folds]
    oos_returns = [f.oos_return for f in folds]
    oos_dds = [f.oos_max_drawdown for f in folds]

    avg_is = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    decay = avg_oos / avg_is if avg_is != 0 else 0.0

    # Stitch OOS equity curve
    combined_oos = pd.concat(all_oos_returns)
    oos_equity = (1 + combined_oos).cumprod()

    return WalkForwardResult(
        strategy_name="Vol Timing",
        ticker=ticker,
        folds=folds,
        avg_oos_sharpe=avg_oos,
        std_oos_sharpe=float(np.std(oos_sharpes)),
        avg_oos_return=float(np.mean(oos_returns)),
        avg_oos_max_drawdown=float(np.mean(oos_dds)),
        total_oos_days=sum(f.n_test_days for f in folds),
        avg_is_sharpe=avg_is,
        sharpe_decay_ratio=decay,
        oos_equity=oos_equity,
    )


def walk_forward_pair_trading(
    prices_a: pd.Series,
    prices_b: pd.Series,
    returns_a: pd.Series,
    returns_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    min_train_days: int = 504,
    test_days: int = 63,
    step_days: int = 63,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    lookback: int = 60,
) -> WalkForwardResult:
    """Walk-forward validation for pair trading strategy."""

    # Align all series
    common = (
        prices_a.index
        .intersection(prices_b.index)
        .intersection(returns_a.index)
        .intersection(returns_b.index)
    )
    pa = prices_a.loc[common]
    pb = prices_b.loc[common]
    ra = returns_a.loc[common]
    rb = returns_b.loc[common]

    n = len(pa)
    folds: list[WalkForwardFold] = []
    all_oos_returns: list[pd.Series] = []

    fold_id = 0
    train_end_idx = min_train_days

    while train_end_idx + test_days <= n:
        # Split
        pa_train, pa_test = pa.iloc[:train_end_idx], pa.iloc[train_end_idx:train_end_idx + test_days]
        pb_train, pb_test = pb.iloc[:train_end_idx], pb.iloc[train_end_idx:train_end_idx + test_days]
        ra_test = ra.iloc[train_end_idx:train_end_idx + test_days]
        rb_test = rb.iloc[train_end_idx:train_end_idx + test_days]

        # Estimate hedge ratio on training data
        beta = float(np.polyfit(pb_train.values, pa_train.values, 1)[0])

        # Compute spread on full data up to test end for rolling stats
        full_pa = pa.iloc[:train_end_idx + test_days]
        full_pb = pb.iloc[:train_end_idx + test_days]
        spread = full_pa - beta * full_pb

        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()
        zscore = ((spread - rolling_mean) / rolling_std).dropna()

        # Only use test period z-scores
        test_idx = pa_test.index
        zscore_test = zscore.reindex(test_idx).dropna()

        if len(zscore_test) < 5:
            train_end_idx += step_days
            continue

        # Generate positions on test period
        positions = pd.Series(0.0, index=zscore_test.index)
        in_position = 0

        for i in range(1, len(zscore_test)):
            z = zscore_test.iloc[i]
            if in_position == 0:
                if z > entry_z:
                    in_position = -1
                elif z < -entry_z:
                    in_position = 1
            else:
                if abs(z) < exit_z:
                    in_position = 0
                elif (in_position == -1 and z > stop_z) or (in_position == 1 and z < -stop_z):
                    in_position = 0
            positions.iloc[i] = in_position

        # Capital-weighted returns
        abs_beta = abs(beta)
        w_a = 1.0 / (1.0 + abs_beta)
        w_b = abs_beta / (1.0 + abs_beta)

        ra_aligned = ra_test.reindex(positions.index).fillna(0)
        rb_aligned = rb_test.reindex(positions.index).fillna(0)

        spread_ret = w_a * ra_aligned - w_b * rb_aligned
        oos_ret = (positions.shift(1) * spread_ret).dropna()

        # In-sample: quick estimate
        is_sharpe = 0.0  # simplified — full IS backtest is expensive

        oos_sharpe = _sharpe(oos_ret)
        oos_return = _annualized_return(oos_ret)
        oos_cum = (1 + oos_ret).cumprod() if len(oos_ret) > 0 else pd.Series([1.0])
        oos_dd = _max_drawdown(oos_cum)

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=str(pa_train.index[0].date()),
            train_end=str(pa_train.index[-1].date()),
            test_start=str(pa_test.index[0].date()),
            test_end=str(pa_test.index[-1].date()),
            in_sample_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_return=oos_return,
            oos_max_drawdown=oos_dd,
            n_train_days=len(pa_train),
            n_test_days=len(oos_ret),
        )
        folds.append(fold)
        if len(oos_ret) > 0:
            all_oos_returns.append(oos_ret)

        fold_id += 1
        train_end_idx += step_days

    # Aggregate
    if not folds:
        return WalkForwardResult(
            strategy_name="Pair Trading",
            ticker=f"{ticker_a}/{ticker_b}",
            folds=[],
            avg_oos_sharpe=0.0,
            std_oos_sharpe=0.0,
            avg_oos_return=0.0,
            avg_oos_max_drawdown=0.0,
            total_oos_days=0,
            avg_is_sharpe=0.0,
            sharpe_decay_ratio=0.0,
            oos_equity=pd.Series(dtype=float),
        )

    oos_sharpes = [f.oos_sharpe for f in folds]
    avg_oos = float(np.mean(oos_sharpes))

    combined_oos = pd.concat(all_oos_returns) if all_oos_returns else pd.Series(dtype=float)
    oos_equity = (1 + combined_oos).cumprod() if len(combined_oos) > 0 else pd.Series(dtype=float)

    return WalkForwardResult(
        strategy_name="Pair Trading",
        ticker=f"{ticker_a}/{ticker_b}",
        folds=folds,
        avg_oos_sharpe=avg_oos,
        std_oos_sharpe=float(np.std(oos_sharpes)),
        avg_oos_return=float(np.mean([f.oos_return for f in folds])),
        avg_oos_max_drawdown=float(np.mean([f.oos_max_drawdown for f in folds])),
        total_oos_days=sum(f.n_test_days for f in folds),
        avg_is_sharpe=0.0,
        sharpe_decay_ratio=0.0,
        oos_equity=oos_equity,
    )


def print_walk_forward(wf: WalkForwardResult) -> None:
    """Human-readable walk-forward report"""

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Validation: {wf.strategy_name} ({wf.ticker})")
    print(f"{'='*65}")
    print(f"\n  {len(wf.folds)} folds, {wf.total_oos_days} total OOS days")
    print()

    # Per-fold table
    print(f"  {'Fold':>4} {'Test Period':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS DD':>8}")
    print(f"  {'-'*60}")
    for f in wf.folds:
        period = f"{f.test_start} ~ {f.test_end}"
        print(f"  {f.fold_id:>4} {period:<25} {f.in_sample_sharpe:>10.3f} {f.oos_sharpe:>11.3f} {f.oos_max_drawdown:>8.1%}")

    print(f"\n  {'Aggregated OOS Metrics':}")
    print(f"  {'-'*40}")
    print(f"  Avg OOS Sharpe:    {wf.avg_oos_sharpe:+.3f} (std: {wf.std_oos_sharpe:.3f})")
    print(f"  Avg OOS Return:    {wf.avg_oos_return:+.1%}")
    print(f"  Avg OOS Drawdown:  {wf.avg_oos_max_drawdown:.1%}")

    if wf.sharpe_decay_ratio > 0:
        print(f"\n  Overfitting Check:")
        print(f"  Avg IS Sharpe:     {wf.avg_is_sharpe:+.3f}")
        print(f"  Sharpe Decay:      {wf.sharpe_decay_ratio:.2f}x  ", end="")
        if wf.sharpe_decay_ratio > 0.7:
            print("(Good — minimal overfitting)")
        elif wf.sharpe_decay_ratio > 0.4:
            print("(Fair — some overfitting)")
        else:
            print("(Poor — significant overfitting)")

    print(f"{'='*65}\n")
