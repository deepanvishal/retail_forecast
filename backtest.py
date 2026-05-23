import numpy as np
import pandas as pd

from data import SERIES, S, pred_array, actual_array
from covariance import relative_residuals, schafer_strimmer, build_W_day
from reconcile import mint_closed


def _raw_covariance_ss(past, series):
    n = len(past)
    p = len(series)
    X_raw = np.zeros((n, p))
    for i, s in enumerate(series):
        X_raw[:, i] = past[s + '_actual'].values - past[s + '_pred'].values
    return schafer_strimmer(X_raw)


def _wls_raw_W(past, series):
    p = len(series)
    variances = np.zeros(p)
    for i, s in enumerate(series):
        r = past[s + '_actual'].values - past[s + '_pred'].values
        variances[i] = np.var(r, ddof=1)
    return np.diag(variances)


def _wls_scaled_W(C_rel, pred_vec):
    diag_vals = np.diag(C_rel) * pred_vec ** 2
    return np.diag(diag_vals)


def _mape_row(actual, ytilde):
    return np.abs(actual - ytilde) / np.abs(actual)


def run_backtest(train, series=None, summing_matrix=None, min_window=365):
    """
    Rolling-origin expanding-window backtest.

    Methods compared:
      base       - no reconciliation (incoherent base preds)
      ols        - MinT with W = I (OLS projection onto coherent subspace)
      wls_raw    - MinT with W = diag(sample raw variances, past only)
      wls_scaled - MinT with W = diag(C_rel[i,i] * pred_i^2, past only)
      mint_raw   - MinT with W = SS-shrunk raw covariance (past only)
      mint_scaled- MinT with W_day = diag(pred) @ C_rel @ diag(pred) (past only)

    W is estimated from PAST-ONLY data at each origin — no lookahead.
    The backtest validates reconciliation mechanics and method ranking; it does
    NOT validate long-horizon error structure (see memo for horizon-mismatch caveat).

    Returns:
      results_df    - per-day per-method APE vectors (one row per backtest day)
      summary_df    - mean MAPE per method (7-series average + per-series)
      meta          - dict with lambda history, neg_leaf_count, n_iter
    """
    if series is None:
        series = SERIES
    if summing_matrix is None:
        summing_matrix = S

    p = len(series)
    n_train = len(train)
    methods = ['base', 'ols', 'wls_raw', 'wls_scaled', 'mint_raw', 'mint_scaled']

    records = []
    lambda_history = []
    neg_leaf_count = 0
    W_eye = np.eye(p)

    for t in range(min_window, n_train):
        past = train.iloc[:t]
        today = train.iloc[t]

        pred_vec = np.array([today[s + '_pred'] for s in series])
        actual_vec = np.array([today[s + '_actual'] for s in series])

        # Relative covariance from past
        X_rel = relative_residuals(past, series)
        C_rel, lam = schafer_strimmer(X_rel)
        lambda_history.append(lam)

        # Per-day W for scaled methods
        W_day = build_W_day(C_rel, pred_vec)
        W_raw, _ = _raw_covariance_ss(past, series)
        W_wls_raw = _wls_raw_W(past, series)
        W_wls_sc = _wls_scaled_W(C_rel, pred_vec)

        ytildes = {
            'base': pred_vec,
            'ols': mint_closed(W_eye, summing_matrix, pred_vec),
            'wls_raw': mint_closed(W_wls_raw, summing_matrix, pred_vec),
            'wls_scaled': mint_closed(W_wls_sc, summing_matrix, pred_vec),
            'mint_raw': mint_closed(W_raw, summing_matrix, pred_vec),
            'mint_scaled': mint_closed(W_day, summing_matrix, pred_vec),
        }

        # Check for any negative leaves in the unconstrained mint_scaled
        if np.any(ytildes['mint_scaled'][3:] < 0):
            neg_leaf_count += 1

        row = {'date': today['date']}
        for method in methods:
            ape = _mape_row(actual_vec, ytildes[method])
            for i, s in enumerate(series):
                row[f'{method}_{s}_ape'] = ape[i]
            row[f'{method}_mean_ape'] = ape.mean()
        records.append(row)

    results_df = pd.DataFrame(records)

    # Summary: mean APE per method (across days), overall and per series
    summary = {}
    for method in methods:
        summary[method] = {
            'overall_mape': results_df[f'{method}_mean_ape'].mean() * 100,
        }
        for s in series:
            summary[method][f'mape_{s}'] = results_df[f'{method}_{s}_ape'].mean() * 100

    summary_rows = []
    for method, vals in summary.items():
        row = {'method': method}
        row.update(vals)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index('method')

    meta = {
        'n_iter': len(records),
        'neg_leaf_count': neg_leaf_count,
        'lambda_mean': float(np.mean(lambda_history)),
        'lambda_min': float(np.min(lambda_history)),
        'lambda_max': float(np.max(lambda_history)),
    }
    return results_df, summary_df, meta


def insample_vs_oos_W(train, series=None, summing_matrix=None, min_window=365):
    """
    Compare W estimated from full train (in-sample / peeking) vs past-only (OOS)
    at each rolling origin. Reports the mean MAPE gap.

    In-sample W: estimated from all 1112 train rows (uses data after the eval day).
    OOS W: estimated from past.iloc[:t] at each origin (no lookahead).

    The gap quantifies covariance-window optimism — one component of in-sample bias.
    The deeper issue (residuals themselves being in-sample or short-horizon vs the
    true long-horizon error structure of the future block) cannot be measured here
    because the base models cannot be refit. See memo for discussion.
    """
    if series is None:
        series = SERIES
    if summing_matrix is None:
        summing_matrix = S

    n_train = len(train)

    # Pre-compute full-data C_insample once
    X_rel_full = relative_residuals(train, series)
    C_insample, _ = schafer_strimmer(X_rel_full)

    oos_apes = []
    ins_apes = []

    for t in range(min_window, n_train):
        past = train.iloc[:t]
        today = train.iloc[t]

        pred_vec = np.array([today[s + '_pred'] for s in series])
        actual_vec = np.array([today[s + '_actual'] for s in series])

        X_rel = relative_residuals(past, series)
        C_oos, _ = schafer_strimmer(X_rel)

        W_oos = build_W_day(C_oos, pred_vec)
        W_ins = build_W_day(C_insample, pred_vec)

        ytilde_oos = mint_closed(W_oos, summing_matrix, pred_vec)
        ytilde_ins = mint_closed(W_ins, summing_matrix, pred_vec)

        oos_apes.append(_mape_row(actual_vec, ytilde_oos).mean())
        ins_apes.append(_mape_row(actual_vec, ytilde_ins).mean())

    mean_oos = float(np.mean(oos_apes)) * 100
    mean_ins = float(np.mean(ins_apes)) * 100
    return {
        'mape_oos_pct': mean_oos,
        'mape_insample_pct': mean_ins,
        'optimism_gap_pct': mean_oos - mean_ins,
    }
