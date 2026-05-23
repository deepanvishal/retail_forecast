import numpy as np
import pandas as pd

from data import SERIES, S, pred_array, actual_array, METHOD_LABELS
from covariance import relative_residuals, schafer_strimmer, build_W_day
from reconcile import mint_closed


def _raw_covariance_ss(past, series):
    """
    Estimate the SS-shrunk covariance matrix from raw (absolute) forecast errors.

    Used for the mint_raw and wls_raw backtest methods, which weight by absolute
    rather than relative error variance.

    Args:
        past (pd.DataFrame): subset of training data before the eval origin.
        series (list[str]): series keys.
    Returns:
        tuple: (C_raw, lam) — SS-shrunk absolute covariance, shrinkage intensity.
    """
    n = len(past)
    p = len(series)
    X_raw = np.zeros((n, p))
    for i, s in enumerate(series):
        X_raw[:, i] = past[s + '_actual'].values - past[s + '_pred'].values
    return schafer_strimmer(X_raw)


def _wls_raw_W(past, series):
    """
    Build a diagonal W from the sample variance of raw (absolute) forecast errors.

    One variance per series, estimated from past-only data at each rolling origin.

    Args:
        past (pd.DataFrame): subset of training data before the eval origin.
        series (list[str]): series keys.
    Returns:
        np.ndarray, shape (p, p): diagonal covariance matrix.
    """
    p = len(series)
    variances = np.zeros(p)
    for i, s in enumerate(series):
        r = past[s + '_actual'].values - past[s + '_pred'].values
        variances[i] = np.var(r, ddof=1)
    return np.diag(variances)


def _wls_scaled_W(C_rel, pred_vec):
    """
    Build a diagonal W from per-day scaled relative variances.

    W[i,i] = C_rel[i,i] * pred_i^2 — diagonal version of the per-day W_day,
    without the off-diagonal cross-series terms.

    Args:
        C_rel (np.ndarray, shape (p, p)): relative covariance matrix.
        pred_vec (np.ndarray, shape (p,)): base forecasts for one day.
    Returns:
        np.ndarray, shape (p, p): diagonal covariance matrix.
    """
    diag_vals = np.diag(C_rel) * pred_vec ** 2
    return np.diag(diag_vals)


def _mape_row(actual, ytilde):
    """
    Compute the per-series absolute percentage error for one day.

    Args:
        actual (np.ndarray, shape (p,)): actual values.
        ytilde (np.ndarray, shape (p,)): reconciled forecast.
    Returns:
        np.ndarray, shape (p,): |actual − ytilde| / |actual| for each series.
    """
    return np.abs(actual - ytilde) / np.abs(actual)


def run_backtest(train, series=None, summing_matrix=None, min_window=365):
    """
    Rolling-origin expanding-window backtest comparing six reconciliation methods.

    Methods evaluated:
      base        — no reconciliation (incoherent base forecasts, included as baseline)
      ols         — MinT with W = I (equal-weight OLS projection onto coherent subspace)
      wls_raw     — MinT with W = diag(past raw error variances)
      wls_scaled  — MinT with W = diag(C_rel[i,i] * pred_i^2) from past data
      mint_raw    — MinT with W = SS-shrunk raw covariance from past data
      mint_scaled — MinT with W_day = diag(pred) @ C_rel @ diag(pred), past data only
                    (this is the shipped method for the future block)

    W is estimated from PAST-ONLY data at each origin — no lookahead.
    The backtest validates reconciliation mechanics and method ranking on the training
    data distribution; it does NOT validate the long-horizon error structure of the
    351-day future block (see memo §8 for the horizon-mismatch caveat).

    Args:
        train (pd.DataFrame): full training set (1112 rows).
        series (list[str], optional): series keys. Defaults to SERIES.
        summing_matrix (np.ndarray, optional): summing matrix S. Defaults to S.
        min_window (int): minimum number of past rows before the first eval origin.
            Default: 365 (ensures at least one year of data for W estimation).
    Returns:
        results_df (pd.DataFrame): per-day, per-method APE vectors (747 rows).
        summary_df (pd.DataFrame): mean MAPE % per method — overall and per series.
        meta (dict): {
            'n_iter': int — number of rolling origins evaluated;
            'neg_leaf_count': int — origins where unconstrained mint_scaled had a
                negative leaf (indicates QP would bind; 0/747 in practice);
            'lambda_mean', 'lambda_min', 'lambda_max': float — SS λ distribution
                across origins, showing λ stability.
        }
    """
    if series is None:
        series = SERIES
    if summing_matrix is None:
        summing_matrix = S

    p       = len(series)
    n_train = len(train)
    methods = ['base', 'ols', 'wls_raw', 'wls_scaled', 'mint_raw', 'mint_scaled']

    records       = []
    lambda_history = []
    neg_leaf_count = 0
    W_eye = np.eye(p)

    for t in range(min_window, n_train):
        past  = train.iloc[:t]
        today = train.iloc[t]

        pred_vec   = np.array([today[s + '_pred']   for s in series])
        actual_vec = np.array([today[s + '_actual'] for s in series])

        X_rel  = relative_residuals(past, series)
        C_rel, lam = schafer_strimmer(X_rel)
        lambda_history.append(lam)

        W_day    = build_W_day(C_rel, pred_vec)
        W_raw, _ = _raw_covariance_ss(past, series)
        W_wls_raw = _wls_raw_W(past, series)
        W_wls_sc  = _wls_scaled_W(C_rel, pred_vec)

        ytildes = {
            'base':        pred_vec,
            'ols':         mint_closed(W_eye,     summing_matrix, pred_vec),
            'wls_raw':     mint_closed(W_wls_raw, summing_matrix, pred_vec),
            'wls_scaled':  mint_closed(W_wls_sc,  summing_matrix, pred_vec),
            'mint_raw':    mint_closed(W_raw,      summing_matrix, pred_vec),
            'mint_scaled': mint_closed(W_day,      summing_matrix, pred_vec),
        }

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

    summary = {}
    for method in methods:
        summary[method] = {
            'overall_mape': results_df[f'{method}_mean_ape'].mean() * 100,
        }
        for s in series:
            summary[method][f'mape_{s}'] = results_df[f'{method}_{s}_ape'].mean() * 100

    summary_rows = [{'method': m, **v} for m, v in summary.items()]
    summary_df = pd.DataFrame(summary_rows).set_index('method')

    meta = {
        'n_iter':        len(records),
        'neg_leaf_count': neg_leaf_count,
        'lambda_mean':   float(np.mean(lambda_history)),
        'lambda_min':    float(np.min(lambda_history)),
        'lambda_max':    float(np.max(lambda_history)),
    }
    return results_df, summary_df, meta


def insample_vs_oos_W(train, series=None, summing_matrix=None, min_window=365):
    """
    Measure the covariance-window component of in-sample optimism.

    Compares mint_scaled accuracy when W is estimated from: (a) past-only data at
    each rolling origin (out-of-sample, no lookahead), vs (b) the full training set
    (in-sample, peeks at data after the eval day).

    The gap quantifies one specific source of optimism: using more data to estimate C.
    The deeper issue — whether the train-period residuals themselves represent genuine
    long-horizon (351-day) forecast errors or shorter-horizon fits — cannot be measured
    here because the base models cannot be refit.  See memo §8.

    Args:
        train (pd.DataFrame): full training set.
        series (list[str], optional): series keys. Defaults to SERIES.
        summing_matrix (np.ndarray, optional): summing matrix S. Defaults to S.
        min_window (int): minimum past-data rows before first eval. Default: 365.
    Returns:
        dict: {
            'mape_oos_pct': float — mean MAPE % using past-only C (no lookahead);
            'mape_insample_pct': float — mean MAPE % using full-train C;
            'optimism_gap_pct': float — mape_oos − mape_insample (positive means
                in-sample C is better, i.e., the peek provides an advantage).
        }
    """
    if series is None:
        series = SERIES
    if summing_matrix is None:
        summing_matrix = S

    n_train = len(train)

    X_rel_full   = relative_residuals(train, series)
    C_insample, _ = schafer_strimmer(X_rel_full)

    oos_apes = []
    ins_apes = []

    for t in range(min_window, n_train):
        past  = train.iloc[:t]
        today = train.iloc[t]

        pred_vec   = np.array([today[s + '_pred']   for s in series])
        actual_vec = np.array([today[s + '_actual'] for s in series])

        X_rel   = relative_residuals(past, series)
        C_oos, _ = schafer_strimmer(X_rel)

        W_oos = build_W_day(C_oos,      pred_vec)
        W_ins = build_W_day(C_insample, pred_vec)

        ytilde_oos = mint_closed(W_oos, summing_matrix, pred_vec)
        ytilde_ins = mint_closed(W_ins, summing_matrix, pred_vec)

        oos_apes.append(_mape_row(actual_vec, ytilde_oos).mean())
        ins_apes.append(_mape_row(actual_vec, ytilde_ins).mean())

    mean_oos = float(np.mean(oos_apes)) * 100
    mean_ins = float(np.mean(ins_apes)) * 100
    return {
        'mape_oos_pct':      mean_oos,
        'mape_insample_pct': mean_ins,
        'optimism_gap_pct':  mean_oos - mean_ins,
    }
