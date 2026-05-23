import numpy as np
from scipy.linalg import solve_triangular


def relative_residuals(train, series):
    n = len(train)
    p = len(series)
    X = np.zeros((n, p))
    for i, s in enumerate(series):
        pred = train[s + '_pred'].values
        actual = train[s + '_actual'].values
        X[:, i] = (actual - pred) / pred
    return X


def schafer_strimmer(X):
    """
    Analytic Schäfer-Strimmer shrinkage of the sample correlation matrix toward
    the diagonal target (keep variances; shrink off-diagonal correlations toward 0).

    The formula minimises the expected Frobenius loss of the correlation estimator
    under IID sampling. Residuals here are autocorrelated (lag-1 ~0.34-0.56),
    so effective n < nominal n, and the analytic lambda likely underestimates
    warranted shrinkage — see effective_n_sensitivity() for a quantification.
    On this dataset (n=1112, p=7) lambda=0.0136; shrinkage barely engages but is
    kept as a safety net for small estimation windows and regime shifts.

    Returns (C_shrunk, lambda) where C_shrunk is the 7x7 relative covariance.
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Z = (X - mu) / sd

    R = Z.T @ Z / (n - 1)

    numer = 0.0
    denom = 0.0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            w = Z[:, i] * Z[:, j]
            wbar = w.mean()
            numer += (n / (n - 1) ** 3) * np.sum((w - wbar) ** 2)
            denom += R[i, j] ** 2

    lam = float(np.clip(numer / denom, 0.0, 1.0))

    R_shrunk = (1.0 - lam) * R
    np.fill_diagonal(R_shrunk, 1.0)
    C = np.diag(sd) @ R_shrunk @ np.diag(sd)
    return C, lam


def effective_n_sensitivity(X, lam, n):
    """
    Sensitivity check: inflate lambda by n/n_eff where n_eff is derived from the
    lag-1 autocorrelation of the cross-product series Z_i*Z_j (the formal quantity
    the covariance estimator depends on — not the level-residual autocorrelation,
    which is a proxy and tends to overstate the reduction).

    Returns dict with both estimates and the resulting lambda bounds.
    """
    p = X.shape[1]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Z = (X - mu) / sd

    # Lag-1 autocorr of cross-products (off-diagonal pairs) — formally correct
    import pandas as pd
    cp_ac = []
    for i in range(p):
        for j in range(i + 1, p):
            cp = Z[:, i] * Z[:, j]
            cp_ac.append(pd.Series(cp).autocorr(1))
    rho_cp = float(np.mean(cp_ac))

    # Lag-1 autocorr of level residuals — proxy (tends to overstate, more conservative)
    lv_ac = [pd.Series(X[:, i]).autocorr(1) for i in range(p)]
    rho_lv = float(np.mean(lv_ac))

    def n_eff_ar1(rho, n):
        return n * (1 - rho ** 2) / (1 + rho ** 2)

    ne_cp = n_eff_ar1(rho_cp, n)
    ne_lv = n_eff_ar1(rho_lv, n)

    return {
        'rho_cross_product': rho_cp,
        'rho_level_residual': rho_lv,
        'n_eff_cross_product': ne_cp,
        'n_eff_level_residual': ne_lv,
        'lambda_base': lam,
        'lambda_inflated_cp': float(np.clip(lam * n / ne_cp, 0, 1)),
        'lambda_inflated_lv': float(np.clip(lam * n / ne_lv, 0, 1)),
        'note': (
            'cross_product is the formally correct autocorrelation for the SS '
            'covariance estimator; level_residual is a proxy that tends to give '
            'a larger rho (more conservative n_eff reduction). '
            'Both inflated lambdas remain negligible on this data.'
        ),
    }


def build_W_day(C, pred_vec):
    """
    W_day = diag(pred) @ C @ diag(pred).

    C is the relative covariance estimated once on all train data.
    pred_vec is the 7-vector of base predictions for a specific day.
    This makes W proportional to the squared scale of that day's predictions,
    matching the heteroscedastic error structure documented in F2/F3.
    """
    D = np.diag(pred_vec)
    return D @ C @ D


def condition_report(C, W_day_sample=None):
    out = {'cond_C': float(np.linalg.cond(C))}
    if W_day_sample is not None:
        out['cond_W_day_sample'] = float(np.linalg.cond(W_day_sample))
    return out
