import numpy as np
from scipy.linalg import solve_triangular


def relative_residuals(train, series):
    """
    Compute the matrix of relative forecast errors: (actual − forecast) / forecast.

    Using relative errors rather than absolute errors stabilises variance across
    the ~10× volume range in the training data (heteroscedasticity documented in
    fig2).  The resulting matrix is the input to Schäfer–Strimmer shrinkage.

    Args:
        train (pd.DataFrame): training DataFrame with columns {s}_pred and
            {s}_actual for each s in series.
        series (list[str]): ordered list of series keys (length p).
    Returns:
        np.ndarray, shape (n, p): relative forecast errors, one column per series.
            Entry [t, i] = (actual[t, i] − pred[t, i]) / pred[t, i].
    """
    n = len(train)
    p = len(series)
    X = np.zeros((n, p))
    for i, s in enumerate(series):
        pred   = train[s + '_pred'].values
        actual = train[s + '_actual'].values
        X[:, i] = (actual - pred) / pred
    return X


def schafer_strimmer(X):
    """
    Analytic Schäfer–Strimmer (SS) shrinkage of the sample covariance matrix.

    Shrinks the sample correlation matrix toward the diagonal target (keep
    per-series variances; shrink off-diagonal correlations toward 0) using the
    closed-form shrinkage intensity λ that minimises the expected Frobenius loss
    of the correlation estimator under IID sampling.

    The formula assumes IID samples.  Residuals here are autocorrelated
    (lag-1 ≈ 0.34–0.56), so the analytic λ likely understates warranted
    shrinkage — see effective_n_sensitivity() for a quantification.  On this
    dataset (n=1112, p=7) λ = 0.0136; shrinkage barely engages but is kept as
    a stability safety net for small estimation windows.

    Args:
        X (np.ndarray, shape (n, p)): matrix of (relative or raw) residuals,
            one column per series.
    Returns:
        C_shrunk (np.ndarray, shape (p, p)): SS-shrunk covariance matrix.
            Positive semi-definite by construction.
        lam (float): shrinkage intensity λ ∈ [0, 1].  0 = no shrinkage;
            1 = diagonal target.
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
    Sensitivity check: inflate λ by n / n_eff to account for residual autocorrelation.

    The SS formula assumes IID samples and derives λ from the nominal sample size n.
    With autocorrelated residuals the effective sample size n_eff < n, so the
    warranted λ is larger.  Two proxies for n_eff are reported:

    - cross_product (formally correct): lag-1 autocorrelation of Z_i * Z_j, the
      quantity the SS covariance estimator formally depends on.
    - level_residual (proxy): lag-1 autocorrelation of the raw residual series,
      which tends to overstate the rho → more conservative (larger) n_eff reduction.

    Args:
        X (np.ndarray, shape (n, p)): relative residual matrix (same as input to
            schafer_strimmer).
        lam (float): SS λ from schafer_strimmer(X).
        n (int): nominal sample size (len(train)).
    Returns:
        dict with keys:
            rho_cross_product, rho_level_residual: mean lag-1 autocorrelations
            n_eff_cross_product, n_eff_level_residual: effective sample sizes
            lambda_inflated_cp, lambda_inflated_lv: λ inflated by n / n_eff
            note: interpretation string
    """
    p = X.shape[1]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Z = (X - mu) / sd

    import pandas as pd
    cp_ac = []
    for i in range(p):
        for j in range(i + 1, p):
            cp = Z[:, i] * Z[:, j]
            cp_ac.append(pd.Series(cp).autocorr(1))
    rho_cp = float(np.mean(cp_ac))

    lv_ac = [pd.Series(X[:, i]).autocorr(1) for i in range(p)]
    rho_lv = float(np.mean(lv_ac))

    def n_eff_ar1(rho, n):
        return n * (1 - rho ** 2) / (1 + rho ** 2)

    ne_cp = n_eff_ar1(rho_cp, n)
    ne_lv = n_eff_ar1(rho_lv, n)

    return {
        'rho_cross_product':    rho_cp,
        'rho_level_residual':   rho_lv,
        'n_eff_cross_product':  ne_cp,
        'n_eff_level_residual': ne_lv,
        'lambda_base':          lam,
        'lambda_inflated_cp':   float(np.clip(lam * n / ne_cp, 0, 1)),
        'lambda_inflated_lv':   float(np.clip(lam * n / ne_lv, 0, 1)),
        'note': (
            'cross_product is the formally correct autocorrelation for the SS '
            'covariance estimator; level_residual is a proxy that tends to give '
            'a larger rho (more conservative n_eff reduction). '
            'Both inflated lambdas remain negligible on this data.'
        ),
    }


def build_W_day(C, pred_vec):
    """
    Construct the per-day absolute forecast-error covariance matrix.

    W_day = diag(pred) @ C @ diag(pred).

    C is the relative covariance estimated from training data (scale-free).
    Scaling by pred² converts it to absolute units for day t, making the
    error variance proportional to the squared forecast level — consistent with
    the strong heteroscedasticity documented in fig2 and fig3.

    Args:
        C (np.ndarray, shape (p, p)): relative forecast-error covariance matrix
            (SS-shrunk output of schafer_strimmer).
        pred_vec (np.ndarray, shape (p,)): base forecast values for one day,
            in SERIES order.
    Returns:
        np.ndarray, shape (p, p): positive-definite per-day covariance W_day.
    """
    D = np.diag(pred_vec)
    return D @ C @ D


def condition_report(C, W_day_sample=None):
    """
    Report condition numbers for C and an optional sample W_day.

    A condition number much larger than 1 / machine epsilon (~1e16 for float64)
    would indicate near-singularity.  Values here are ~400 (C) and ~460–3200
    (W_day), well within safe bounds.

    Args:
        C (np.ndarray, shape (p, p)): relative covariance matrix.
        W_day_sample (np.ndarray, shape (p, p), optional): a representative
            W_day (e.g., built from median future forecasts).
    Returns:
        dict: {'cond_C': float, 'cond_W_day_sample': float (if provided)}.
    """
    out = {'cond_C': float(np.linalg.cond(C))}
    if W_day_sample is not None:
        out['cond_W_day_sample'] = float(np.linalg.cond(W_day_sample))
    return out
