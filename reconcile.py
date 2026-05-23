import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls


def _mint_b(W, S, yhat):
    WinvS = np.linalg.solve(W, S)
    A = S.T @ WinvS
    return np.linalg.solve(A, WinvS.T @ yhat)


def mint_closed(W, S, yhat):
    """
    MinT/GLS closed form: ytilde = S (S^T W^{-1} S)^{-1} S^T W^{-1} yhat.
    Solves two linear systems; W is never explicitly inverted.
    """
    return S @ _mint_b(W, S, yhat)


def nnls_qp(W, S, yhat):
    """
    Nonneg QP over the 4 leaves:
        minimise  (Sb - yhat)^T W^{-1} (Sb - yhat)   s.t.  b >= 0

    Factored as NNLS: factor W = L L^T (Cholesky, lower);
    let M = L^{-1} S, d = L^{-1} yhat; solve scipy.optimize.nnls(M, d).

    Upper levels are rebuilt by summation ytilde = S @ b, so all three
    hierarchy identities are satisfied to floating-point precision by construction.

    Returns (ytilde, b, qp_active) where qp_active=True if the unconstrained
    MinT leaf solution had any negative component (i.e., QP bound bound was binding).
    """
    L = np.linalg.cholesky(W)
    M = solve_triangular(L, S, lower=True)
    d = solve_triangular(L, yhat, lower=True)
    b, _ = nnls(M, d)
    ytilde = S @ b

    b_unc = _mint_b(W, S, yhat)
    qp_active = bool(np.any(b_unc < -1e-8))
    return ytilde, b, qp_active


def coherence_gaps(ytilde):
    return {
        'agg_minus_cohorts': abs(ytilde[0] - ytilde[1] - ytilde[2]),
        'cohortA_minus_leaves': abs(ytilde[1] - ytilde[3] - ytilde[4]),
        'cohortB_minus_leaves': abs(ytilde[2] - ytilde[5] - ytilde[6]),
    }


def assert_coherent(ytilde, tol=1e-6):
    gaps = coherence_gaps(ytilde)
    for name, val in gaps.items():
        assert val < tol, f'Coherence violation {name}: {val:.2e} (tol={tol:.0e})'


def assert_nonneg(ytilde):
    leaves = ytilde[3:]
    assert np.all(leaves >= 0), f'Negative leaf: min={leaves.min():.4f}'
    return float(leaves.min())
