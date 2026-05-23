"""
MinT/GLS hierarchical forecast reconciliation.

Plain-language guide to what each step does
────────────────────────────────────────────
RECONCILIATION PROBLEM
  The base forecasts for the 7 series are incoherent: they do not add up
  correctly (cohort totals do not equal the total, leaves do not equal their
  cohort parent).  Reconciliation finds a coherent set of numbers that are as
  close as possible to the original forecasts.

S — the summing matrix (7 × 4)
  S is the "adding-up map": given 4 leaf values, S produces all 7 series by
  addition.  Any vector ytilde = S @ b is coherent by construction, because
  the additions are encoded in S's structure.

W — the reliability scorecard (7 × 7, symmetric positive-definite)
  W describes how uncertain each base forecast is.  A large diagonal entry
  W[i,i] means series i is unreliable (large expected forecast error); a small
  entry means reliable.  The GLS objective uses 1/W as the weight: reliable
  series are moved less.  W is never explicitly inverted — linear solves are
  used throughout.

GLS closed form (mint_closed)
  Find the leaf vector b that minimises the weighted squared distance from the
  reconciled forecast (S @ b) to the base forecast (yhat), with weights 1/W:

      minimise  (S b − yhat)ᵀ W⁻¹ (S b − yhat)

  Setting the gradient to zero gives the closed-form solution:

      b* = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ yhat

  Implemented via two linear solves; W is never explicitly inverted.
  All 7 series are rebuilt as ytilde = S @ b*.

NNLS step (nnls_qp)
  The closed-form solution can produce negative leaf values (impossible in
  context).  Adding the constraint b ≥ 0 turns the problem into a
  non-negative least squares (NNLS) problem:

      minimise  (S b − yhat)ᵀ W⁻¹ (S b − yhat)   subject to  b ≥ 0

  Solved by Cholesky-factoring W = L Lᵀ, then transforming to standard NNLS:

      minimise  ‖ L⁻¹ S b − L⁻¹ yhat ‖²   subject to  b ≥ 0

  scipy.optimize.nnls solves this efficiently.

S @ b rebuild
  Once b is found, all 7 series are computed as ytilde = S @ b.
  This rebuilds cohort totals and the grand total by simple addition, so all
  three hierarchy identities hold to floating-point precision by construction —
  not by post-hoc rounding.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls


def _mint_b(W, S, yhat):
    """
    Solve for the optimal leaf vector b* in the unconstrained GLS objective.

    Computes b* = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ yhat via two linear solves.
    W is never explicitly inverted.

    Args:
        W (np.ndarray, shape (7, 7)): symmetric positive-definite forecast-error
            covariance matrix for one day.
        S (np.ndarray, shape (7, 4)): summing matrix.
        yhat (np.ndarray, shape (7,)): base forecast vector for one day.
    Returns:
        np.ndarray, shape (4,): unconstrained optimal leaf vector b*.
    """
    WinvS = np.linalg.solve(W, S)
    A = S.T @ WinvS
    return np.linalg.solve(A, WinvS.T @ yhat)


def mint_closed(W, S, yhat):
    """
    Closed-form MinT/GLS reconciliation: ytilde = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ yhat.

    Finds the coherent forecast vector closest to yhat in the W⁻¹ (reliability-
    weighted) metric.  Does not enforce nonnegativity — use nnls_qp when leaves
    must be ≥ 0.

    Args:
        W (np.ndarray, shape (7, 7)): per-day forecast-error covariance matrix.
        S (np.ndarray, shape (7, 4)): summing matrix mapping 4 leaves to 7 series.
        yhat (np.ndarray, shape (7,)): base forecast vector for one day.
    Returns:
        np.ndarray, shape (7,): coherent reconciled forecast in SERIES order
            [aggregate, cohort_A, cohort_B, A1, A2, B1, B2].
    Notes:
        W is never explicitly inverted; two linear systems are solved instead.
    """
    return S @ _mint_b(W, S, yhat)


def nnls_qp(W, S, yhat):
    """
    Non-negative constrained reconciliation: same GLS objective as mint_closed,
    with the additional constraint that all 4 leaf values must be ≥ 0.

    Solves: minimise (S b − yhat)ᵀ W⁻¹ (S b − yhat)  subject to  b ≥ 0.

    Method: Cholesky-factor W = L Lᵀ; transform to standard NNLS
    (minimise ‖ L⁻¹ S b − L⁻¹ yhat ‖² s.t. b ≥ 0); solve with scipy.nnls.
    Upper levels are rebuilt as ytilde = S @ b so all three hierarchy identities
    hold to floating-point precision by construction.

    Args:
        W (np.ndarray, shape (7, 7)): per-day forecast-error covariance matrix.
        S (np.ndarray, shape (7, 4)): summing matrix.
        yhat (np.ndarray, shape (7,)): base forecast vector for one day.
    Returns:
        ytilde (np.ndarray, shape (7,)): coherent, nonneg reconciled forecast
            in SERIES order.
        b (np.ndarray, shape (4,)): nonneg optimal leaf vector.
        qp_active (bool): True if the unconstrained MinT solution had any
            negative leaf component (i.e., the nonnegativity constraint bound).
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
    """
    Compute the absolute identity-violation at each of the three hierarchy levels.

    Args:
        ytilde (np.ndarray, shape (7,)): forecast vector in SERIES order
            [aggregate, cohort_A, cohort_B, A1, A2, B1, B2].
    Returns:
        dict with keys:
            'agg_minus_cohorts': |ytilde[0] − ytilde[1] − ytilde[2]|
            'cohortA_minus_leaves': |ytilde[1] − ytilde[3] − ytilde[4]|
            'cohortB_minus_leaves': |ytilde[2] − ytilde[5] − ytilde[6]|
    """
    return {
        'agg_minus_cohorts':    abs(ytilde[0] - ytilde[1] - ytilde[2]),
        'cohortA_minus_leaves': abs(ytilde[1] - ytilde[3] - ytilde[4]),
        'cohortB_minus_leaves': abs(ytilde[2] - ytilde[5] - ytilde[6]),
    }


def assert_coherent(ytilde, tol=1e-6):
    """
    Assert all three hierarchy identities hold to within tol.

    Args:
        ytilde (np.ndarray, shape (7,)): reconciled forecast vector.
        tol (float): absolute tolerance. Default: 1e-6.
    Raises:
        AssertionError: if any identity violation exceeds tol, with the
            violating gap name and value in the message.
    """
    gaps = coherence_gaps(ytilde)
    for name, val in gaps.items():
        assert val < tol, f'Coherence violation {name}: {val:.2e} (tol={tol:.0e})'


def assert_nonneg(ytilde):
    """
    Assert all four leaf forecasts are nonneg and return the minimum leaf value.

    Args:
        ytilde (np.ndarray, shape (7,)): reconciled forecast vector.
    Returns:
        float: minimum leaf value (ytilde[3:].min()).
    Raises:
        AssertionError: if any leaf is negative.
    """
    leaves = ytilde[3:]
    assert np.all(leaves >= 0), f'Negative leaf: min={leaves.min():.4f}'
    return float(leaves.min())
