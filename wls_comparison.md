# WLS-Scaled vs Full-W MinT — Comparison Report

**Purpose:** This is a rigor cross-check.  The shipped method is full-W MinT (brief specification).
The comparison documents the behavior difference and is not a recommendation to change the deliverable.

---

## 1. Method Definitions

| Method | W matrix | Description |
|---|---|---|
| base | — | Raw base predictions, no reconciliation |
| ols | W = I | Ordinary least squares; equal weights |
| wls_scaled | diag(W_day) | Diagonal heteroscedastic; per-day pred² scaling, no cross-series |
| **mint_scaled (shipped)** | **full W_day** | **Full 7×7 covariance; per-day pred² scaling; SS shrinkage** |

All four produce exactly coherent, nonneg outputs (via NNLS QP + summation rebuild).
The difference is entirely in how the GLS distributes the correction across series and levels.

---

## 2. Accuracy — Backtest Results

Rolling-origin expanding window, 747 origins, min 365 training rows.

| Method | Overall MAPE % | Peak-quintile MAPE % |
|---|---|---|
| base | 9.81% | 7.77% |
| ols | 9.91% | 7.80% |
| wls_scaled | **9.81%** | **7.81%** |
| mint_scaled | **9.81%** | 8.02% |

**Overall:** mint_scaled and wls_scaled are both within 0.01% of base — accuracy-neutral on average.  
**Peak days (top quintile by volume, most analogous to the future period):** mint_scaled degrades
median APE by +0.25 percentage points vs base.  wls_scaled is neutral (+0.04pp).

---

## 3. Anchor-Gate Behavior on Future Predictions

Where does the reconciled aggregate land, relative to the base aggregate and the cohort sum?

| Method | Reconciled agg BELOW base agg | Median aggregate delta |
|---|---|---|
| base (unreconciled) | — | 0% |
| ols | 6/351 (1.7%) | +8.6% (moves UP toward cohort sum) |
| wls_scaled | 14/351 (4.0%) | +19.2% (moves UP toward cohort sum) |
| mint_scaled | **339/351 (96.6%)** | **−37.2% (BELOW base agg)** |

OLS and wls_scaled both resolve the incoherence by splitting the difference — reconciled aggregate
lands between the base aggregate and the cohort sum, which is the geometrically natural resolution.

Full-W (mint_scaled) pushes the reconciled aggregate *below* the base aggregate on 96.6% of days,
with a median delta of −37% and a p5 of −61%.  This is not a reliability-weighted split —
it is an artifact of the high-correlation W applied to an incoherence regime outside its
calibration range (see §5).

---

## 4. Condition Numbers

| Quantity | Value |
|---|---|
| C (relative covariance) | κ ≈ 400 |
| W_day median-pred (all methods share this C) | κ ≈ 720 (range 460–3200) |
| W_day peak day (Nov 2026) | κ ≈ 750 |

W_day is well-conditioned on all 351 future days (max κ 3200).  Conditioning is not the source of the
over-correction problem.  The mechanism is the off-diagonal structure of C, not numerical instability.

---

## 5. Off-Diagonal Mechanism — Why Full-W Over-Corrects

C has uniformly high positive inter-series correlations (all pairwise 0.30–0.96; dominant
eigenvalue 4.95, capturing 71% of variance).  Inverting C produces large negative off-diagonal
elements: 28 of 42 off-diagonal elements of C⁻¹ are negative.  W⁻¹ row sums are negative for
aggregate, cohort_A, and cohort_B — their net GLS signal is subtractive.

When all 7 series simultaneously predict above the feasible coherent region (as on 349/351 future
days, with cohorts overshooting aggregate by a median of 17.7%), the GLS interprets this as a
shared upward error in the common factor and corrects strongly downward.

**The calibration problem:** Training data had a maximum top-level gap of 1.99%.  Future data has
a median gap of 17.7% (9–40× larger).  W was estimated on the training distribution and cannot
be validated on the future incoherence regime.  Diagonal W (wls_scaled) does not use off-diagonal
correlations and is therefore immune to this extrapolation — it simply scales corrections by
individual series reliability.

---

## 6. Synthetic Stability Test

Artificially imposing a 20% top-level gap on a typical training day (agg base 4.35M):

| Method | Reconciled aggregate | Delta vs base agg |
|---|---|---|
| ols | +12.2% (between agg and cohort sum) | +12.2% |
| wls_scaled | +21.1% (between agg and cohort sum) | +21.1% |
| mint_scaled | −57% (well below base agg) | −57% |

This confirms the over-correction is triggered by the incoherence magnitude, not by the future
prediction scale.  wls_scaled produces a 20% upward adjustment (splitting the gap); mint_scaled
extrapolates the off-diagonal correction and overshoots by a factor of ~3× below the aggregate.

---

## 7. Summary: Tradeoff

| Criterion | wls_scaled | mint_scaled (shipped) |
|---|---|---|
| Uses cross-series reliability signal | No (diagonal only) | Yes (full 7×7) |
| Anchor-gate behavior | Natural (splits gap) | Over-corrects below aggregate |
| Overall backtest accuracy | Neutral vs base | Neutral vs base |
| Peak-day backtest accuracy | Neutral (+0.04pp) | Degrades (+0.25pp) |
| Calibration-range sensitivity | Low | High |
| Condition robustness | Same (both use per-day W_day) | Same |
| Coherence and nonnegativity | Exact (NNLS QP) | Exact (NNLS QP) |

**wls_scaled is better-behaved on every dimension tested.**  It handles heteroscedasticity
correctly (the key motivation for scaling), is peak-neutral, places reconciled values
geometrically sensibly, and does not depend on the extrapolation of a high-correlation
structure to an incoherence regime outside its calibration range.

**We ship full-W MinT because the brief explicitly specifies MinT/GLS with a shrunk full
covariance matrix.**  This comparison is a transparency disclosure, not a post-hoc substitution.
The finding — that full-W mint_scaled degrades peak-day accuracy and over-corrects due to
extrapolation — is documented in §5 of the technical memo and in the verification report.
