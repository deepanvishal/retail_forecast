# Hierarchical Forecast Reconciliation — Technical Memo

## Coverage Checklist

| Brief requirement | Section |
|---|---|
| Hierarchy and summing matrix S | §1 |
| Methodology: W + shrinkage + nonnegativity | §2 |
| Numerical validation (coherence + nonnegativity) | §3 |
| Impact analysis: which series moved and why | §4 |
| Required in-sample-W question | §5 |
| Interpretation: which series moved most; why | §4.2 |
| Interpretation: did aggregate pull leaves or vice versa | §4.3 |

---

## 1. What Reconciliation Is and Why It Is Needed

The base model produces seven separate forecasts.  They should add up: the cohort
forecasts should equal the total, and the leaf forecasts should sum to their cohort.
They don't — the base forecasts are *incoherent*.  On 349 of 351 future days the
two cohort forecasts together exceed the total forecast, by a median of 17.7%.

Reconciliation finds a single set of numbers that (a) add up correctly at every
level of the hierarchy, and (b) stay as close as possible to the original forecasts,
trusting the more reliable series more.

The output, `reconciled_forecasts.csv`, contains exactly coherent forecasts for all
351 future days.  The hierarchy has 7 series: Total, Cohort A, Cohort B, and four
leaves A1/A2 (under Cohort A) and B1/B2 (under Cohort B).

---

## 2. Methodology

The method is **MinT/GLS** (Minimum Trace, Generalised Least Squares) with full
7×7 covariance, Schäfer–Strimmer analytic shrinkage, and a nonnegativity guarantee.

### 2.1 Relative Errors and Per-Day Scaling

Forecast-error variance grows proportionally with forecast level (correlation
of |error| with forecast level: 0.62–0.73).  Using raw errors would let the
highest-volume days dominate the covariance estimate.  Instead, we use
*relative errors* = (actual − forecast) / forecast, which are roughly
homoscedastic across the ~10× volume range in the training data.

The covariance matrix for each day is then:
`W_day = diag(forecast_day) × C × diag(forecast_day)`
where C is a single relative covariance estimated from all training data.
This makes error variance proportional to the squared forecast on that day.

### 2.2 Schäfer–Strimmer Shrinkage

C is shrunk analytically toward the diagonal target (preserve per-series
variances; shrink off-diagonal correlations toward zero), using the closed-form
intensity λ that minimises expected estimation error under independent sampling.
On this dataset: **λ = 0.0136** — shrinkage barely engages but is retained as a
stability safeguard for small estimation windows.

### 2.3 GLS Closed Form and Nonnegativity

The reconciled forecast minimises
`(S b − yhat)ᵀ W⁻¹ (S b − yhat)  subject to  b ≥ 0`
where S is the 7×4 summing matrix (the "adding-up map") and b is the 4-leaf
vector.  Solved via NNLS after Cholesky-factoring W (W is never explicitly
inverted).  All 7 series are rebuilt as `ytilde = S @ b`, so all three hierarchy
identities hold to floating-point precision by construction.

---

## 3. Validation

| Check | Result | Status |
|---|---|---|
| Max absolute identity violation | 3.7×10⁻⁹ (relative: <10⁻¹⁵) | PASSED (tol 10⁻⁶) |
| Min leaf value | 155,135 | PASSED (≥ 0) |
| QP binding days (future) | 0 / 351 | PASSED |
| QP binding days (backtest) | 0 / 747 | PASSED |
| NNLS vs closed-form MinT | max dev 3.1×10⁻⁷ | Equivalent |
| Output shape / NaN | 351 × 7, 0 NaNs | PASSED |

---

## 4. Impact Analysis

### 4.1 Headline Accuracy

Rolling-origin backtest (747 origins, min 365 training rows, W from past-only data):

| Method | Overall MAPE % | Peak-quintile MAPE % |
|---|---|---|
| Base (no reconciliation) | 9.81% | 7.77% |
| OLS — equal weights | 9.91% | 7.80% |
| WLS-scaled (diagonal) | 9.81% | 7.81% |
| **MinT-scaled (shipped)** | **9.81%** | **8.02%** |

**On average, reconciliation is accuracy-neutral** (within 0.1% MAPE vs base).
**Qualification:** On peak-volume days (top quintile of training data — most
analogous to the November 2026 future period), MinT-scaled degrades median
absolute percentage error by **+0.25 percentage points** vs base.  Diagonal WLS
is neutral (+0.04pp) on the same days.  The brief specifies full-W MinT; see
`wls_comparison.md` for the full tradeoff analysis.

### 4.2 Which Series Moved Most, and Why

Mean absolute adjustment (% of base forecast) per series:

| Series | Rel. error variance | Mean \|adjustment\| % |
|---|---|---|
| B2 (Cohort B leaf) | 0.025 | −56.8% |
| A2 (Cohort A leaf) | 0.044 | −52.3% |
| Cohort B | 0.019 | −48.4% |
| B1 (Cohort B leaf) | 0.024 | −47.5% |
| Cohort A | 0.027 | −37.7% |
| Total | 0.021 | −34.8% |
| A1 (Cohort A leaf) | 0.028 | −34.5% |

**All series were adjusted downward.**  The spread across series reflects both
individual reliability (relative error variance) and the cross-series correlation
structure — not a simple monotone ranking.

### 4.3 The "Who Pulled Whom" Question — And the Honest Answer

The reconciled Total falls **below the base Total forecast** on 339 of 351 future
days (96.6%), with a median delta of **−37%**.  This is not the Total acting as an
upper constraint pulling leaves down.

**The mechanism:** The future base predictions are systematically incoherent
(cohorts exceed total by a median of 17.7%).  The covariance matrix C has uniformly
high positive inter-series correlations (dominant eigenvalue 4.95, capturing 71% of
variance; all pairs 0.30–0.96).  When a matrix with large positive off-diagonal
elements is inverted, it produces large *negative* off-diagonal elements.  In W⁻¹,
the row sums for Total, Cohort A, and Cohort B are negative — their net GLS weight
is subtractive.  When all 7 series simultaneously over-predict relative to the
feasible coherent region, the GLS interprets this as evidence of a shared upward
error in the common factor and corrects strongly downward.

**The calibration-range problem:** Training data had a maximum top-level gap of
1.99% (median ~0.3%).  The future median gap is 17.7% — 9–40× outside the regime
in which W was estimated.  The method extrapolates.  For comparison:
- OLS (W = I): reconciled Total lands *between* base Total and cohort sum on
  98.3% of future days, with a median delta of +8.6%.
- Diagonal WLS: same direction, median delta +19.2%.
- Full-W MinT: below base Total on 96.6% of days, median −37%.

The 35–57% downward adjustments are a mathematical consequence of applying a
high-correlation W to an incoherence regime far outside its calibration range.
The base model calibration should be investigated upstream.

---

## 5. Required Question: Why Is In-Sample W a Poor Choice?

### (i) General argument

Estimating W from in-sample training residuals has two problems.

**Overfitting bias:** In-sample residuals are smaller than genuine forecast errors
because the base model was fitted on the same data.  W estimated from these residuals
understates true forecast error covariance.

**The correct approach** is to estimate W from **out-of-sample residuals** produced
by a rolling-origin backtest of the base models at the same forecast horizon h as the
reconciliation target.  For a 351-day future block, this means: hold out 351-day
windows, generate h-step base-model forecasts from before each window, and estimate W
from those genuine h-step residuals.  This ensures W reflects true forecast uncertainty
at the intended horizon, including error-variance growth with h.

### (ii) Dataset-specific limitation

This dataset provides fixed pre-computed base predictions; the base models cannot
be refit.  Therefore:

- **Covariance-window optimism** (whether full-train vs past-only C improves accuracy)
  was isolated at ~0.025% — negligible.
- **Horizon mismatch** (training residuals at unknown horizon vs 351-day future block)
  is the dominant unquantified risk.  The systematic future incoherence (median 17.7%
  gap, absent from training) is likely a manifestation: at longer horizons, independently-
  fitted sub-models diverge more than the aggregate model.  W was not calibrated for this
  regime and cannot be validated against it.

---

## 6. Bias Disclosure

Mean forecast error (actual − forecast) is positive for all 7 series: 1.1%–4.1%
of mean forecast level.  The model over-predicts on >50% of days; the positive mean
bias comes from large under-shoots on high-volume days.  **Reconciliation cannot fix
this** — it redistributes within a day across the hierarchy; the per-day total
forecast is unchanged in expectation.

---

## Appendix A: Effective-n Sensitivity

The SS formula assumes IID samples.  With lag-1 residual autocorrelation 0.34–0.56,
the effective sample size n_eff < 1,112.  Two proxies:

- Cross-product lag-1 autocorrelation (formally correct for the SS estimator): ρ = 0.292,
  n_eff ≈ 938, inflated λ ≈ 0.016
- Level-residual proxy (conservative upper bound): ρ = 0.522, n_eff ≈ 635, inflated λ ≈ 0.024

Both inflated λ values remain negligible and change reconciled outputs immaterially.

## Appendix B: Additional Diagnostics

Available in `figures/`:

- **fig8\_residual\_tails.png:** QQ plots of relative forecast errors per series.  Heavy
  tails (excess kurtosis > 0) confirm covariance-based MinT is an approximation; it is
  statistically consistent but may under-weight tail events.
- **fig9\_rolling\_stability.png:** 180-day rolling correlations for key series pairs.
  Broadly stable over the training period; supports using the full-train C for the
  future period.
- **Condition numbers:** C ≈ 400; W_day ranges 460–3,200 across future days (peak day 750).
  Well-conditioned throughout; numerical instability is not the source of the large deltas.
