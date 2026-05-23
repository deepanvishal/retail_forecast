# Hierarchical Forecast Reconciliation — Technical Memo

## 1. Hierarchy and Summing Matrix

Seven time-series form a 3-level hierarchy with three accounting identities:
`aggregate = cohort_A + cohort_B`, `cohort_A = A1 + A2`, `cohort_B = B1 + B2`.
The 7×4 summing matrix S maps the 4 leaves [A1, A2, B1, B2] to all 7 series.
Any coherent forecast vector is written `ytilde = S @ b`, so coherence holds
by construction once b is found.

## 2. Methodology

### 2.1 W Estimation via Relative Residuals and Per-Day Rescaling

**Why relative residuals:** `corr(|residual|, level=pred)` is 0.62–0.73 across
all 7 series; with `level=actual` the range is 0.67–0.82.  Both confirm strong
positive heteroscedasticity.  The pred-based number is operationally relevant
because actuals are unavailable at forecast time.  The actual-based number is
slightly higher because actual embeds the residual — in this data both track the
same volume driver, making the ordering an empirical tendency, not a mathematical
identity.

Aggregate monthly mean grows ~10× over 3 years (~855k to ~8.4M mean daily), with
November peaks of 4.9M → 7.5M → 13.7M YoY.  A single absolute covariance W
would be dominated by high-volume days.  Relative residuals
`r = (actual − pred) / pred` stabilise variance across the volume range.

**Per-day rescaling:** `W_day = diag(pred_day) @ C @ diag(pred_day)`.  C is the
relative covariance estimated once from all 1112 train rows; pred_day is that
day's base-prediction vector.  This makes the error variance proportional to the
squared scale of each day's predictions.

### 2.2 Schäfer–Strimmer Shrinkage

The sample correlation matrix is shrunk analytically toward the diagonal target
(keep variances; shrink off-diagonal correlations toward 0), using the closed-form
intensity λ that minimises the expected Frobenius loss under IID sampling.

On this dataset (n=1112, p=7): **λ = 0.0136**.  Shrinkage barely engages; it is
retained as a stability safety net for small windows.  The SS formula assumes IID
samples; with lag-1 residual autocorrelation 0.34–0.56, the analytic λ likely
understates the warranted shrinkage.  A sensitivity check (Appendix A) shows the
inflated λ is still negligible on this data.

### 2.3 Full 7×7 W

All 7 series — including aggregate and cohort levels — are included in the GLS
objective.  This uses the reliability signal from every level rather than leaves
only.  An important consequence is discussed in §5: because aggregate has the
largest prediction magnitude, it also has the largest absolute variance
(diag(W_day)), making it the **least trusted** series in the GLS, not an anchor.

### 2.4 MinT/GLS Closed Form

```
ytilde = S (S^T W^{-1} S)^{-1} S^T W^{-1} yhat
```
Solved via two linear systems; W is never explicitly inverted.

### 2.5 Nonnegativity via NNLS QP and Summation Rebuild

The QP `min (Sb−yhat)^T W^{-1} (Sb−yhat) s.t. b≥0` is solved as NNLS after
Cholesky-factoring W.  All levels are rebuilt as `ytilde = S @ b`, so all three
identities hold to floating-point precision by construction.

## 3. Validation

| Check | Result | Status |
|---|---|---|
| Max abs identity violation | 1.1×10⁻⁸ (relative: 2.3×10⁻¹⁶) | PASSED (tol 1×10⁻⁶) |
| Min leaf value | 155,135 | PASSED (≥ 0) |
| QP binding days (future) | 0 / 351 | PASSED |
| QP binding days (backtest) | 0 / 747 | PASSED |
| NNLS vs closed-form MinT | max dev 3.1×10⁻⁷ | Equivalent |
| Output shape / NaN | 351 × 7, 0 NaNs | PASSED |

Condition numbers: C (relative covariance) ≈ 400; W_day ranges 460–3,200 across
future days, including the November 2026 peak (750 on the peak day).

## 4. Backtest Results

Rolling-origin expanding window, min 365 training rows, W estimated from past-only
data at each of 747 origins.

| Method | Overall MAPE % | Peak-quintile MAPE % |
|---|---|---|
| base | 9.81% | 10.72% |
| ols (W=I) | 9.91% | 10.74% |
| wls_scaled (diag) | 9.81% | 10.79% |
| **mint_scaled (full-W, shipped)** | **9.81%** | **11.05%** |

In-sample vs OOS W optimism gap: 0.025% — negligible.

**Headline:** Reconciliation delivers exact coherence.  On average it is
accuracy-neutral vs base (within 0.1% MAPE across 7 series).  On peak-volume
days (top quintile, most analogous to the November–December 2026 future period),
mint_scaled degrades median aggregate APE by **+0.25 percentage points** vs base.
This is the honest qualification of the "no accuracy cost" claim; see §5 for the
mechanism.

## 5. Impact Analysis

### 5.1 Structure of Future Incoherence

The future base predictions are structurally incoherent in a systematic direction:
on **349 of 351 future days**, the sum of the base cohort predictions exceeds the
base aggregate prediction.

| Gap | Median % of parent | Max |
|---|---|---|
| g_top = cohA+cohB − agg | **+17.7%** | +57.6% |
| g_cohA = A1+A2 − cohA | +2.1% | +14.1% |
| g_cohB = B1+B2 − cohB | +10.6% | +22.1% |
| g_leaf = leaves − agg | **+24.8%** | +55.1% |

The training data **never** had a top-level gap exceeding 2.0%.  The future median
gap of 17.7% is therefore 9–40× outside the incoherence regime in which W was
estimated.

### 5.2 Where the Reconciled Aggregate Lands

| Location | Count |
|---|---|
| Reconciled aggregate **below** base aggregate | **339/351 (96.6%)** |
| Reconciled aggregate between base agg and cohort sum | 12/351 (3.4%) |
| Reconciled aggregate above cohort sum | 0/351 |

Aggregate delta distribution: median **−37%**, p5 **−61%**, p95 **−0.7%**.

### 5.3 Mechanism: Why the GLS Over-Corrects Downward

This result is not a sign-convention error or a reliability-justified pull toward
the aggregate.  The mechanism is as follows.

**Aggregate is the least-trusted series in absolute W_day terms.**  diag(W_day) =
pred² × diag(C).  Even though aggregate has the second-best relative reliability
(diag(C) rank 2 of 7), its prediction magnitude dominates: aggregate's absolute
variance is the largest on every future day (rank 7 of 7).  The GLS weights
series by 1/W_day_ii; aggregate receives the smallest weight.

**Negative W⁻¹ off-diagonal elements create a subtractive signal.**  C has high
positive inter-series correlations (all pairs 0.30–0.96; dominant eigenvalue 4.95,
capturing 71% of variance).  Inverting a matrix with large positive off-diagonal
elements produces large negative off-diagonal elements in the inverse.  28 of 42
off-diagonal elements of C⁻¹ are negative.  The row sums of W⁻¹ are negative for
aggregate, cohort_A, and cohort_B — their net GLS signal is subtractive.

**When all 7 series simultaneously predict high relative to the constraint**, the
GLS interprets this as a large shared upward error in the common factor and
over-corrects, pushing the reconciled aggregate well below any individual
prediction.  With training gaps < 2%, this mechanism produced small, accurate
corrections.  With future gaps of 17.7%+, it extrapolates strongly, pulling
the reconciled aggregate below the base aggregate on 96.6% of days.

This is confirmed by method comparison: OLS (W=I) and diagonal-W both place the
reconciled aggregate *between* base aggregate and cohort sum on ≥ 98% of future
days; only full-W produces the below-aggregate result.

### 5.4 Who Moved and How Much

Mean absolute delta (%) across the 351 future days, sorted by movement:

| Series | Rel. variance (diag C) | Abs W rank | Mean delta % | Mean \|delta\| % |
|---|---|---|---|---|
| B2 | 0.025 | 3 | −56.8% | 56.8% |
| A2 | 0.044 | 1 (most trusted abs.) | −52.3% | 52.3% |
| cohort_B | 0.019 | 5 | −48.4% | 48.4% |
| B1 | 0.024 | 2 | −47.5% | 47.5% |
| cohort_A | 0.027 | 6 | −37.7% | 37.7% |
| aggregate | 0.021 | 7 (least trusted abs.) | −34.8% | 34.8% |
| A1 | 0.028 | 4 | −34.5% | 34.5% |

All deltas are negative (all series reduced) because the shared-factor over-correction
operates symmetrically across the hierarchy.  The spread across series reflects both
individual reliability and the cross-series correlation structure.

### 5.5 Implication for Decision-Making

The 35–57% downward adjustments are a mathematical consequence of applying a
high-correlation W to an incoherence regime far outside its calibration range.
The reconciled forecasts are exactly coherent and nonnegative.  Whether they
are more useful than the raw base forecasts for decision-making depends on whether
the base model's systematic sub-cohort over-prediction (relative to the aggregate)
at long horizons is a genuine signal or a model artefact — a question the
reconciler cannot answer.  The base model calibration should be investigated upstream.

## 6. Headline: Coherence at No Average Accuracy Cost

The reconciled forecasts satisfy all three hierarchy identities on all 351 future
days (max violation 1.1×10⁻⁸; relative precision 2.3×10⁻¹⁶).  The rolling-origin
backtest confirms accuracy is neutral on average (within 0.1% MAPE).

**Qualification:** On peak-volume days (top quintile of training data, most analogous
to the November 2026 future period), mint_scaled degrades median APE by +0.25pp vs
base.  Diagonal-W (wls_scaled) is neutral on peak days.  See the method comparison
report for a fuller analysis of the tradeoff.  The brief specifies full-W MinT; that
is what ships.

## 7. Bias Disclosure

Every series under-predicts on average: mean residual (actual − pred) is 1.1%–4.1%
of mean pred level.  The model over-predicts on >50% of days; the positive mean
bias comes from a small number of large under-shoots on high-volume days.
**Reconciliation cannot fix this bias** — it redistributes within-day across the
hierarchy; the per-day total forecast is unchanged in expectation.

## 8. Required Question: Why Is In-Sample W a Poor Choice?

### (i) The general argument

Estimating W from in-sample training residuals is problematic for two reasons.

**Overfitting bias:** in-sample residuals are smaller than genuine forecast errors
because the base model was fitted on the same data.  W estimated from these residuals
understates true forecast error covariance.

**The correct approach** is to estimate W from **out-of-sample residuals** produced
by a rolling-origin backtest of the base forecasting models at the same horizon h
as the intended reconciliation target.  For a 351-day-ahead block, this means:
hold out 351-day windows, generate base-model h-step forecasts from before each
window, and estimate W from those genuine h-step residuals.  This ensures W reflects
the true uncertainty at the forecast horizon, including error-variance growth with h.

### (ii) Dataset-specific limitation

This dataset provides fixed precomputed base predictions; the base models cannot
be refit.  Therefore:

- **The deep optimism** (whether train residuals are in-sample fits vs genuine rolling
  OOS forecasts) can be reasoned about but not measured.
- **The covariance-window effect** (past-only vs full-train C) was isolated at ~0.025%
  optimism — negligible.
- **The horizon mismatch** between train residuals (unknown horizon, likely short)
  and the future block (up to 351 days) is the dominant unquantified risk.  The
  systematic future incoherence (median 17.7% gap, absent from training) is likely
  a manifestation of this mismatch: at longer horizons, independently-fitted
  sub-models diverge more than the aggregate model.  W was not calibrated for this
  regime and cannot be validated against it.

---

## Appendix A: Effective-n Sensitivity

The SS formula assumes IID samples.  With lag-1 residual autocorrelation 0.34–0.56,
effective n is approximately 635–938 (depending on whether level-residual or
cross-product autocorrelation is used as a proxy; the cross-product series — the
quantity the covariance estimator formally depends on — gives n_eff ≈ 938).  At
n_eff=938, the inflated λ ≈ 0.016; at n_eff=635, λ ≈ 0.024.  Both remain
negligible and change reconciled outputs immaterially.  The level-residual proxy
is more conservative (larger reduction in n_eff) and gives the upper bound.

## Appendix B: Additional Diagnostics

Three additional diagnostics were produced during development and are available in
`figures/`:

- **fig8_residual_tails.png:** QQ plots of relative residuals per series.  Heavy
  tails (excess kurtosis > 0) confirm covariance-based MinT is an approximation;
  it is statistically consistent but may under-weight tail events.  Log transform
  would better handle multiplicative noise but breaks additive coherence
  identities (log(aggregate) ≠ log(cohort_A) + log(cohort_B)); relative residuals
  preserve both.
- **fig9_rolling_stability.png:** 180-day rolling correlations for key series pairs.
  Broadly stable over the training period; supports using the full-train C for the
  future period.
- **Effective-n sensitivity:** see Appendix A.
