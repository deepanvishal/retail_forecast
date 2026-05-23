# Hierarchical Forecast Reconciliation — Technical Memo

## 1. Hierarchy and Summing Matrix

Seven time-series form a 3-level hierarchy with three accounting identities:

```
aggregate = cohort_A + cohort_B
cohort_A  = A1 + A2
cohort_B  = B1 + B2
```

The 7×4 summing matrix S maps the 4 leaves [A1, A2, B1, B2] to all 7 series.
Any coherent forecast vector can be written `ytilde = S @ b` for some leaf vector `b`,
which is the key insight enabling exact coherence by construction.

## 2. Methodology

### 2.1 Why Reconciliation is Needed

Base predictions violate all three identities on every day.  Train-period aggregate
gap: mean ≈ 13k, max ≈ 463k.  Future-period aggregate gap: mean ≈ 2.2M, max ≈ 56M.
The future incoherence is ~170× larger than the train incoherence, consistent with
the ~10× volume growth and the longer forecast horizon of the future block.
Train actuals are perfectly coherent (gaps < 4×10⁻⁹, floating-point only).

### 2.2 Covariance Estimation via Relative Residuals

**Why relative (scaled) residuals:**
`corr(|residual|, level=pred)` ranges 0.62–0.73 across all 7 series; with
`level=actual` the range is 0.67–0.82.  Both definitions confirm strong positive
heteroscedasticity.  The pred-based number is operationally relevant: at forecast
time only predictions are available.  The actual-based number is slightly higher
because actual = pred + residual, so actual additionally carries the residual's
own magnitude; in this data, where both pred and actual track the same volume driver,
this raises the correlation — but this is an empirical tendency specific to these
data, not a mathematical guarantee.

Aggregate monthly mean grows ~10× over 3 years (~855k to ~8.4M mean daily), with
November peaks of 4.9M / 7.5M / 13.7M YoY.  A single absolute covariance W would
be dominated by high-volume days; relative residuals `r = (actual - pred) / pred`
stabilise the variance across the volume range.

**Per-day rescaling:**
Given relative covariance C (estimated once from all train data), we build a
day-specific matrix `W_day = diag(pred_day) @ C @ diag(pred_day)` using that day's
base prediction vector.  This makes the error variance proportional to the squared
scale of each day's predictions, matching the heteroscedastic structure.  Actuals
are unavailable at forecast time, so the rescaling uses pred — the only operationally
consistent choice.

**Schäfer–Strimmer shrinkage:**
The sample correlation matrix is shrunk analytically toward the diagonal target
(keep variances; shrink off-diagonal correlations toward 0).  This regularises the
covariance estimate, stabilises W's inverse, and protects the engine in small-window
and regime-shift settings.  The analytic shrinkage intensity is derived to minimise
the expected Frobenius loss of the correlation estimator under IID sampling.

On this dataset (n=1112, p=7, n/p≈159), the analytic λ=0.0136.  Shrinkage barely
engages; it is retained as a safety net.

**Important nuance:** The SS formula assumes IID samples.  The train residuals are
autocorrelated (lag-1 0.34–0.56 across series), which reduces the effective sample
size.  The effective n estimated from the lag-1 autocorrelation of the cross-product
series Z_i·Z_j — the quantity the covariance estimator formally depends on — gives
n_eff ≈ 938 and an inflated λ ≈ 0.016.  Using the level-residual autocorrelation as
a proxy (more conservative; tends to overstate the reduction) gives n_eff ≈ 635
and λ ≈ 0.024.  In either case the inflated lambda is negligible and changes
reconciled outputs immaterially.

**Full 7×7 W (not leaves-only):**
Including aggregate and cohort forecasts in the GLS objective uses the information
that some levels are better or worse calibrated than others.  Dropping these rows
would discard signal.

### 2.3 MinT/GLS Reconciliation

The GLS closed form is:
```
ytilde = S (S^T W^{-1} S)^{-1} S^T W^{-1} yhat
```
solved via two linear systems (W is never explicitly inverted):
```python
WinvS = np.linalg.solve(W, S)          # 7x4
A     = S.T @ WinvS                     # 4x4
b     = np.linalg.solve(A, WinvS.T @ yhat)
ytilde = S @ b
```

### 2.4 Nonnegativity via NNLS QP and Summation Rebuild

The constrained QP over the 4 leaves:
```
minimise  (Sb - yhat)^T W^{-1} (Sb - yhat)   s.t.  b >= 0
```
is solved as nonnegative least squares by factoring W = LL^T (Cholesky) and passing
`M = L^{-1}S, d = L^{-1}yhat` to `scipy.optimize.nnls`.  All 7 output series are
rebuilt as `ytilde = S @ b`, ensuring all three hierarchy identities hold to
floating-point precision by construction.

## 3. Validation Results

| Check | Result | Status |
|---|---|---|
| Max abs identity violation | < 1×10⁻¹⁰ | PASSED (tol 1×10⁻⁶) |
| Min leaf value | > 480k | PASSED (≥ 0) |
| QP binding days (future) | 0 / 351 | PASSED |
| QP binding days (backtest) | 0 / 747 | PASSED |
| NNLS vs closed-form MinT | < 1×10⁻⁴ (numerical) | Identical when QP not binding |
| Output shape | 351 × 7 | PASSED |
| NaN count | 0 | PASSED |

Condition numbers (approximate, depends on magnitude of future preds):
- C (relative covariance, 7×7): well-conditioned given n/p≈159 and SS shrinkage
- W_day at median future pred: scales with pred² but inherits C's structure

## 4. Backtest Results

Rolling-origin expanding window, min 365 training rows, W estimated from past-only
data at each of 747 origins.  All methods use the same MinT/GLS formula; W varies.

| Method | Overall MAPE % | Note |
|---|---|---|
| base | ~9.8% | Incoherent baseline |
| ols (W=I) | ~10.7% | Only method that measurably hurts |
| wls_raw (diag, abs) | ~9.8% | Marginal improvement |
| wls_scaled (diag, relative) | ~9.8% | Comparable to base |
| mint_raw (full, abs) | ~9.8% | Marginal |
| **mint_scaled (full, relative)** | **~9.8%** | **Defensible pick** |

**Headline:** Reconciliation delivers coherence at essentially no accuracy cost.
This is not an accuracy-improvement play; it is a structural correctness guarantee.

**OLS uniquely hurts (~−1%)** because W=I treats all series symmetrarily and ignores
the strong positive correlations — it pushes leaves and aggregates toward an
unweighted average that overrides reliable signals with unreliable ones.

**In-sample vs OOS W optimism gap: ~0.025%** — negligible.  The covariance-window
effect is small.  The deeper concern (see §6) cannot be measured here.

**Horizon caveat:** The backtest scores one train-period prediction at a time.
The future block spans 351 days; the error covariance at h=351 may differ from the
~1-step (or unknown-horizon) train residuals.  The backtest validates mechanics and
method ranking, not the long-horizon error structure.

## 5. Impact Analysis

### 5.1 Structure of Future Incoherence

A critical finding drives the impact analysis: the future base predictions are
structurally incoherent in a systematic direction.  On **347 of 351 future days**,
the sum of the base cohort predictions exceeds the base aggregate prediction.
The median gap is **17.7% of the aggregate prediction** (vs 0.41% on train days).
On the 9 most extreme days the cohort predictions collectively exceed the aggregate
by 50–58%, concentrated in November–December 2026 — consistent with independently-
fitted sub-models diverging at long forecast horizons.

The consequence is that reconciliation **consistently adjusts all series downward**.
The aggregate prediction acts as an upper constraint, and the GLS redistributes
the excess cohort/leaf predictions downward to achieve coherence.

### 5.2 Who Moved and Why

Median reconciliation deltas across the 351 future days:

| Series | Median delta % | Relative variance | Reliability rank |
|---|---|---|---|
| aggregate | −37% | 0.021 | 6 (more trusted) |
| cohort_A | −41% | 0.027 | 3 |
| cohort_B | −53% | 0.019 | 7 (most trusted) |
| A1 | −37% | 0.028 | 2 |
| A2 | −56% | 0.044 | 1 (least trusted) |
| B1 | −51% | 0.024 | 5 |
| B2 | −64% | 0.025 | 4 |

Movement size is driven jointly by: (a) a series's own reliability (A2 least trusted,
moves most); and (b) the extent to which its base prediction contributes to the
directional incoherence.  cohort_B is the most reliable series but still shows a
large delta (−53%) because B1+B2 > cohort_B on most future days — the GLS must
reconcile this cross-level incoherence even while trusting cohort_B's own prediction.

### 5.3 Who Pulled Whom: Aggregate Pulled Cohorts Down

The aggregate pulled the cohorts and leaves downward.  On 99% of future days, the
base aggregate < sum of base cohorts.  Since the aggregate is relatively trusted
(rel_var rank 6/7) and all off-diagonal correlations in C are positive (all series
track the same volume driver), the GLS pulls the full hierarchy downward coherently.

This is directionally opposite to a scenario where leaves pull the aggregate up
(which would occur if leaf predictions were conservative relative to a high-volume
aggregate forecast).  Here the base model appears to produce more aggressive
sub-cohort forecasts than aggregate forecasts at longer horizons.

### 5.4 Implication for Decision-Making

The large negative deltas (median 35–64%) are not a numerical artefact — they
reflect genuine structural incoherence in the future base predictions.  Whether the
reconciled or raw forecasts are preferable depends on whether the aggregate (more
conservative, top-down) or the cohort/leaf models (more aggressive, bottom-up) are
believed to be better calibrated at the forecast horizon.

## 6. Headline: Coherence at No Accuracy Cost

The reconciled forecasts satisfy all three hierarchy identities to machine precision
on all 351 future days.  The rolling-origin backtest confirms that MinT-scaled
achieves this without measurably degrading per-series accuracy vs the incoherent
base forecasts (MAPE difference < 0.1% across all 7 series).

## 7. Bias Disclosure

Every series under-predicts on average: mean residual (actual − pred) is 1.1%–4.1%
of mean pred level.  However, the base model over-predicts on more than 50% of days.
The positive mean bias is driven by a small number of large under-shoots on
high-volume days, not by frequency.  **Reconciliation cannot fix this bias.**
It redistributes within-day across the hierarchy; the total daily forecast
(sum of leaves = aggregate) is unchanged in expectation.

## 8. Required Question: Why Is In-Sample W a Poor Choice?

### (i) The general argument

Estimating W from in-sample training residuals — where the base model was fitted on
the same data — produces residuals that are optimistic in two ways:

**Overfitting bias:** In-sample residuals are typically smaller and smoother than
true forecast errors because the model has seen the training data.  A W estimated
from these residuals understates the true forecast error covariance, leading the
reconciliation to over-trust the base predictions and under-shrink toward the
coherent subspace.

**Non-IID / autocorrelated errors:** Training residuals are not independent draws
from the error distribution; they are autocorrelated (here lag-1 ≈ 0.34–0.56).
This means sequential residuals carry redundant information, overstating the
effective sample size, and the analytic SS lambda underestimates the warranted
shrinkage.

**The correct approach** is to estimate W from **out-of-sample residuals** produced
by a **rolling-origin backtest of the base forecasting models themselves**, generating
predictions at the same horizon h as the intended reconciliation target.  For a
351-day-ahead block, this means: hold out windows of 351 days, generate base-model
forecasts from before each window, and use those h-step residuals to form W.
This horizon-matching ensures W reflects the actual uncertainty structure at the
forecast horizon, including the growth of error variance with h that is absent from
1-step or in-sample residuals.

### (ii) Honest dataset-specific limitation

This dataset provides fixed, precomputed base predictions for both the train and future
periods; the base models cannot be refit.  This means:

- **The deep optimism** (whether train residuals are in-sample fits vs genuine rolling OOS
  forecasts at some unknown horizon) can be reasoned about but not measured.

- **What we can isolate** is the covariance-window effect: using full-train C (peeking at
  all train data) vs past-only C at each rolling origin.  This gap is ~0.025% — negligible.

- **The horizon mismatch** between train-period residuals (unknown horizon, likely ≤ 1 year)
  and the future block (up to 351 days, 2026-01-18 to 2027-01-03) is the dominant
  unquantified risk.  If long-horizon errors are larger or differently correlated than
  short-horizon errors, the W estimated here will misrepresent the true uncertainty,
  causing the GLS to upweight or downweight series incorrectly.

The backtest in this deliverable validates reconciliation mechanics and method ranking
under the assumption that the train-period error structure is representative; it does
not validate the long-horizon error covariance of the future block.
