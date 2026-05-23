# Verification Report: Large Downward Reconciliation Deltas

## Verdict

**Verdict (c): the downward pull is a calibration-range artifact of the full-W off-diagonal
structure, not a reliability-weighted adjustment. It cannot be validated against ground truth
because the incoherence regime in the future (median gap 17.7%) never appeared in training
(max gap 1.99%). On the subset of training days most analogous to the future (peak volume),
mint_scaled degrades median accuracy vs base. The memo's "who pulled whom" framing is
factually wrong in two specific sentences identified below.**

---

## Task 1: Incoherence Decomposition Across All Levels

All gaps are directional (cohort/leaf overshoots aggregate), not random noise.

| Gap | Sign (+= overshoot) | Median % of parent | Days positive | Max |
|---|---|---|---|---|
| g_top = cohA+cohB − agg | +17.7% | 349/351 | +57.6% |
| g_cohA = A1+A2 − cohA | +2.1% | 289/351 | +14.1% |
| g_cohB = B1+B2 − cohB | +10.6% | 305/351 | +22.1% |
| g_leaf = leaves − agg | +24.8% | 349/351 | +55.1% |

**How 17.7% bridges to 35–64% leaf deltas:** the leaf-to-aggregate gap is 24.8% (not 17.7%),
because the within-cohort B gap (+10.6%) compounds with the top-level gap. g_leaf = g_top +
weighted(g_cohA + g_cohB). The arithmetic explains the delta magnitude partially — but not
fully. The remaining amplification comes from Task 5 (W's off-diagonal structure, not
incoherence geometry). Task 5 is the decisive driver.

---

## Task 2: Reliability Ranking — Relative (C) vs Absolute (W_day)

C diagonal (scale-free, relative reliability):

| Series | diag(C) | Relative rank (1=most reliable) |
|---|---|---|
| cohort_B | 0.01940 | 1 |
| aggregate | 0.02074 | 2 |
| B1 | 0.02390 | 3 |
| B2 | 0.02526 | 4 |
| cohort_A | 0.02704 | 5 |
| A1 | 0.02747 | 6 |
| A2 | 0.04444 | 7 |

diag(W_day) = pred² × diag(C) — what MinT actually uses for weighting:

| Series (median-vol day) | diag(W_day) | Absolute rank (1=most trusted by GLS) |
|---|---|---|
| A2 | 2.4e+10 | 1 |
| B1 | 4.3e+10 | 2 |
| B2 | 6.9e+10 | 3 |
| A1 | 1.2e+11 | 4 |
| cohort_B | 1.3e+11 | 5 |
| cohort_A | 2.1e+11 | 6 |
| **aggregate** | **3.9e+11** | **7 (LEAST trusted)** |

**Aggregate has the largest absolute variance on every future day (rank 7 of 7).** The relative
ranking (diag(C) rank 2) and the absolute ranking (rank 7) are inverted because aggregate has
the largest prediction magnitude. The GLS uses absolute W, not relative C. The memo's
statement that aggregate is "relatively trusted (rel_var rank 6/7)" conflates the two
rankings — rel_var rank 6 means relative variance is 6th largest (2nd most reliable in relative
terms), but that translates to the largest absolute variance because aggregate dominates by scale.

---

## Task 3: Who Pulled Whom — Measured

Where the reconciled aggregate lands, relative to base aggregate and cohort sum:

| Location | Count | % |
|---|---|---|
| Reconciled agg BELOW base aggregate | 339/351 | **96.6%** |
| Reconciled agg between base agg and cohort sum | 12/351 | 3.4% |
| Reconciled agg above cohort sum | 0/351 | 0% |

**The reconciled aggregate does not land between the two predictions — it overshoots BELOW the
aggregate on 96.6% of future days.** Median aggregate delta: −37.2% (p5: −60.8%, p95: −0.7%).

For comparison, OLS places the reconciled aggregate between base aggregate and cohort sum on
345/351 days (98.3%) with a median delta of +8.6% (moves aggregate UP toward the cohort-sum).

The "aggregate as upper constraint / anchor" story is OLS's story. Under full-W, the aggregate
is not an anchor — it is the least-trusted series (in absolute terms) and the reconciled value
falls well below it.

---

## Task 4: Ground-Truth Test — The Decisive One

**Critical finding: the training data never had a top-level gap > 2.0%. The future has a
median gap of 17.7%, with 248/351 days exceeding 10%. W is calibrated to a fundamentally
different incoherence regime and is being extrapolated 9–40× beyond its training distribution.**

| Subset | n | base median APE | mint_scaled median APE | verdict |
|---|---|---|---|---|
| All backtest days | 747 | 6.50% | 6.47% | neutral (−0.03%) |
| Overshoot train days (gap < 2%) | 214 | 7.14% | 7.08% | marginal help |
| **Peak (top-quintile) days** | **198** | **7.77%** | **8.02%** | **HURTS (+0.25%)** |
| Peak + overshoot | 49 | 7.11% | 6.55% | small help |

On peak days (the training analogue of the high-volume future days), mint_scaled **degrades**
median APE by 0.25 percentage points. wls_diag on the same days: 7.81% (essentially neutral
vs base). OLS: 7.80%.

The overall backtest being accuracy-neutral hides this peak-day degradation. The headline
"no accuracy cost" is true on average but false on the tail of the distribution that matters most.

On the analogous peak+overshoot days (n=49), mint_scaled helps (7.11% → 6.55%), but:
(a) the maximum training gap there is still < 2%; (b) wls_diag achieves 6.54% without the
over-correction problem; (c) this is a sample of 49 days vs the 297+ future days with gap > 5%.

---

## Task 5: W's Contribution vs Pure Geometry

Results for the same two days under three W choices:

| Method | Median-vol day (agg base 4.35M) | Peak day (agg base 108.6M) |
|---|---|---|
| OLS (W=I) | rec_agg = 4.84M (+11.5%) | 121.9M (+12.2%) |
| diag-W | rec_agg = 5.26M (+21.1%) | 131.1M (+20.7%) |
| full-W | rec_agg = 2.48M (−42.9%) | 41.3M (−62.0%) |

OLS and diagonal-W both resolve the incoherence by moving the aggregate **upward** toward the
cohort sum (the natural geometric resolution: split the difference). Only full-W produces
the large downward overshoot. The "leaves down" result is NOT driven by the incoherence
geometry or the summing constraint — it is entirely driven by the off-diagonal correlation
structure of W.

**Mechanism:** C has high positive correlations (dominant eigenvalue 4.95 of 7-series spectrum;
all pairwise correlations 0.30–0.98). Inverting a matrix with large positive off-diagonal
elements produces large negative off-diagonal elements in the inverse. W^{-1} row sums are
NEGATIVE for aggregate, cohort_A, and cohort_B (their net signal to the GLS is subtractive).

When all 7 series simultaneously predict high (as they do on every future day, given the
systematic g_top > 0 incoherence), the GLS interprets this as evidence of a shared upward
error in the common factor and corrects it strongly — pushing the reconciled value well below
any individual prediction.

This is mathematically self-consistent within the GLS framework, but it requires the
correlation structure to correctly represent the joint error distribution at the encountered
magnitude of incoherence. Training gaps were < 2%; future gaps are 17.7%. The model is
extrapolating, not interpolating.

A synthetic test confirms: artificially imposing a 20% top-level gap on a training day
produces −57% aggregate delta — identical in magnitude to the actual future behavior.

---

## Task 6: Correlation Routing

| Pair | Correlation |
|---|---|
| agg–A1 | 0.955 |
| agg–cohort_A | 0.958 |
| cohort_A–A1 | 0.979 |
| agg–B1 | 0.719 |
| cohort_B–B1 | 0.958 |
| A1–A2 | 0.671 |
| B1–B2 | 0.404 |
| cohort_A–cohort_B | 0.604 |

The aggregate–cohort_A–A1 chain is near-perfect (all ~0.96). In the inverse W^{-1}, these
near-collinear series "cancel" each other, producing the large negative row sums for upper levels.

Sibling routing: A1–A2 correlation (0.671) routes the within-cohort A adjustment so A2 moves
more than A1 (A2 is less reliable and more tightly constrained by the correlation). B1–B2
correlation is weaker (0.404), so B2 moves more independently. The within-leaf splits are
consistent with the correlation structure and are the least problematic part of the result.

---

## Task 7: Artifact and Bug Checks

| Check | Result |
|---|---|
| Sign convention (cohorts > agg → reconciliation reduces) | Correct, no flipped sign |
| W_day uses that day's future pred vector | Confirmed; per-day construction verified |
| NNLS == closed-form MinT (0/351 binding) | Max deviation 3.05e-07; equivalent |
| W_day condition numbers | min 462, median 720, max 3,200 (peak day 750) — well-conditioned |
| C is PSD | Confirmed; SS shrinkage ensures PSD |

**No bugs found.** The result is mathematically correct given the inputs. The problem is not
an implementation defect but a calibration-range extrapolation.

---

## Task 8: Masking Question

The directional gap (cohorts systematically exceed aggregate by 17.7% median) is not random
incoherence — it is a structural bias between independently-fitted model groups. Reconciliation
delivers one coherent number per day but cannot identify which group is miscalibrated.

**Ground-truth evidence from Task 4:** on train overshoot days, the aggregate prediction is
closer to the actual aggregate on 57–62% of days (slight but consistent edge). This mildly
supports trusting the aggregate, but: (a) all training overshoot days have gaps < 2%, far from
the 17.7% median future gap; (b) the reconciled value goes far below the aggregate, not just
to it; (c) on peak days where future concentrates, mint_scaled hurts accuracy.

**What can be said:** the future base model produces sub-cohort predictions approximately
20% larger than the aggregate prediction on nearly every future day. This is a systematic
calibration issue in the base model — the independently-fitted sub-models are more aggressive
at longer forecast horizons than the aggregate model. Reconciliation will impose an arbitrary
compromise that is sensitive to the W choice. This warrants upstream investigation of the
base model.

---

## Peak-Period Check

Restricting to top-quintile training days (volume-matched to future peak period):
- **mint_scaled median APE: 8.02%** vs base 7.77% — **degradation of +0.25%**
- wls_diag: 7.81% (neutral, no degradation)
- OLS: 7.80% (neutral)

The overall backtest headline (accuracy-neutral within 0.1%) masks peak-day degradation.
For a retailer making allocation decisions during the high-stakes November peak, mint_scaled
is the only reconciler that measurably hurts.

---

## Corrections Required in memo.md

**§5.3 "Who Pulled Whom" — two sentences to correct:**

1. **Incorrect:** *"The aggregate prediction acts as an upper constraint"*
   **Correct:** The reconciled aggregate falls below the base aggregate on 96.6% of future days.
   There is no upper constraint; the aggregate is the least trusted series in absolute W_day
   terms. OLS and diagonal-W would correctly place the reconciled aggregate between base
   aggregate and cohort sum. Full-W overshoots below both.

2. **Incorrect:** *"The aggregate pulled the cohorts and leaves downward"*
   **Correct:** The off-diagonal correlations in W^{-1} (which are negative due to high
   positive inter-series correlations) create a subtractive signal that pushes all series
   below their predictions. The aggregate does not "pull" anything; it is the series with
   the largest absolute variance and the weakest GLS anchor.

**§6 "Headline: Coherence at No Accuracy Cost" — qualification required:**
The headline holds on average (within 0.1% MAPE). On peak-volume days (top quintile,
most analogous to the high-stakes future period), mint_scaled degrades median accuracy by
+0.25 percentage points vs base. "No accuracy cost on average" would be accurate. "No
accuracy cost" without qualification is not.

---

## Recommended Action

The current full-W produces a mathematically valid but calibration-range-dependent result
that degrades peak-day accuracy. Two defensible alternatives:

1. **Use diagonal-W (wls_scaled) for the final deliverable.** It handles heteroscedasticity
   correctly, is peak-neutral, places reconciled values sensibly between predictions, and
   avoids the off-diagonal extrapolation problem. The memo's justification for full-W
   ("full 7×7 W uses reliability of aggregate and cohort forecasts") does not survive
   scrutiny: in absolute terms, including those series makes the aggregate the least-trusted
   input, which is the opposite of the claimed justification.

2. **Keep full-W but disclose the over-correction explicitly**, with a caveat that the future
   incoherence regime (median 17.7%) is ~9–40× outside the training distribution (max 2%)
   and the reconciled values should not be interpreted as a reliability-weighted split between
   the aggregate and cohort predictions — they are a calibration extrapolation.

Given that wls_scaled matches or beats mint_scaled on every train subset examined and does
not require the untestable extrapolation, **Option 1 is the stronger deliverable**.
