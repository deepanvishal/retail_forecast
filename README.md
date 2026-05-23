# Hierarchical Forecast Reconciliation

MinT/GLS reconciliation of a 7-series retail hierarchy with per-day heteroscedastic
covariance, Schäfer–Strimmer shrinkage, and nonnegative QP guarantee.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Outputs: `reconciled_forecasts.csv`, `backtest_results.csv`, `figures/`, and
executed `eda.ipynb` / `model.ipynb` with inline results.

To skip notebook execution (headless / CI environments):
```bash
python run.py --skip-notebooks
```

## Project Structure

```
data.py          Load and parse data; define hierarchy constants and S matrix
covariance.py    Relative residuals, Schäfer–Strimmer shrinkage, W_day builder
reconcile.py     MinT closed-form + NNLS nonneg QP + coherence/nonnegativity checks
eda.py           9 EDA figures (6 required + 3 additional), saved to figures/
backtest.py      Rolling-origin 6-method comparison + in-sample vs OOS W gap
impact.py        Base vs reconciled deltas, who-moved analysis, timeline plots
run.py           Single headless entrypoint orchestrating the full pipeline
eda.ipynb        EDA narrative (imports from modules; inline plots + decision notes)
model.ipynb      Backtest + reconciliation + validation + impact (imports from modules)
memo.md          Technical memo: methodology, validation, impact, bias disclosure,
                 required in-sample-W question
reconciled_forecasts.csv   351-row output (date + 7 series, exactly coherent)
backtest_results.csv       Per-day per-method APE from rolling-origin backtest
figures/                   All PNG plots
```

## Hierarchy

```
aggregate = cohort_A + cohort_B
cohort_A  = A1 + A2
cohort_B  = B1 + B2
```

4 leaves, 7 series total.  The 7×4 summing matrix S is defined in `data.py`.

## Key Design Decisions

| Decision | Evidence |
|---|---|
| Full 7×7 W, not leaves-only | All 7 series carry reliability signal |
| Relative residuals + per-day W_day | corr(|resid|, level) 0.62–0.82; 10× volume growth |
| Schäfer–Strimmer shrinkage | Stabilises W inverse; λ=0.0136 (safety net) |
| NNLS QP for nonnegativity | Required guarantee; 0/747 binding days in backtest |
| Disclose base-forecast bias | Under-pred on only 44–49% of days; cannot fix via reconciliation |

## Reproducibility

- All randomness: none.  Results are deterministic.
- Python ≥ 3.10 recommended.  Dependencies pinned in `requirements.txt`.
- Notebooks executed with `jupyter nbconvert --execute --inplace` (fresh kernel).

## Data

`forecast_data_anonymized.csv` — 1463 daily rows, train=1112, future=351.
One 2-day gap in train (flagged on load); does not affect covariance estimation.
