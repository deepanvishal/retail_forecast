# Hierarchical Forecast Reconciliation

MinT/GLS reconciliation of a 7-series retail hierarchy with per-day heteroscedastic
covariance, Schäfer–Strimmer shrinkage, and nonnegative QP guarantee.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

The two files the brief asks for:
- **`reconciled_forecasts.csv`** — 351-row output, one column per series, exactly coherent at every hierarchy level
- **`memo.md`** — technical memo: methodology, validation, impact analysis, bias disclosure, and the required in-sample-W question

To skip notebook execution (headless / CI environments):
```bash
python run.py --skip-notebooks
```

---

## Deliverables

### Core (what the brief asks for)

| File | What it is |
|---|---|
| `reconciled_forecasts.csv` | **The answer.** 351-day coherent forecasts for all 7 series |
| `memo.md` | Technical memo covering all brief requirements (see coverage checklist inside) |
| `run.py` | Single entrypoint — runs the full pipeline end to end |
| `eda.ipynb` | EDA narrative with inline plots and decision notes |
| `model.ipynb` | Backtest, reconciliation, validation, and impact with inline outputs |
| `requirements.txt` | Pinned dependencies |
| `figures/fig1_*.png` … `fig6_*.png` | Six EDA figures (brief-specified) |
| `figures/fig10_*.png` … `fig12_*.png` | Timeline, adjustment, and impact summary plots |

### Supporting / Appendix (extra rigor, not required by brief)

| File | What it is |
|---|---|
| `wls_comparison.md` | Full WLS vs MinT tradeoff analysis with accuracy tables |
| `verification_report.md` | Root-cause analysis of the 35–57% downward adjustments |
| `backtest_results.csv` | Per-day per-method APE from 747-origin rolling backtest |
| `figures/fig8_*.png` | Residual tail QQ plots (heavy-tail diagnostic) |
| `figures/fig9_*.png` | Rolling 180-day correlation stability plot |

---

## Project Structure

```
data.py          Load and parse data; hierarchy constants, S matrix, display labels
covariance.py    Relative residuals, Schäfer–Strimmer shrinkage, W_day builder
reconcile.py     MinT closed-form + NNLS nonneg QP + coherence checks
eda.py           EDA figures (6 brief-specified + 2 appendix), saved to figures/
backtest.py      Rolling-origin 6-method comparison + in-sample vs OOS W gap
impact.py        Base vs reconciled deltas, who-moved analysis, timeline plots
run.py           Single headless entrypoint orchestrating the full pipeline

eda.ipynb        EDA narrative (imports from modules; inline plots + decision notes)
model.ipynb      Backtest + reconciliation + validation + impact (imports from modules)

memo.md                    Technical memo (methodology, results, required questions)
wls_comparison.md          WLS vs MinT tradeoff analysis [supporting]
verification_report.md     Root-cause analysis of large deltas [supporting]

reconciled_forecasts.csv   351-row output (date + 7 series, exactly coherent)
backtest_results.csv       Per-day per-method APE from rolling-origin backtest
figures/                   All PNG plots
```

---

## Hierarchy

```
Total     = Cohort A + Cohort B
Cohort A  = A1 + A2
Cohort B  = B1 + B2
```

4 leaves, 7 series total. The 7×4 summing matrix S is defined in `data.py`.

---

## Key Design Decisions

| Decision | Evidence |
|---|---|
| Full 7×7 W, not leaves-only | All 7 series carry reliability signal; brief specifies MinT/GLS |
| Relative residuals + per-day W_day | corr(\|error\|, forecast level) 0.62–0.73; 10× volume range in training |
| Schäfer–Strimmer shrinkage | Stabilises W inverse; λ=0.0136 (safety net on this dataset) |
| NNLS QP for nonnegativity | Required guarantee; 0/747 binding days in backtest |
| Disclose base-forecast bias | Under-pred on only 44–49% of days; reconciliation cannot fix this |

---

## Reproducibility

- All randomness: none. Results are deterministic.
- Python ≥ 3.10 recommended. Dependencies pinned in `requirements.txt`.
- Notebooks executed with `jupyter nbconvert --execute --inplace` (fresh kernel).

## Data

`forecast_data_anonymized.csv` — 1463 daily rows, train=1112, future=351.
One 2-day gap in train (flagged on load); does not affect covariance estimation.
