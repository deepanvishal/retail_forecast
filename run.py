"""
Single entrypoint.  Runs end to end:
  EDA -> backtest -> fit C -> reconcile future -> validate -> impact -> write outputs
  -> execute notebooks

Usage:
  python run.py
  python run.py --skip-notebooks   (skips nbconvert; useful in headless environments)
"""
import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from data import SERIES, LEAVES, S, load, pred_array, actual_array
from covariance import (
    relative_residuals, schafer_strimmer, build_W_day,
    condition_report, effective_n_sensitivity
)
from reconcile import nnls_qp, assert_coherent, assert_nonneg, coherence_gaps
import eda
from backtest import run_backtest, insample_vs_oos_W
from impact import (
    compute_impact, analyze_who_moved, who_pulled_whom,
    plot_timeline, plot_deltas, plot_who_moved
)


FIGURES_DIR = 'figures'
OUTPUT_CSV = 'reconciled_forecasts.csv'
BACKTEST_CSV = 'backtest_results.csv'


def banner(msg):
    print(f'\n{"="*60}\n{msg}\n{"="*60}')


def main(skip_notebooks=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    banner('1. Loading data')
    train, future, gap_info = load()
    print(f"Train: {gap_info['n_train']} rows  "
          f"Future: {gap_info['n_future']} rows")
    print(f"Date gap in train: max {gap_info['max_gap_days']} day(s)")
    if gap_info['gap_dates']:
        print(f"  Gap at: {[str(d.date()) for d in gap_info['gap_dates']]}")

    # Verify future preds are strictly positive (required for relative-residual scaling)
    future_preds = pred_array(future)
    min_future_pred = future_preds.min()
    assert min_future_pred > 0, (
        f'Future pred has non-positive value: {min_future_pred:.2f}. '
        'Cannot form relative covariance W_day. Inspect series: '
        f'{[SERIES[i] for i in np.where(future_preds.min(axis=0) <= 0)[0]]}'
    )
    print(f'Min future pred across all 7 series: {min_future_pred:,.0f}  [OK]')

    # ------------------------------------------------------------------
    # 2. EDA — reconciliation-focused diagnostics
    # ------------------------------------------------------------------
    banner('2. Running EDA plots')
    X_rel_full = relative_residuals(train, SERIES)
    C_full, lambda_full = schafer_strimmer(X_rel_full)

    eff_n = effective_n_sensitivity(X_rel_full, lambda_full, len(train))
    print(f'SS lambda (analytic): {lambda_full:.5f}')
    print(f'  Effective n (cross-product lag-1 autocorr = '
          f'{eff_n["rho_cross_product"]:.3f}): {eff_n["n_eff_cross_product"]:.0f}')
    print(f'  Lambda inflated by n/n_eff (cross-product): '
          f'{eff_n["lambda_inflated_cp"]:.5f}')
    print(f'  Effective n (level-residual proxy, rho={eff_n["rho_level_residual"]:.3f}): '
          f'{eff_n["n_eff_level_residual"]:.0f}')
    print(f'  Lambda inflated by n/n_eff (level-residual proxy): '
          f'{eff_n["lambda_inflated_lv"]:.5f}')
    print(f'  Note: {eff_n["note"]}')

    # Sample W_day using median future pred for condition reporting
    med_pred = np.median(future_preds, axis=0)
    W_sample = build_W_day(C_full, med_pred)
    cond = condition_report(C_full, W_sample)
    print(f'\nCondition numbers:')
    print(f'  C (relative covariance): {cond["cond_C"]:.2e}')
    print(f'  W_day (median future pred): {cond["cond_W_day_sample"]:.2e}')

    figs = eda.run_all(train, future, C_full, save_dir=FIGURES_DIR)
    for fig in figs:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f'Saved {len(figs)} figures to {FIGURES_DIR}/')

    # ------------------------------------------------------------------
    # 3. Backtest
    # ------------------------------------------------------------------
    banner('3. Running rolling-origin backtest (min_window=365)')
    results_df, summary_df, meta = run_backtest(train)
    print(f"Backtest: {meta['n_iter']} iterations")
    print(f"SS lambda: mean={meta['lambda_mean']:.5f}  "
          f"min={meta['lambda_min']:.5f}  max={meta['lambda_max']:.5f}")
    print(f"Neg-leaf days (unconstrained mint_scaled): {meta['neg_leaf_count']}/{meta['n_iter']}")
    print()
    print(summary_df[['overall_mape']].rename(
        columns={'overall_mape': 'MAPE % (7-series mean)'}).round(4).to_string())

    results_df.to_csv(BACKTEST_CSV, index=False)
    print(f'\nPer-day backtest results saved to {BACKTEST_CSV}')

    # In-sample vs OOS W comparison
    print('\nIn-sample vs OOS W comparison (computing ~747 iterations with insample C)...')
    oos_vs_ins = insample_vs_oos_W(train)
    print(f"  OOS W MAPE:       {oos_vs_ins['mape_oos_pct']:.4f}%")
    print(f"  In-sample W MAPE: {oos_vs_ins['mape_insample_pct']:.4f}%")
    print(f"  Optimism gap:     {oos_vs_ins['optimism_gap_pct']:.4f}%  "
          f"({'in-sample better' if oos_vs_ins['optimism_gap_pct'] > 0 else 'OOS better'})")

    # ------------------------------------------------------------------
    # 4. Fit C on ALL train data, reconcile all 351 future days
    # ------------------------------------------------------------------
    banner('4. Reconciling 351 future days')
    n_future = len(future)
    reconciled_arr = np.zeros((n_future, len(SERIES)))
    qp_binding_days = 0

    for t in range(n_future):
        pred_vec = future_preds[t, :]
        W_day = build_W_day(C_full, pred_vec)
        ytilde, b, qp_active = nnls_qp(W_day, S, pred_vec)
        reconciled_arr[t, :] = ytilde
        if qp_active:
            qp_binding_days += 1

    print(f'QP binding days: {qp_binding_days}/{n_future}')

    # ------------------------------------------------------------------
    # 5. Validation
    # ------------------------------------------------------------------
    banner('5. Validation')
    max_gap = 0.0
    for t in range(n_future):
        ytilde = reconciled_arr[t, :]
        assert_coherent(ytilde, tol=1e-6)
        gaps = coherence_gaps(ytilde)
        max_gap = max(max_gap, max(gaps.values()))

    all_leaves = reconciled_arr[:, 3:]
    min_leaf = float(all_leaves.min())
    assert np.all(all_leaves >= 0), f'Negative leaf found: {min_leaf:.4f}'

    assert reconciled_arr.shape[0] == 351, \
        f'Expected 351 future rows, got {reconciled_arr.shape[0]}'
    assert reconciled_arr.shape[1] == len(SERIES), \
        f'Expected {len(SERIES)} series columns'
    assert not np.any(np.isnan(reconciled_arr)), 'NaN in reconciled output'

    print(f'Coherence: max abs identity violation = {max_gap:.2e}  [< 1e-6 PASSED]')
    print(f'Nonnegativity: min leaf = {min_leaf:,.2f}  [>= 0 PASSED]')
    print(f'Shape: {reconciled_arr.shape[0]} rows x {reconciled_arr.shape[1]} series  [OK]')
    print(f'NaN check: PASSED')

    # ------------------------------------------------------------------
    # 6. Impact analysis
    # ------------------------------------------------------------------
    banner('6. Impact analysis')
    impact_df = compute_impact(future, reconciled_arr)
    who_moved_df = analyze_who_moved(impact_df, C_full)
    level_moves = who_pulled_whom(impact_df)

    # Future incoherence structure — key context for impact interpretation
    future_preds_raw = pred_array(future)
    gaps_agg_pct = (np.abs(future_preds_raw[:,0] - future_preds_raw[:,1] - future_preds_raw[:,2])
                    / future_preds_raw[:,0] * 100)
    n_over = int((future_preds_raw[:,1] + future_preds_raw[:,2] > future_preds_raw[:,0]).sum())
    print(f'\nFuture base incoherence (agg gap as % of agg pred):')
    print(f'  Days where cohortA+B > aggregate: {n_over}/351')
    print(f'  Median gap %: {np.median(gaps_agg_pct):.1f}%  Mean: {gaps_agg_pct.mean():.1f}%')
    print(f'  -> Mechanism: high positive C correlations -> negative W^-1 off-diagonals')
    print(f'     -> subtractive GLS signal for upper levels -> reconciled agg BELOW base agg')
    print(f'     -> this is calibration-range extrapolation (train max gap was ~2%; future median {np.median(gaps_agg_pct):.1f}%)')

    print('\nMean absolute delta % per series (sorted by movement):')
    print(who_moved_df[['label', 'relative_variance', 'rel_var_rank',
                          'mean_abs_delta_pct', 'mean_delta_pct']].round(4).to_string())

    print('\nMean |delta %| by hierarchy level:')
    for k, v in level_moves.items():
        print(f'  {k}: {v:.4f}%')

    import matplotlib.pyplot as plt
    for fig in [
        plot_timeline(train, future, reconciled_arr, save_dir=FIGURES_DIR),
        plot_deltas(impact_df, save_dir=FIGURES_DIR),
        plot_who_moved(who_moved_df, save_dir=FIGURES_DIR),
    ]:
        plt.close(fig)
    print(f'Impact figures saved to {FIGURES_DIR}/')

    # ------------------------------------------------------------------
    # 7. Write reconciled_forecasts.csv
    # ------------------------------------------------------------------
    banner('7. Writing output CSV')
    out = pd.DataFrame({'date': future['date']})
    for i, s in enumerate(SERIES):
        out[s] = reconciled_arr[:, i]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f'Wrote {OUTPUT_CSV}: {out.shape[0]} rows x {out.shape[1]} cols')
    print(f'Columns: {out.columns.tolist()}')

    # ------------------------------------------------------------------
    # 8. Execute notebooks
    # ------------------------------------------------------------------
    if not skip_notebooks:
        banner('8. Executing notebooks (restart-and-run-all)')
        for nb in ['eda.ipynb', 'model.ipynb']:
            if not os.path.exists(nb):
                print(f'  {nb} not found — skipping')
                continue
            cmd = [
                sys.executable, '-m', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=300',
                nb,
            ]
            print(f'  Executing {nb} ...')
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f'  {nb}: OK')
            else:
                print(f'  {nb}: FAILED')
                print(result.stderr[-2000:])
    else:
        print('\n[Skipping notebook execution (--skip-notebooks)]')

    banner('Done')
    print(f'Outputs: {OUTPUT_CSV}, {BACKTEST_CSV}, {FIGURES_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-notebooks', action='store_true',
                        help='Skip nbconvert notebook execution')
    args = parser.parse_args()
    main(skip_notebooks=args.skip_notebooks)
