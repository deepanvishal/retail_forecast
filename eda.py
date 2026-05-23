import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from data import SERIES, LEAVES, S, pred_array, actual_array
from covariance import relative_residuals, schafer_strimmer


_COHORT_LABELS = {
    'aggregate': 'Aggregate',
    'cohort_A': 'Cohort A',
    'cohort_B': 'Cohort B',
    'A1': 'A1',
    'A2': 'A2',
    'B1': 'B1',
    'B2': 'B2',
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')


def _acf(x, nlags=40):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c0 = np.dot(x, x) / len(x)
    return np.array([
        np.dot(x[:len(x) - k], x[k:]) / (len(x) * c0) for k in range(nlags + 1)
    ])


# ---------------------------------------------------------------------------
# Fig 1: Base-pred incoherence over time
# Motivates the reconciliation task: actuals are perfectly coherent; base preds
# are badly incoherent, and the gap grows with volume in the future period.
# ---------------------------------------------------------------------------
def plot_incoherence(train, future, save_dir='figures'):
    _ensure_dir(save_dir)

    dates_tr = train['date'].values
    dates_fu = future['date'].values

    identities = [
        ('aggregate = cohort_A + cohort_B',
         train['aggregate_pred'] - train['cohort_A_pred'] - train['cohort_B_pred'],
         future['aggregate_pred'] - future['cohort_A_pred'] - future['cohort_B_pred']),
        ('cohort_A = A1 + A2',
         train['cohort_A_pred'] - train['A1_pred'] - train['A2_pred'],
         future['cohort_A_pred'] - future['A1_pred'] - future['A2_pred']),
        ('cohort_B = B1 + B2',
         train['cohort_B_pred'] - train['B1_pred'] - train['B2_pred'],
         future['cohort_B_pred'] - future['B1_pred'] - future['B2_pred']),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    fig.suptitle('Base Prediction Incoherence (identity gap = pred_parent - sum(pred_children))',
                 fontsize=11)

    for ax, (label, gap_tr, gap_fu) in zip(axes, identities):
        ax.plot(dates_tr, gap_tr.abs(), color='steelblue', lw=0.7, alpha=0.8,
                label=f'Train  mean={gap_tr.abs().mean():.0f}  max={gap_tr.abs().max():.0f}')
        ax.plot(dates_fu, gap_fu.abs(), color='darkorange', lw=0.7, alpha=0.8,
                label=f'Future mean={gap_fu.abs().mean():.0f}  max={gap_fu.abs().max():.0f}')
        ax.axvline(pd.Timestamp('2026-01-18'), color='black', ls='--', lw=0.9, alpha=0.6)
        ax.set_ylabel('|gap|')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v/1e6:.1f}M' if v >= 1e6 else f'{v/1e3:.0f}k'))

    axes[-1].set_xlabel('Date')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig1_incoherence.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Heteroscedasticity
# Three panels:
#   A. |resid| vs level scatter (aggregate, both level=pred and level=actual shown)
#   B. Residual std by volume quintile (aggregate)
#   C. Spread comparison: raw vs relative residuals per series
# Defends: scaled residuals + per-day W_day rescaling.
# ---------------------------------------------------------------------------
def plot_heteroscedasticity(train, save_dir='figures'):
    _ensure_dir(save_dir)

    res_raw = actual_array(train) - pred_array(train)
    pred_mat = pred_array(train)
    actual_mat = actual_array(train)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: scatter |resid_agg| vs level
    ax_a = fig.add_subplot(gs[0])
    agg_res = np.abs(res_raw[:, 0])
    agg_pred = pred_mat[:, 0]
    agg_actual = actual_mat[:, 0]

    ax_a.scatter(agg_pred / 1e6, agg_res / 1e6, s=6, alpha=0.35,
                 color='steelblue', label='level = pred')
    ax_a.scatter(agg_actual / 1e6, agg_res / 1e6, s=6, alpha=0.25,
                 color='darkorange', label='level = actual')
    # OLS trend lines
    for xv, col in [(agg_pred, 'steelblue'), (agg_actual, 'darkorange')]:
        m, b_int = np.polyfit(xv / 1e6, agg_res / 1e6, 1)
        xl = np.linspace(xv.min(), xv.max(), 200) / 1e6
        ax_a.plot(xl, m * xl + b_int, color=col, lw=1.2, alpha=0.8)

    # Correlation annotations (operationally: pred is relevant; actual shown for context)
    r_pred = np.corrcoef(agg_res, agg_pred)[0, 1]
    r_actual = np.corrcoef(agg_res, agg_actual)[0, 1]
    ax_a.set_xlabel('Level (M)')
    ax_a.set_ylabel('|Residual| (M)')
    ax_a.set_title(
        f'|resid| vs level — aggregate\n'
        f'corr(level=pred)={r_pred:.3f}   corr(level=actual)={r_actual:.3f}\n'
        '(pred is operationally relevant; actual shown for comparison)',
        fontsize=8
    )
    ax_a.legend(fontsize=8)

    # Panel B: residual std by aggregate volume quintile
    ax_b = fig.add_subplot(gs[1])
    df_tmp = pd.DataFrame({
        'agg_pred': agg_pred,
        'agg_res': res_raw[:, 0],
    })
    df_tmp['quintile'] = pd.qcut(df_tmp['agg_pred'], 5, labels=False)
    q_stats = df_tmp.groupby('quintile').agg(
        std=('agg_res', 'std'),
        mean_pred=('agg_pred', 'mean'),
        n=('agg_res', 'count'),
    ).reset_index()
    q_stats['pct'] = q_stats['std'] / q_stats['mean_pred'] * 100

    bars = ax_b.bar(range(5), q_stats['std'] / 1e3, color='steelblue', alpha=0.75)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(range(5), q_stats['pct'], 'o-', color='crimson', lw=1.5, ms=5,
               label='% of level (right)')
    ax_b.set_xticks(range(5))
    ax_b.set_xticklabels([f'Q{i+1}\n{v/1e6:.1f}M' for i, v in
                          enumerate(q_stats['mean_pred'])], fontsize=7)
    ax_b.set_xlabel('Volume quintile (mean pred)')
    ax_b.set_ylabel('Residual std (k)')
    ax_b2.set_ylabel('Std as % of level', color='crimson')
    ax_b2.tick_params(axis='y', colors='crimson')
    ax_b.set_title('Residual std by volume quintile\n(aggregate)', fontsize=9)
    ax_b2.legend(fontsize=8, loc='upper left')

    # Panel C: raw vs relative residual spread per series
    ax_c = fig.add_subplot(gs[2])
    rel_res = res_raw / pred_mat

    positions_raw = np.arange(len(SERIES)) * 2.5
    positions_rel = positions_raw + 0.9

    bp_raw = ax_c.boxplot(
        [res_raw[:, i] / 1e3 for i in range(len(SERIES))],
        positions=positions_raw, widths=0.7,
        patch_artist=True, medianprops=dict(color='black', lw=1.5),
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        flierprops=dict(marker='.', ms=2, alpha=0.3),
        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )
    # Relative residuals on twin axis
    ax_c2 = ax_c.twinx()
    bp_rel = ax_c2.boxplot(
        [rel_res[:, i] for i in range(len(SERIES))],
        positions=positions_rel, widths=0.7,
        patch_artist=True, medianprops=dict(color='black', lw=1.5),
        boxprops=dict(facecolor='darkorange', alpha=0.6),
        flierprops=dict(marker='.', ms=2, alpha=0.3),
        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )

    ax_c.set_xticks(positions_raw + 0.45)
    ax_c.set_xticklabels([_COHORT_LABELS[s] for s in SERIES], rotation=30,
                          ha='right', fontsize=7)
    ax_c.set_ylabel('Raw residual (k)', color='steelblue')
    ax_c.tick_params(axis='y', colors='steelblue')
    ax_c2.set_ylabel('Relative residual', color='darkorange')
    ax_c2.tick_params(axis='y', colors='darkorange')
    ax_c.set_title('Raw (blue, k) vs relative (orange)\nresidual spread per series', fontsize=9)
    from matplotlib.patches import Patch
    ax_c.legend(handles=[Patch(color='steelblue', alpha=0.7, label='Raw (k)'),
                          Patch(color='darkorange', alpha=0.7, label='Relative')],
                fontsize=8, loc='upper left')

    fig.suptitle('Heteroscedasticity: residual variance grows with scale', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig2_heteroscedasticity.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Volume trend + monthly seasonality
# Documents the ~10x growth (explains why a single absolute W would be dominated
# by high-volume periods) and the November spike pattern.
# ---------------------------------------------------------------------------
def plot_volume_trend(train, save_dir='figures'):
    _ensure_dir(save_dir)

    train = train.copy()
    train['ym'] = train['date'].dt.to_period('M')
    monthly = train.groupby('ym')['aggregate_actual'].mean().reset_index()
    monthly['date'] = monthly['ym'].dt.to_timestamp()
    monthly['month'] = monthly['date'].dt.month
    monthly['is_nov'] = monthly['month'] == 11
    monthly['is_apr'] = monthly['month'] == 4

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel A: monthly mean over time
    ax = axes[0]
    colors = np.where(monthly['is_nov'], 'crimson',
                      np.where(monthly['is_apr'], 'darkorange', 'steelblue'))
    ax.bar(monthly['date'], monthly['aggregate_actual'] / 1e6,
           width=25, color=colors, alpha=0.8, edgecolor='none')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='crimson', alpha=0.8, label='November'),
        Patch(color='darkorange', alpha=0.8, label='April'),
        Patch(color='steelblue', alpha=0.8, label='Other'),
    ], fontsize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean daily aggregate actual (M)')
    ax.set_title('Monthly mean aggregate — ~10x growth over 3 years\n'
                 'November spike every year; elevated April', fontsize=9)

    # Annotate November values
    nov_rows = monthly[monthly['is_nov']]
    for _, row in nov_rows.iterrows():
        ax.annotate(f"{row['aggregate_actual']/1e6:.1f}M",
                    xy=(row['date'], row['aggregate_actual'] / 1e6),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=7, color='crimson')

    # Panel B: average by calendar month (seasonality profile)
    ax2 = axes[1]
    seasonal = train.copy()
    seasonal['month'] = train['date'].dt.month
    by_month = seasonal.groupby('month')['aggregate_actual'].mean()
    bar_colors = ['crimson' if m == 11 else ('darkorange' if m == 4 else 'steelblue')
                  for m in by_month.index]
    ax2.bar(by_month.index, by_month.values / 1e6, color=bar_colors, alpha=0.8)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.set_xlabel('Calendar month')
    ax2.set_ylabel('Mean daily actual (M)')
    ax2.set_title('Average seasonality profile\n(Nov peak drives W estimation sensitivity)', fontsize=9)

    fig.suptitle('Volume Trend and Seasonality (train actuals)', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig3_volume_trend.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Bias
# Mean residual is positive (under-predicting on average) for all 7 series,
# but the base model over-predicts on >50% of days.  Bias comes from a small
# number of large under-shoots on high-volume days.  Reconciliation distributes
# within a day only and cannot fix this bias.
# ---------------------------------------------------------------------------
def plot_bias(train, save_dir='figures'):
    _ensure_dir(save_dir)

    res_raw = actual_array(train) - pred_array(train)
    pred_mat = pred_array(train)

    mean_res = res_raw.mean(axis=0)
    mean_pred = pred_mat.mean(axis=0)
    mean_pct = mean_res / mean_pred * 100
    pct_underpred = (res_raw > 0).mean(axis=0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [_COHORT_LABELS[s] for s in SERIES]
    x = np.arange(len(SERIES))
    w = 0.6

    # Panel A: mean residual (absolute)
    ax = axes[0]
    ax.bar(x, mean_res / 1e3, width=w, color='steelblue', alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mean residual (k = actual - pred)')
    ax.set_title('Mean residual > 0 for all series\n(under-predicting on average)', fontsize=9)

    # Panel B: mean residual as % of level
    ax2 = axes[1]
    ax2.bar(x, mean_pct, width=w, color='darkorange', alpha=0.8)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Mean residual as % of mean pred level')
    ax2.set_title('Bias: 1.1%–4.1% of level\n(reconciliation cannot fix this)', fontsize=9)
    for i, v in enumerate(mean_pct):
        ax2.text(i, v + 0.05, f'{v:.1f}%', ha='center', fontsize=7)

    # Panel C: frequency vs magnitude — aggregate series
    ax3 = axes[2]
    agg_res = res_raw[:, 0]
    pct_up = (agg_res > 0).mean() * 100
    pct_dn = 100 - pct_up
    mean_under = agg_res[agg_res > 0].mean()
    mean_over = agg_res[agg_res <= 0].mean()

    ax3_r = ax3.twinx()
    ax3.bar(['Under-pred\n(actual>pred)', 'Over-pred\n(actual<pred)'],
            [pct_up, pct_dn], color=['steelblue', 'darkorange'], alpha=0.75, width=0.5)
    ax3_r.bar(['Under-pred\n(actual>pred)', 'Over-pred\n(actual<pred)'],
              [mean_under / 1e3, abs(mean_over) / 1e3],
              color=['steelblue', 'darkorange'], alpha=0.35, width=0.3,
              label='Mean |resid| (k)')

    ax3.set_ylabel('% of train days', color='black')
    ax3_r.set_ylabel('Mean |residual| (k)', color='grey')
    ax3.set_title('Aggregate: under-pred on only ~45% of days\nbut mean |under-shoot| >> mean |over-shoot|',
                  fontsize=9)
    ax3.text(0, pct_up + 0.5, f'{pct_up:.1f}%', ha='center', fontsize=8)
    ax3.text(1, pct_dn + 0.5, f'{pct_dn:.1f}%', ha='center', fontsize=8)

    fig.suptitle('Forecast Bias — disclose, do not attempt to fix via reconciliation', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig4_bias.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Residual autocorrelation
# Lag-1 ACF 0.34–0.56 and max same-sign streaks up to 23 days show errors are
# not iid in time.  This motivates out-of-sample (rolling-origin) W estimation
# and is a known limitation of the analytic SS lambda formula (derived under IID).
# ---------------------------------------------------------------------------
def plot_acf(train, save_dir='figures'):
    _ensure_dir(save_dir)

    res_raw = actual_array(train) - pred_array(train)
    nlags = 40
    conf = 1.96 / np.sqrt(len(train))
    n_series = len(SERIES)
    ncols = 4
    nrows = (n_series + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(SERIES):
        ax = axes[i]
        acf_vals = _acf(res_raw[:, i], nlags=nlags)
        lags = np.arange(nlags + 1)
        ax.bar(lags[1:], acf_vals[1:], color='steelblue', alpha=0.75, width=0.8)
        ax.axhline(conf, color='crimson', ls='--', lw=0.9)
        ax.axhline(-conf, color='crimson', ls='--', lw=0.9)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'{_COHORT_LABELS[s]}  lag1={acf_vals[1]:.3f}', fontsize=9)
        ax.set_xlabel('Lag (days)', fontsize=8)
        ax.set_ylim(-0.35, 0.75)

        # Annotate max same-sign streak
        signs = np.sign(res_raw[:, i])
        streak = max_streak = 1
        for k in range(1, len(signs)):
            streak = streak + 1 if signs[k] == signs[k - 1] else 1
            max_streak = max(max_streak, streak)
        ax.text(0.97, 0.92, f'max streak={max_streak}d', transform=ax.transAxes,
                ha='right', fontsize=7, color='darkred')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('ACF of Raw Residuals per Series  (red dashed = 95% IID bounds)\n'
                 'Lag-1 autocorr 0.34–0.56; errors are not iid — motivates OOS W estimation',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig5_acf.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Estimated W — correlation structure and reliability ranking
# The relative covariance C drives who-pulls-whom: high relative variance
# (diagonal) = less reliable = moved more by reconciliation.
# Off-diagonal correlations determine the direction of adjustment.
# ---------------------------------------------------------------------------
def plot_W_structure(C, series, save_dir='figures'):
    _ensure_dir(save_dir)

    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    labels = [_COHORT_LABELS[s] for s in series]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(series)))
    ax.set_yticks(range(len(series)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(series)):
        for j in range(len(series)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(corr[i, j]) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Relative-residual correlation matrix C\n'
                 '(estimated from all train data, SS-shrunk)', fontsize=9)

    # Panel B: per-series relative variance (= reliability ranking)
    ax2 = axes[1]
    rel_var = np.diag(C)
    order = np.argsort(rel_var)[::-1]
    bar_labels = [labels[i] for i in order]
    bar_vals = rel_var[order]
    colors = ['crimson' if rel_var[i] == rel_var.max() else
              ('darkorange' if rel_var[i] > np.median(rel_var) else 'steelblue')
              for i in order]
    ax2.barh(range(len(series)), bar_vals, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(series)))
    ax2.set_yticklabels(bar_labels, fontsize=9)
    ax2.set_xlabel('Relative variance (diagonal of C)')
    ax2.set_title('Reliability ranking\nhigher = less reliable = moved more', fontsize=9)
    for i, v in enumerate(bar_vals):
        ax2.text(v + v * 0.01, i, f'{v:.4f}', va='center', fontsize=8)

    fig.suptitle('Estimated Relative Covariance C: Correlation Structure and Reliability',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig6_W_structure.png'))
    return fig


# ---------------------------------------------------------------------------
# Additional Fig 7: Sibling and cross-cohort correlation structure
# Directly drives the direction of reconciliation adjustments:
# high sibling correlation = when one is pulled, the other follows.
# ---------------------------------------------------------------------------
def plot_sibling_correlations(C, series, train, save_dir='figures'):
    _ensure_dir(save_dir)

    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    labels = [_COHORT_LABELS[s] for s in series]

    idx = {s: i for i, s in enumerate(series)}

    pairs = [
        ('A1', 'A2', 'Siblings within cohort_A'),
        ('B1', 'B2', 'Siblings within cohort_B'),
        ('cohort_A', 'cohort_B', 'Across cohorts'),
        ('A1', 'B1', 'Cross-cohort: A1 vs B1'),
        ('A2', 'B2', 'Cross-cohort: A2 vs B2'),
        ('aggregate', 'cohort_A', 'Aggregate vs cohort_A'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    rel_res = relative_residuals(train, series)
    for ax, (s1, s2, title) in zip(axes, pairs):
        i1, i2 = idx[s1], idx[s2]
        ax.scatter(rel_res[:, i1], rel_res[:, i2], s=5, alpha=0.3, color='steelblue')
        c = corr[i1, i2]
        m, b_int = np.polyfit(rel_res[:, i1], rel_res[:, i2], 1)
        xl = np.linspace(rel_res[:, i1].min(), rel_res[:, i1].max(), 100)
        ax.plot(xl, m * xl + b_int, color='crimson', lw=1.5)
        ax.set_xlabel(_COHORT_LABELS[s1], fontsize=8)
        ax.set_ylabel(_COHORT_LABELS[s2], fontsize=8)
        ax.set_title(f'{title}\ncorr = {c:.3f}', fontsize=9)

    fig.suptitle('Relative-Residual Cross-Correlations\n'
                 '(drives direction of reconciliation: who pulls whom)', fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig7_sibling_correlations.png'))
    return fig


# ---------------------------------------------------------------------------
# Additional Fig 8: Residual non-Gaussianity
# Heavy tails in the residual distribution imply that covariance-based MinT
# (which implicitly assumes Gaussian errors for full optimality) is a reasonable
# approximation but may underweight tail events.
# ---------------------------------------------------------------------------
def plot_residual_tails(train, save_dir='figures'):
    _ensure_dir(save_dir)

    rel_res = relative_residuals(train, SERIES)
    ncols = 4
    nrows = (len(SERIES) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(SERIES):
        ax = axes[i]
        x = rel_res[:, i]
        kurt = float(pd.Series(x).kurtosis())
        skew = float(pd.Series(x).skew())

        # QQ plot against normal
        (osm, osr), (slope, intercept, r) = stats.probplot(x, dist='norm')
        ax.scatter(osm, osr, s=5, alpha=0.4, color='steelblue')
        ql = np.linspace(min(osm), max(osm), 100)
        ax.plot(ql, slope * ql + intercept, color='crimson', lw=1.2)
        ax.set_title(f'{_COHORT_LABELS[s]}\nexcess_kurt={kurt:.1f}  skew={skew:.2f}', fontsize=8)
        ax.set_xlabel('Theoretical quantile', fontsize=7)
        ax.set_ylabel('Sample quantile', fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('QQ Plots of Relative Residuals vs Normal\n'
                 'Heavy tails visible — covariance-based MinT is approximately correct '
                 'but may underweight extreme days', fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig8_residual_tails.png'))
    return fig


# ---------------------------------------------------------------------------
# Additional Fig 9: Rolling correlation stability
# If the relative-covariance correlation structure is stable over time, using
# the full-train C for future reconciliation is justified.  If it drifts, a
# recent-window estimator would be preferable.
# ---------------------------------------------------------------------------
def plot_rolling_stability(train, save_dir='figures'):
    _ensure_dir(save_dir)

    rel_res = relative_residuals(train, SERIES)
    dates = train['date'].values

    # Track 6 correlation pairs over a 180-day rolling window
    pairs = [
        (SERIES.index('A1'), SERIES.index('A2'), 'A1-A2 (siblings)'),
        (SERIES.index('B1'), SERIES.index('B2'), 'B1-B2 (siblings)'),
        (SERIES.index('cohort_A'), SERIES.index('cohort_B'), 'cohortA-cohortB'),
        (SERIES.index('aggregate'), SERIES.index('cohort_A'), 'agg-cohortA'),
    ]

    window = 180
    roll_dates = []
    roll_corrs = {label: [] for _, _, label in pairs}

    for end in range(window, len(rel_res)):
        window_data = rel_res[end - window:end, :]
        roll_dates.append(dates[end])
        for i1, i2, label in pairs:
            r = np.corrcoef(window_data[:, i1], window_data[:, i2])[0, 1]
            roll_corrs[label].append(r)

    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ['steelblue', 'darkorange', 'green', 'crimson']
    for (_, _, label), col in zip(pairs, colors):
        ax.plot(roll_dates, roll_corrs[label], lw=1.2, label=label, color=col, alpha=0.85)

    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Date (end of rolling window)')
    ax.set_ylabel('Rolling correlation (180-day window)')
    ax.set_title('Rolling Correlation Stability\n'
                 'Broadly stable = full-train C is reasonable for future reconciliation', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig9_rolling_stability.png'))
    return fig


def run_all(train, future, C, save_dir='figures'):
    """Run all EDA plots and return list of figures."""
    figs = []
    figs.append(plot_incoherence(train, future, save_dir))
    figs.append(plot_heteroscedasticity(train, save_dir))
    figs.append(plot_volume_trend(train, save_dir))
    figs.append(plot_bias(train, save_dir))
    figs.append(plot_acf(train, save_dir))
    figs.append(plot_W_structure(C, SERIES, save_dir))
    figs.append(plot_sibling_correlations(C, SERIES, train, save_dir))
    figs.append(plot_residual_tails(train, save_dir))
    figs.append(plot_rolling_stability(train, save_dir))
    return figs
