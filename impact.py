import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from data import SERIES, actual_array, pred_array


_COHORT_LABELS = {
    'aggregate': 'Aggregate',
    'cohort_A': 'Cohort A',
    'cohort_B': 'Cohort B',
    'A1': 'A1',
    'A2': 'A2',
    'B1': 'B1',
    'B2': 'B2',
}

_HIER_LEVEL = {
    'aggregate': 0,
    'cohort_A': 1,
    'cohort_B': 1,
    'A1': 2,
    'A2': 2,
    'B1': 2,
    'B2': 2,
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')


def compute_impact(future, reconciled_arr, series=None):
    """
    Returns a wide-format DataFrame indexed by date with columns
    {s}_base, {s}_reconciled, {s}_delta_abs, {s}_delta_pct for each series.
    """
    if series is None:
        series = SERIES

    base_arr = pred_array(future)
    df = pd.DataFrame({'date': future['date'].values})
    for i, s in enumerate(series):
        df[f'{s}_base'] = base_arr[:, i]
        df[f'{s}_reconciled'] = reconciled_arr[:, i]
        df[f'{s}_delta_abs'] = reconciled_arr[:, i] - base_arr[:, i]
        df[f'{s}_delta_pct'] = (reconciled_arr[:, i] - base_arr[:, i]) / base_arr[:, i] * 100
    return df


def analyze_who_moved(impact_df, C, series=None):
    """
    Connects the magnitude and direction of reconciliation adjustments to
    the reliability structure in C.

    Returns a summary DataFrame with mean |delta_pct| per series, relative
    variance (unreliability) rank, and hierarchy level.
    """
    if series is None:
        series = SERIES

    rel_var = np.diag(C)
    rows = []
    for i, s in enumerate(series):
        mean_delta_pct = impact_df[f'{s}_delta_pct'].mean()
        mean_abs_delta_pct = impact_df[f'{s}_delta_pct'].abs().mean()
        rows.append({
            'series': s,
            'label': _COHORT_LABELS[s],
            'hierarchy_level': _HIER_LEVEL[s],
            'relative_variance': rel_var[i],
            'mean_delta_pct': mean_delta_pct,
            'mean_abs_delta_pct': mean_abs_delta_pct,
        })
    df = pd.DataFrame(rows).set_index('series')
    df['rel_var_rank'] = df['relative_variance'].rank(ascending=False).astype(int)
    return df.sort_values('mean_abs_delta_pct', ascending=False)


def who_pulled_whom(impact_df, series=None):
    """
    Determines whether the aggregate or the leaves drove the reconciliation
    direction by comparing normalised mean |delta_pct| at each hierarchy level.

    If leaves moved more (relative to their own scale) than the aggregate, the
    leaves were pulling.  If the aggregate moved more, the aggregate was pulling.
    """
    if series is None:
        series = SERIES

    level_groups = {0: ['aggregate'], 1: ['cohort_A', 'cohort_B'],
                    2: ['A1', 'A2', 'B1', 'B2']}
    results = {}
    for lvl, members in level_groups.items():
        vals = [impact_df[f'{s}_delta_pct'].abs().mean() for s in members]
        results[f'level_{lvl}_mean_abs_delta_pct'] = float(np.mean(vals))
    return results


# ---------------------------------------------------------------------------
# Fig 10: Timeline — actuals (train) + base (future) + reconciled (future)
# ---------------------------------------------------------------------------
def plot_timeline(train, future, reconciled_arr, series=None, save_dir='figures'):
    _ensure_dir(save_dir)
    if series is None:
        series = SERIES

    base_arr = pred_array(future)
    actual_tr = actual_array(train)
    ncols = 4
    nrows = (len(series) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()

    boundary = future['date'].iloc[0]

    for i, s in enumerate(series):
        ax = axes[i]
        ax.plot(train['date'], actual_tr[:, i] / 1e6,
                color='steelblue', lw=0.8, alpha=0.7, label='Actual (train)')
        ax.plot(future['date'], base_arr[:, i] / 1e6,
                color='darkorange', lw=1.0, ls='--', alpha=0.85, label='Base pred (future)')
        ax.plot(future['date'], reconciled_arr[:, i] / 1e6,
                color='crimson', lw=1.2, alpha=0.9, label='Reconciled (future)')
        ax.axvline(boundary, color='black', ls=':', lw=0.8)
        ax.set_title(_COHORT_LABELS[s], fontsize=9)
        ax.set_ylabel('M', fontsize=7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v:.1f}'))
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Actuals (train) + Base Predictions (future, dashed) + '
                 'Reconciled (future, solid red)', fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig10_timeline.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 11: Per-series delta (reconciled - base) over future period
# ---------------------------------------------------------------------------
def plot_deltas(impact_df, series=None, save_dir='figures'):
    _ensure_dir(save_dir)
    if series is None:
        series = SERIES

    ncols = 4
    nrows = (len(series) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(series):
        ax = axes[i]
        delta = impact_df[f'{s}_delta_pct']
        colors = np.where(delta >= 0, 'steelblue', 'crimson')
        ax.bar(range(len(delta)), delta, color=colors, alpha=0.7, width=1.0)
        ax.axhline(0, color='black', lw=0.6)
        mean_d = delta.mean()
        ax.axhline(mean_d, color='darkgreen', ls='--', lw=1.0,
                   label=f'mean={mean_d:.2f}%')
        ax.set_title(_COHORT_LABELS[s], fontsize=9)
        ax.set_ylabel('Delta %', fontsize=7)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Reconciliation Delta per Series: (reconciled - base) / base × 100\n'
                 'Blue = upward adjustment, Red = downward', fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig11_deltas.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 12: Who-moved summary — mean |delta_pct| vs relative variance
# ---------------------------------------------------------------------------
def plot_who_moved(who_moved_df, save_dir='figures'):
    _ensure_dir(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df = who_moved_df.reset_index()

    # Panel A: mean |delta_pct| per series (sorted)
    ax = axes[0]
    ax.barh(range(len(df)), df['mean_abs_delta_pct'],
            color=['crimson' if lvl == 0 else ('darkorange' if lvl == 1 else 'steelblue')
                   for lvl in df['hierarchy_level']],
            alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Mean |delta| %')
    ax.set_title('Mean absolute reconciliation adjustment\nby series', fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='crimson', alpha=0.8, label='Aggregate'),
        Patch(color='darkorange', alpha=0.8, label='Cohort level'),
        Patch(color='steelblue', alpha=0.8, label='Leaves'),
    ], fontsize=8, loc='lower right')

    # Panel B: scatter mean |delta_pct| vs relative variance
    ax2 = axes[1]
    sc = ax2.scatter(df['relative_variance'], df['mean_abs_delta_pct'],
                     c=df['hierarchy_level'], cmap='Set1', s=80, zorder=3)
    for _, row in df.iterrows():
        ax2.annotate(row['label'],
                     xy=(row['relative_variance'], row['mean_abs_delta_pct']),
                     xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Relative variance (diagonal of C) — higher = less reliable')
    ax2.set_ylabel('Mean |delta| %')
    ax2.set_title('Less reliable series move more\n(relative variance drives adjustment size)',
                  fontsize=9)

    fig.suptitle('Impact Analysis: Who Moved and Why', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig12_who_moved.png'))
    return fig
