import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from data import SERIES, actual_array, pred_array, DISPLAY_LABELS, SHORT_LABELS

_HIER_LEVEL = {
    'aggregate': 0,
    'cohort_A':  1,
    'cohort_B':  1,
    'A1': 2, 'A2': 2,
    'B1': 2, 'B2': 2,
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')


def compute_impact(future, reconciled_arr, series=None):
    """
    Compute the per-day, per-series reconciliation adjustments.

    Returns a wide-format DataFrame with base forecast, reconciled forecast,
    and two adjustment columns (absolute and % of base) for each series.

    Args:
        future (pd.DataFrame): future data with {s}_pred columns.
        reconciled_arr (np.ndarray, shape (n, 7)): reconciled forecasts in
            SERIES order.
        series (list[str], optional): series keys. Defaults to SERIES.
    Returns:
        pd.DataFrame: columns are date, then for each series:
            {s}_base, {s}_reconciled, {s}_delta_abs, {s}_delta_pct.
            delta_pct = (reconciled − base) / base × 100.
    """
    if series is None:
        series = SERIES

    base_arr = pred_array(future)
    df = pd.DataFrame({'date': future['date'].values})
    for i, s in enumerate(series):
        df[f'{s}_base']        = base_arr[:, i]
        df[f'{s}_reconciled']  = reconciled_arr[:, i]
        df[f'{s}_delta_abs']   = reconciled_arr[:, i] - base_arr[:, i]
        df[f'{s}_delta_pct']   = (reconciled_arr[:, i] - base_arr[:, i]) / base_arr[:, i] * 100
    return df


def analyze_who_moved(impact_df, C, series=None):
    """
    Summarise how much each series was adjusted and why.

    Connects the magnitude and direction of reconciliation adjustments to the
    per-series relative forecast-error variance (diagonal of C).  Series with
    higher relative variance are less reliable and are moved more by the GLS.

    Args:
        impact_df (pd.DataFrame): output of compute_impact().
        C (np.ndarray, shape (7, 7)): SS-shrunk relative covariance matrix.
        series (list[str], optional): series keys. Defaults to SERIES.
    Returns:
        pd.DataFrame, indexed by series key, sorted by mean absolute adjustment
        descending.  Columns: label, hierarchy_level, relative_variance,
        rel_var_rank (1 = most reliable), mean_delta_pct, mean_abs_delta_pct.
    """
    if series is None:
        series = SERIES

    rel_var = np.diag(C)
    rows = []
    for i, s in enumerate(series):
        mean_delta_pct     = impact_df[f'{s}_delta_pct'].mean()
        mean_abs_delta_pct = impact_df[f'{s}_delta_pct'].abs().mean()
        rows.append({
            'series':           s,
            'label':            DISPLAY_LABELS[s],
            'hierarchy_level':  _HIER_LEVEL[s],
            'relative_variance': rel_var[i],
            'mean_delta_pct':    mean_delta_pct,
            'mean_abs_delta_pct': mean_abs_delta_pct,
        })
    df = pd.DataFrame(rows).set_index('series')
    df['rel_var_rank'] = df['relative_variance'].rank(ascending=True).astype(int)
    return df.sort_values('mean_abs_delta_pct', ascending=False)


def who_pulled_whom(impact_df, series=None):
    """
    Summarise the mean absolute adjustment (% of base forecast) by hierarchy level.

    Under a naive "who pulled whom" framing, a larger adjustment at the leaf level
    (vs aggregate) would suggest the leaves were pulling.  In practice (see memo §5),
    all three levels are pulled downward by the GLS off-diagonal over-correction —
    no single level is the "anchor."

    Args:
        impact_df (pd.DataFrame): output of compute_impact().
        series (list[str], optional): series keys. Defaults to SERIES.
    Returns:
        dict: {
            'level_0_mean_abs_delta_pct': float — Total series
            'level_1_mean_abs_delta_pct': float — Cohort A and B average
            'level_2_mean_abs_delta_pct': float — Leaf average
        }
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
# Brief-specified plot: per-series view of what the reconciler produced.
# ---------------------------------------------------------------------------
def plot_timeline(train, future, reconciled_arr, series=None, save_dir='figures'):
    """
    Plot training actuals, future base forecasts, and reconciled forecasts per series.

    One subplot per series.  Blue = training actuals; orange dashed = base forecasts
    (future); red = reconciled forecasts (future).  Vertical dotted line marks the
    train/future boundary.

    Args:
        train (pd.DataFrame): training data.
        future (pd.DataFrame): future data.
        reconciled_arr (np.ndarray, shape (351, 7)): reconciled forecasts.
        series (list[str], optional): series keys. Defaults to SERIES.
        save_dir (str): directory for saving fig10_timeline.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)
    if series is None:
        series = SERIES

    base_arr   = pred_array(future)
    actual_tr  = actual_array(train)
    ncols, nrows = 4, (len(series) + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()

    boundary = future['date'].iloc[0]

    for i, s in enumerate(series):
        ax = axes[i]
        ax.plot(train['date'],  actual_tr[:, i]      / 1e6,
                color='steelblue', lw=0.8, alpha=0.7, label='Actual (training)')
        ax.plot(future['date'], base_arr[:, i]        / 1e6,
                color='darkorange', lw=1.0, ls='--', alpha=0.85, label='Base forecast (future)')
        ax.plot(future['date'], reconciled_arr[:, i]  / 1e6,
                color='crimson', lw=1.2, alpha=0.9, label='Reconciled (future)')
        ax.axvline(boundary, color='black', ls=':', lw=0.8)
        ax.set_title(DISPLAY_LABELS[s], fontsize=9)
        ax.set_ylabel('Millions', fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Training Actuals + Future Base Forecasts (dashed) + Reconciled Forecasts (red)',
        fontsize=10
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig10_timeline.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 11: Per-series reconciliation adjustment over the future period
# Brief-specified plot: (reconciled − base) / base × 100 per day and series.
# ---------------------------------------------------------------------------
def plot_deltas(impact_df, series=None, save_dir='figures'):
    """
    Bar chart of the daily reconciliation adjustment (% of base forecast) per series.

    Blue bars = upward adjustments; red bars = downward.  Green dashed line = mean.
    Reveals the direction, magnitude, and time-variation of the GLS correction.

    Args:
        impact_df (pd.DataFrame): output of compute_impact().
        series (list[str], optional): series keys. Defaults to SERIES.
        save_dir (str): directory for saving fig11_deltas.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)
    if series is None:
        series = SERIES

    ncols, nrows = 4, (len(series) + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(series):
        ax = axes[i]
        delta  = impact_df[f'{s}_delta_pct']
        colors = np.where(delta >= 0, 'steelblue', 'crimson')
        ax.bar(range(len(delta)), delta, color=colors, alpha=0.7, width=1.0)
        ax.axhline(0, color='black', lw=0.6)
        mean_d = delta.mean()
        ax.axhline(mean_d, color='darkgreen', ls='--', lw=1.0,
                   label=f'mean = {mean_d:.1f}%')
        ax.set_title(DISPLAY_LABELS[s], fontsize=9)
        ax.set_ylabel('Adjustment (% of base forecast)', fontsize=7)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Reconciliation Adjustment per Series: (reconciled − base) / base × 100\n'
        'Blue = upward adjustment, Red = downward adjustment',
        fontsize=10
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig11_deltas.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 12: Who-moved summary — mean absolute adjustment vs relative variance
# ---------------------------------------------------------------------------
def plot_who_moved(who_moved_df, save_dir='figures'):
    """
    Two-panel summary of which series moved most and why.

    Panel A: mean absolute adjustment (% of base forecast) per series, colored
             by hierarchy level (crimson = Total, orange = cohort, blue = leaf).
    Panel B: scatter of mean absolute adjustment vs relative forecast-error
             variance, showing that less reliable series are moved more.

    Args:
        who_moved_df (pd.DataFrame): output of analyze_who_moved().
        save_dir (str): directory for saving fig12_who_moved.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df = who_moved_df.reset_index()

    # Panel A: horizontal bar chart sorted by mean absolute adjustment
    ax = axes[0]
    ax.barh(
        range(len(df)),
        df['mean_abs_delta_pct'],
        color=['crimson'    if lvl == 0 else
               'darkorange' if lvl == 1 else 'steelblue'
               for lvl in df['hierarchy_level']],
        alpha=0.8
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Mean absolute adjustment (% of base forecast)')
    ax.set_title('Mean absolute reconciliation adjustment\nby series', fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='crimson',    alpha=0.8, label='Total (level 0)'),
        Patch(color='darkorange', alpha=0.8, label='Cohort (level 1)'),
        Patch(color='steelblue',  alpha=0.8, label='Leaf (level 2)'),
    ], fontsize=8, loc='lower right')

    # Panel B: scatter — adjustment vs relative forecast-error variance
    ax2 = axes[1]
    ax2.scatter(
        df['relative_variance'], df['mean_abs_delta_pct'],
        c=df['hierarchy_level'], cmap='Set1', s=80, zorder=3
    )
    for _, row in df.iterrows():
        ax2.annotate(
            row['label'],
            xy=(row['relative_variance'], row['mean_abs_delta_pct']),
            xytext=(3, 3), textcoords='offset points', fontsize=8
        )
    ax2.set_xlabel(
        'Relative forecast-error variance (diagonal of C)\nhigher = less reliable'
    )
    ax2.set_ylabel('Mean absolute adjustment (% of base forecast)')
    ax2.set_title(
        'Less reliable series are moved more\n'
        '(relative variance drives adjustment size)',
        fontsize=9
    )

    fig.suptitle('Impact Analysis: Who Moved and Why', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig12_who_moved.png'))
    return fig
