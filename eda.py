import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from data import SERIES, LEAVES, S, pred_array, actual_array, DISPLAY_LABELS, SHORT_LABELS
from covariance import relative_residuals, schafer_strimmer


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')


def _acf(x, nlags=40):
    """Compute sample autocorrelation function up to nlags lags."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c0 = np.dot(x, x) / len(x)
    return np.array([
        np.dot(x[:len(x) - k], x[k:]) / (len(x) * c0) for k in range(nlags + 1)
    ])


# ---------------------------------------------------------------------------
# Fig 1: Base-forecast incoherence over time
# Decision defended: motivates the reconciliation task.
# Train actuals are perfectly coherent; base forecasts violate all three
# hierarchy identities.  The future gap is ~170× larger than the training gap,
# consistent with the ~10× volume growth and a longer forecast horizon.
# ---------------------------------------------------------------------------
def plot_incoherence(train, future, save_dir='figures'):
    """
    Plot the absolute hierarchy identity violation for all three levels over time.

    Shows train and future periods side-by-side.  Motivates why reconciliation
    is needed: base forecasts are incoherent, and the future incoherence is far
    larger than anything seen in training.

    Args:
        train (pd.DataFrame): training data with {s}_pred columns.
        future (pd.DataFrame): future data with {s}_pred columns.
        save_dir (str): directory for saving fig1_incoherence.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    dates_tr = train['date'].values
    dates_fu = future['date'].values

    identities = [
        ('Total = Cohort A + Cohort B',
         train['aggregate_pred']  - train['cohort_A_pred']  - train['cohort_B_pred'],
         future['aggregate_pred'] - future['cohort_A_pred'] - future['cohort_B_pred']),
        ('Cohort A = A1 + A2',
         train['cohort_A_pred']  - train['A1_pred']  - train['A2_pred'],
         future['cohort_A_pred'] - future['A1_pred'] - future['A2_pred']),
        ('Cohort B = B1 + B2',
         train['cohort_B_pred']  - train['B1_pred']  - train['B2_pred'],
         future['cohort_B_pred'] - future['B1_pred'] - future['B2_pred']),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    fig.suptitle(
        'Base Forecast Incoherence\n'
        'Identity violation = parent forecast − sum of child forecasts',
        fontsize=11
    )

    for ax, (label, gap_tr, gap_fu) in zip(axes, identities):
        mean_tr = gap_tr.abs().mean()
        max_tr  = gap_tr.abs().max()
        mean_fu = gap_fu.abs().mean()
        max_fu  = gap_fu.abs().max()

        ax.plot(dates_tr, gap_tr.abs() / 1e6, color='steelblue', lw=0.7, alpha=0.8,
                label=f'Training  mean={mean_tr/1e6:.2f}M  max={max_tr/1e6:.2f}M')
        ax.plot(dates_fu, gap_fu.abs() / 1e6, color='darkorange', lw=0.7, alpha=0.8,
                label=f'Future    mean={mean_fu/1e6:.2f}M  max={max_fu/1e6:.2f}M')
        ax.axvline(pd.Timestamp('2026-01-18'), color='black', ls='--', lw=0.9, alpha=0.6)
        ax.set_ylabel('Absolute incoherence (millions)')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Date')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig1_incoherence.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Heteroscedasticity
# Decision defended: relative forecast errors + per-day W_day rescaling.
# Three panels:
#   A. |forecast error| vs level scatter (Total series; pred and actual shown)
#   B. Forecast-error standard deviation by volume quintile (Total)
#   C. Absolute vs relative forecast-error spread per series
# ---------------------------------------------------------------------------
def plot_heteroscedasticity(train, save_dir='figures'):
    """
    Demonstrate that forecast-error variance grows proportionally with forecast level.

    Three panels show the heteroscedasticity evidence that motivates using relative
    forecast errors and per-day W_day = diag(pred) @ C @ diag(pred).

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig2_heteroscedasticity.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    res_raw  = actual_array(train) - pred_array(train)
    pred_mat = pred_array(train)
    actual_mat = actual_array(train)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: scatter |forecast error| vs level — Total series
    ax_a = fig.add_subplot(gs[0])
    agg_res    = np.abs(res_raw[:, 0])
    agg_pred   = pred_mat[:, 0]
    agg_actual = actual_mat[:, 0]

    ax_a.scatter(agg_pred / 1e6, agg_res / 1e6, s=6, alpha=0.35,
                 color='steelblue', label='forecast level')
    ax_a.scatter(agg_actual / 1e6, agg_res / 1e6, s=6, alpha=0.25,
                 color='darkorange', label='actual level')
    for xv, col in [(agg_pred, 'steelblue'), (agg_actual, 'darkorange')]:
        m, b_int = np.polyfit(xv / 1e6, agg_res / 1e6, 1)
        xl = np.linspace(xv.min(), xv.max(), 200) / 1e6
        ax_a.plot(xl, m * xl + b_int, color=col, lw=1.2, alpha=0.8)

    r_pred   = np.corrcoef(agg_res, agg_pred)[0, 1]
    r_actual = np.corrcoef(agg_res, agg_actual)[0, 1]
    ax_a.set_xlabel('Level (millions)')
    ax_a.set_ylabel('Absolute forecast error (millions)')
    ax_a.set_title(
        f'|Forecast error| vs level — Total\n'
        f'r(error, forecast) = {r_pred:.3f}   r(error, actual) = {r_actual:.3f}\n'
        '(forecast level is operationally relevant; actual shown for reference)',
        fontsize=8
    )
    ax_a.legend(fontsize=8)

    # Panel B: forecast-error standard deviation by volume quintile (Total)
    ax_b = fig.add_subplot(gs[1])
    df_tmp = pd.DataFrame({'agg_pred': agg_pred, 'agg_res': res_raw[:, 0]})
    df_tmp['quintile'] = pd.qcut(df_tmp['agg_pred'], 5, labels=False)
    q_stats = df_tmp.groupby('quintile').agg(
        std=('agg_res', 'std'),
        mean_pred=('agg_pred', 'mean'),
        n=('agg_res', 'count'),
    ).reset_index()
    q_stats['pct'] = q_stats['std'] / q_stats['mean_pred'] * 100

    ax_b.bar(range(5), q_stats['std'] / 1e6, color='steelblue', alpha=0.75)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(range(5), q_stats['pct'], 'o-', color='crimson', lw=1.5, ms=5,
               label='As % of level (right axis)')
    ax_b.set_xticks(range(5))
    ax_b.set_xticklabels(
        [f'Q{i+1}\n{v/1e6:.1f}M' for i, v in enumerate(q_stats['mean_pred'])],
        fontsize=7
    )
    ax_b.set_xlabel('Volume quintile (mean forecast, millions)')
    ax_b.set_ylabel('Forecast-error standard deviation (millions)')
    ax_b2.set_ylabel('Standard deviation as % of level', color='crimson')
    ax_b2.tick_params(axis='y', colors='crimson')
    ax_b.set_title('Forecast-error std by volume quintile\n(Total series)', fontsize=9)
    ax_b2.legend(fontsize=8, loc='upper left')

    # Panel C: absolute vs relative forecast-error spread per series
    ax_c = fig.add_subplot(gs[2])
    rel_res = res_raw / pred_mat

    positions_raw = np.arange(len(SERIES)) * 2.5
    positions_rel = positions_raw + 0.9

    ax_c.boxplot(
        [res_raw[:, i] / 1e6 for i in range(len(SERIES))],
        positions=positions_raw, widths=0.7,
        patch_artist=True, medianprops=dict(color='black', lw=1.5),
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        flierprops=dict(marker='.', ms=2, alpha=0.3),
        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )
    ax_c2 = ax_c.twinx()
    ax_c2.boxplot(
        [rel_res[:, i] for i in range(len(SERIES))],
        positions=positions_rel, widths=0.7,
        patch_artist=True, medianprops=dict(color='black', lw=1.5),
        boxprops=dict(facecolor='darkorange', alpha=0.6),
        flierprops=dict(marker='.', ms=2, alpha=0.3),
        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )

    ax_c.set_xticks(positions_raw + 0.45)
    ax_c.set_xticklabels([SHORT_LABELS[s] for s in SERIES], rotation=30,
                          ha='right', fontsize=7)
    ax_c.set_ylabel('Absolute forecast error (millions)', color='steelblue')
    ax_c.tick_params(axis='y', colors='steelblue')
    ax_c2.set_ylabel('Relative forecast error', color='darkorange')
    ax_c2.tick_params(axis='y', colors='darkorange')
    ax_c.set_title(
        'Absolute (blue, millions) vs relative (orange)\nforecast-error spread per series',
        fontsize=9
    )
    from matplotlib.patches import Patch
    ax_c.legend(
        handles=[Patch(color='steelblue', alpha=0.7, label='Absolute (millions)'),
                 Patch(color='darkorange', alpha=0.7, label='Relative')],
        fontsize=8, loc='upper left'
    )

    fig.suptitle(
        'Heteroscedasticity: forecast-error variance grows proportionally with scale',
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig2_heteroscedasticity.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Volume trend and monthly seasonality
# Decision defended: per-day W_day rescaling.
# ~10× growth over 3 years (mean daily ~855k → ~8.4M); November spike every year.
# A single absolute W would be dominated by peak-period days; per-day scaling
# ensures W reflects the error structure at each day's scale.
# ---------------------------------------------------------------------------
def plot_volume_trend(train, save_dir='figures'):
    """
    Show the ~10× volume growth and November seasonality pattern in the training data.

    Justifies using per-day W_day = diag(pred) @ C @ diag(pred) rather than a
    single absolute covariance matrix for all time periods.

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig3_volume_trend.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    train = train.copy()
    train['ym'] = train['date'].dt.to_period('M')
    monthly = train.groupby('ym')['aggregate_actual'].mean().reset_index()
    monthly['date']   = monthly['ym'].dt.to_timestamp()
    monthly['month']  = monthly['date'].dt.month
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
        Patch(color='crimson',    alpha=0.8, label='November'),
        Patch(color='darkorange', alpha=0.8, label='April'),
        Patch(color='steelblue',  alpha=0.8, label='Other months'),
    ], fontsize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean daily total actual (millions)')
    ax.set_title(
        'Monthly mean total — ~10× growth over 3 years\n'
        'November spike every year; elevated April',
        fontsize=9
    )
    nov_rows = monthly[monthly['is_nov']]
    for _, row in nov_rows.iterrows():
        ax.annotate(
            f"{row['aggregate_actual']/1e6:.1f}M",
            xy=(row['date'], row['aggregate_actual'] / 1e6),
            xytext=(0, 4), textcoords='offset points',
            ha='center', fontsize=7, color='crimson'
        )

    # Panel B: average by calendar month (seasonality profile)
    ax2 = axes[1]
    seasonal = train.copy()
    seasonal['month'] = train['date'].dt.month
    by_month = seasonal.groupby('month')['aggregate_actual'].mean()
    bar_colors = ['crimson' if m == 11 else ('darkorange' if m == 4 else 'steelblue')
                  for m in by_month.index]
    ax2.bar(by_month.index, by_month.values / 1e6, color=bar_colors, alpha=0.8)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        rotation=30, ha='right', fontsize=7)
    ax2.set_xlabel('Calendar month')
    ax2.set_ylabel('Mean daily actual (millions)')
    ax2.set_title(
        'Seasonal profile — November peak drives W estimation sensitivity',
        fontsize=9
    )

    fig.suptitle('Volume Growth and Seasonality (training actuals)', fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig3_volume_trend.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Forecast bias
# Decision defended: disclose, do not attempt to fix via reconciliation.
# Mean forecast error (actual − pred) is positive for all 7 series (1.1%–4.1%
# of mean level), but the base model over-predicts on >50% of days.
# Positive mean bias comes from a small number of large under-shoots on
# high-volume days.  Reconciliation redistributes within-day only and cannot
# change the per-day total.
# ---------------------------------------------------------------------------
def plot_bias(train, save_dir='figures'):
    """
    Show the systematic forecast bias present in the base predictions.

    Mean forecast error is positive (under-predicting on average) for all 7 series,
    despite the model over-predicting on >50% of training days.  The bias is driven
    by a small number of large under-shoots on high-volume days.  Reconciliation
    cannot correct this bias.

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig4_bias.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    res_raw  = actual_array(train) - pred_array(train)
    pred_mat = pred_array(train)

    mean_res  = res_raw.mean(axis=0)
    mean_pred = pred_mat.mean(axis=0)
    mean_pct  = mean_res / mean_pred * 100
    pct_underpred = (res_raw > 0).mean(axis=0) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [DISPLAY_LABELS[s] for s in SERIES]
    x = np.arange(len(SERIES))
    w = 0.6

    # Panel A: mean forecast error (millions)
    ax = axes[0]
    ax.bar(x, mean_res / 1e6, width=w, color='steelblue', alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Mean forecast error (millions)\n= mean (actual − forecast)')
    ax.set_title('Mean forecast error > 0 for all series\n(under-predicting on average)', fontsize=9)

    # Panel B: mean forecast error as % of level
    ax2 = axes[1]
    ax2.bar(x, mean_pct, width=w, color='darkorange', alpha=0.8)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax2.set_ylabel('Mean forecast error as % of mean forecast level')
    ax2.set_title('Bias: 1.1%–4.1% of level\n(reconciliation cannot fix this)', fontsize=9)
    for i, v in enumerate(mean_pct):
        ax2.text(i, v + 0.05, f'{v:.1f}%', ha='center', fontsize=7)

    # Panel C: frequency vs magnitude — Total series
    ax3 = axes[2]
    agg_res = res_raw[:, 0]
    pct_up = (agg_res > 0).mean() * 100
    pct_dn = 100 - pct_up
    mean_under = agg_res[agg_res > 0].mean()
    mean_over  = agg_res[agg_res <= 0].mean()

    ax3_r = ax3.twinx()
    ax3.bar(
        ['Under-forecast\n(actual > forecast)', 'Over-forecast\n(actual < forecast)'],
        [pct_up, pct_dn], color=['steelblue', 'darkorange'], alpha=0.75, width=0.5
    )
    ax3_r.bar(
        ['Under-forecast\n(actual > forecast)', 'Over-forecast\n(actual < forecast)'],
        [mean_under / 1e6, abs(mean_over) / 1e6],
        color=['steelblue', 'darkorange'], alpha=0.35, width=0.3
    )
    ax3.set_ylabel('% of training days')
    ax3_r.set_ylabel('Mean absolute error (millions)', color='grey')
    ax3.set_title(
        'Total: under-forecast on only ~45% of days\n'
        'but mean under-shoot >> mean over-shoot',
        fontsize=9
    )
    ax3.text(0, pct_up + 0.5, f'{pct_up:.1f}%', ha='center', fontsize=8)
    ax3.text(1, pct_dn + 0.5, f'{pct_dn:.1f}%', ha='center', fontsize=8)

    fig.suptitle(
        'Forecast Bias — disclose, do not attempt to correct via reconciliation',
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig4_bias.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Residual autocorrelation
# Decision defended: rolling-origin out-of-sample W estimation in the backtest;
# and acknowledging that the analytic SS λ is derived under IID assumptions.
# Lag-1 autocorrelation 0.34–0.56 and max same-sign runs up to 23 days show
# errors are not independent in time — persistent runs of over/under-prediction
# are common.  Motivates the rolling-origin backtest design.
# ---------------------------------------------------------------------------
def plot_acf(train, save_dir='figures'):
    """
    Plot the Autocorrelation Function (ACF) of raw forecast errors per series.

    Demonstrates that training forecast errors are serially correlated (lag-1
    autocorrelation 0.34–0.56), which motivates estimating W from past-only data
    at each rolling-origin step rather than from a shuffled or pooled sample.

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig5_acf.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    res_raw = actual_array(train) - pred_array(train)
    nlags   = 40
    conf    = 1.96 / np.sqrt(len(train))
    n_series = len(SERIES)
    ncols, nrows = 4, (n_series + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(SERIES):
        ax = axes[i]
        acf_vals = _acf(res_raw[:, i], nlags=nlags)
        lags = np.arange(nlags + 1)
        ax.bar(lags[1:], acf_vals[1:], color='steelblue', alpha=0.75, width=0.8)
        ax.axhline( conf, color='crimson', ls='--', lw=0.9)
        ax.axhline(-conf, color='crimson', ls='--', lw=0.9)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(
            f'{DISPLAY_LABELS[s]}\nLag-1 autocorrelation = {acf_vals[1]:.3f}',
            fontsize=9
        )
        ax.set_xlabel('Lag (days)', fontsize=8)
        ax.set_ylim(-0.35, 0.75)

        signs = np.sign(res_raw[:, i])
        streak = max_streak = 1
        for k in range(1, len(signs)):
            streak = streak + 1 if signs[k] == signs[k - 1] else 1
            max_streak = max(max_streak, streak)
        ax.text(0.97, 0.92, f'Longest same-sign run: {max_streak} days',
                transform=ax.transAxes, ha='right', fontsize=7, color='darkred')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Autocorrelation Function (ACF) of Forecast Errors per Series\n'
        'Red dashed = 95% confidence bounds (assuming independent errors)\n'
        'Lag-1 autocorrelation 0.34–0.56 — errors are not independent, '
        'motivating rolling-origin W estimation',
        fontsize=10
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig5_acf.png'))
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Estimated W — correlation structure, reliability ranking, key pairs
# Decision defended: full 7×7 W (not leaves-only); reliability-weighted
# reconciliation.  Fig 7 (sibling scatter plots) is consolidated here.
#
# Panel A: relative forecast-error correlation heatmap (all 7×7 pairs)
# Panel B: per-series relative forecast-error variance (reliability ranking)
# Panel C: three key scatter pairs — sibling leaves within each cohort, and
#          the near-collinear Total–Cohort A pair that drives the over-correction
# ---------------------------------------------------------------------------
def plot_W_structure(C, series, train, save_dir='figures'):
    """
    Show the correlation structure and reliability ranking of the covariance matrix C,
    plus scatter plots for three key series pairs.

    Panel A: relative forecast-error correlation heatmap (Schäfer–Strimmer shrunk).
    Panel B: reliability ranking — relative forecast-error variance per series
             (higher = less reliable = moved more by reconciliation).
    Panel C: key scatter pairs (A1–A2 siblings, B1–B2 siblings, Total–Cohort A)
             showing the correlation structure that drives reconciliation adjustments.

    Args:
        C (np.ndarray, shape (p, p)): SS-shrunk relative covariance matrix.
        series (list[str]): series keys in SERIES order.
        train (pd.DataFrame): training data (for relative residual scatter plots).
        save_dir (str): directory for saving fig6_W_structure.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    sd   = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    short   = [SHORT_LABELS[s]   for s in series]
    display = [DISPLAY_LABELS[s] for s in series]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1],
                           hspace=0.48, wspace=0.42)

    # Panel A: correlation heatmap (top, spans first 2 columns)
    ax_heat = fig.add_subplot(gs[0, :2])
    im = ax_heat.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_heat.set_xticks(range(len(series)))
    ax_heat.set_yticks(range(len(series)))
    ax_heat.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticklabels(short, fontsize=8)
    for i in range(len(series)):
        for j in range(len(series)):
            ax_heat.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                         fontsize=7, color='black' if abs(corr[i, j]) < 0.7 else 'white')
    plt.colorbar(im, ax=ax_heat, shrink=0.8)
    ax_heat.set_title(
        'Relative forecast-error correlation matrix C\n'
        '(Schäfer–Strimmer shrinkage applied; estimated from all 1,112 training days)',
        fontsize=9
    )

    # Panel B: reliability ranking bar chart (top-right)
    ax_bar = fig.add_subplot(gs[0, 2])
    rel_var = np.diag(C)
    order   = np.argsort(rel_var)[::-1]
    bar_labels = [display[i] for i in order]
    bar_vals   = rel_var[order]
    bar_colors = ['crimson' if rel_var[i] == rel_var.max() else
                  ('darkorange' if rel_var[i] > np.median(rel_var) else 'steelblue')
                  for i in order]
    ax_bar.barh(range(len(series)), bar_vals, color=bar_colors, alpha=0.8)
    ax_bar.set_yticks(range(len(series)))
    ax_bar.set_yticklabels(bar_labels, fontsize=8)
    ax_bar.set_xlabel('Relative forecast-error variance\n(higher = less reliable = moved more)')
    ax_bar.set_title('Reliability ranking', fontsize=9)
    for i, v in enumerate(bar_vals):
        ax_bar.text(v * 1.01, i, f'{v:.4f}', va='center', fontsize=7)

    # Panel C: three key scatter pairs (bottom row)
    rel_res = relative_residuals(train, series)
    idx = {s: i for i, s in enumerate(series)}

    key_pairs = [
        ('A1', 'A2',        f'Sibling leaves — Cohort A\nr = '),
        ('B1', 'B2',        f'Sibling leaves — Cohort B\nr = '),
        ('aggregate', 'cohort_A', f'Total vs Cohort A (near-collinear)\nr = '),
    ]

    scatter_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for ax, (s1, s2, title_prefix) in zip(scatter_axes, key_pairs):
        i1, i2 = idx[s1], idx[s2]
        c_val = corr[i1, i2]
        ax.scatter(rel_res[:, i1], rel_res[:, i2], s=5, alpha=0.3, color='steelblue')
        m, b_int = np.polyfit(rel_res[:, i1], rel_res[:, i2], 1)
        xl = np.linspace(rel_res[:, i1].min(), rel_res[:, i1].max(), 100)
        ax.plot(xl, m * xl + b_int, color='crimson', lw=1.5)
        ax.set_xlabel(f'{SHORT_LABELS[s1]} relative forecast error', fontsize=8)
        ax.set_ylabel(f'{SHORT_LABELS[s2]} relative forecast error', fontsize=8)
        ax.set_title(f'{title_prefix}{c_val:.3f}', fontsize=9)

    fig.suptitle(
        'Forecast-Error Covariance Structure: Correlations, Reliability, and Key Pairs',
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig6_W_structure.png'))
    return fig


# ---------------------------------------------------------------------------
# Appendix Fig 8: Residual non-Gaussianity
# MinT/GLS is optimal under Gaussian errors.  Heavy tails imply the GLS weights
# are approximately correct but may under-penalise extreme days.  This does not
# invalidate the method but is disclosed as a known limitation.
# ---------------------------------------------------------------------------
def plot_residual_tails(train, save_dir='figures'):
    """
    QQ plots of relative forecast errors per series vs the normal distribution.

    Heavy tails (excess kurtosis > 0) indicate that covariance-based MinT/GLS is
    a reasonable approximation but may under-weight tail events relative to an
    optimal robust estimator.

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig8_residual_tails.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    rel_res = relative_residuals(train, SERIES)
    ncols, nrows = 4, (len(SERIES) + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, s in enumerate(SERIES):
        ax = axes[i]
        x    = rel_res[:, i]
        kurt = float(pd.Series(x).kurtosis())
        skew = float(pd.Series(x).skew())

        (osm, osr), (slope, intercept, r) = stats.probplot(x, dist='norm')
        ax.scatter(osm, osr, s=5, alpha=0.4, color='steelblue')
        ql = np.linspace(min(osm), max(osm), 100)
        ax.plot(ql, slope * ql + intercept, color='crimson', lw=1.2)
        ax.set_title(
            f'{DISPLAY_LABELS[s]}\nExcess kurtosis = {kurt:.1f}  Skewness = {skew:.2f}',
            fontsize=8
        )
        ax.set_xlabel('Theoretical quantile (normal)', fontsize=7)
        ax.set_ylabel('Sample quantile', fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'QQ Plots of Relative Forecast Errors vs Normal Distribution\n'
        'Heavy tails visible — covariance-based MinT is approximately correct '
        'but may under-weight extreme days',
        fontsize=10
    )
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig8_residual_tails.png'))
    return fig


# ---------------------------------------------------------------------------
# Appendix Fig 9: Rolling correlation stability
# Decision defended: using full-train C for future reconciliation.
# If correlations drift substantially over time, a recent-window estimator
# would be preferable.  Broad stability supports using the full-train C.
# ---------------------------------------------------------------------------
def plot_rolling_stability(train, save_dir='figures'):
    """
    Track key pairwise relative-error correlations over a 180-day rolling window.

    If the correlation structure is broadly stable, the full-training-data C is a
    reliable choice for future reconciliation (lower estimation variance than a
    recent-window estimator).

    Args:
        train (pd.DataFrame): training data.
        save_dir (str): directory for saving fig9_rolling_stability.png.
    Returns:
        matplotlib.figure.Figure
    """
    _ensure_dir(save_dir)

    rel_res = relative_residuals(train, SERIES)
    dates   = train['date'].values

    pairs = [
        (SERIES.index('A1'),        SERIES.index('A2'),        'A1 – A2 (sibling leaves, Cohort A)'),
        (SERIES.index('B1'),        SERIES.index('B2'),        'B1 – B2 (sibling leaves, Cohort B)'),
        (SERIES.index('cohort_A'),  SERIES.index('cohort_B'),  'Cohort A – Cohort B'),
        (SERIES.index('aggregate'), SERIES.index('cohort_A'),  'Total – Cohort A'),
    ]

    window     = 180
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
        ax.plot(roll_dates, roll_corrs[label], lw=1.2, label=label,
                color=col, alpha=0.85)

    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Date (end of rolling 180-day window)')
    ax.set_ylabel('Rolling correlation')
    ax.set_title(
        '180-Day Rolling Correlation Stability\n'
        'Broadly stable → full-training C is a reliable choice for future reconciliation',
        fontsize=10
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'fig9_rolling_stability.png'))
    return fig


def run_all(train, future, C, save_dir='figures'):
    """
    Run all main EDA figures and return the list of Figure objects.

    Figures 8 and 9 (non-Gaussianity and rolling stability) are produced here
    for completeness; they appear in the Appendix section of eda.ipynb.

    Args:
        train (pd.DataFrame): training data.
        future (pd.DataFrame): future data.
        C (np.ndarray, shape (7, 7)): SS-shrunk relative covariance matrix.
        save_dir (str): directory for saving all PNGs.
    Returns:
        list[matplotlib.figure.Figure]: one figure per plot function called.
    """
    figs = []
    figs.append(plot_incoherence(train, future, save_dir))
    figs.append(plot_heteroscedasticity(train, save_dir))
    figs.append(plot_volume_trend(train, save_dir))
    figs.append(plot_bias(train, save_dir))
    figs.append(plot_acf(train, save_dir))
    figs.append(plot_W_structure(C, SERIES, train, save_dir))
    figs.append(plot_residual_tails(train, save_dir))
    figs.append(plot_rolling_stability(train, save_dir))
    return figs
