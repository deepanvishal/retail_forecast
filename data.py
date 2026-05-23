import numpy as np
import pandas as pd

SERIES = ['aggregate', 'cohort_A', 'cohort_B', 'A1', 'A2', 'B1', 'B2']
LEAVES = ['A1', 'A2', 'B1', 'B2']

# 7x4 summing matrix: rows = all 7 series, cols = 4 leaves [A1, A2, B1, B2]
S = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=float)

# Reader-friendly display names — single source of truth for all plot labels,
# legends, and printed tables.  Internal code/CSV keys (SERIES list above) are
# unchanged; these names are for display only.
#
# Full form: includes hierarchy role (panel titles, legends, standalone labels).
# Short form: for cramped contexts (heatmap tick labels, dense axis ticks).
DISPLAY_LABELS = {
    'aggregate': 'Total',
    'cohort_A':  'Cohort A',
    'cohort_B':  'Cohort B',
    'A1': 'A1 (Cohort A leaf)',
    'A2': 'A2 (Cohort A leaf)',
    'B1': 'B1 (Cohort B leaf)',
    'B2': 'B2 (Cohort B leaf)',
}

SHORT_LABELS = {
    'aggregate': 'Total',
    'cohort_A':  'Cohort A',
    'cohort_B':  'Cohort B',
    'A1': 'A1',
    'A2': 'A2',
    'B1': 'B1',
    'B2': 'B2',
}

# Display names for reconciliation methods (backtest summaries and plots).
# Internal keys (base, ols, …) are unchanged; these names are for display only.
METHOD_LABELS = {
    'base':        'Base (no reconciliation)',
    'ols':         'OLS - equal weights',
    'wls_raw':     'WLS-raw (absolute variance, diagonal)',
    'wls_scaled':  'WLS-scaled (relative variance, diagonal)',
    'mint_raw':    'MinT-raw (absolute covariance, full)',
    'mint_scaled': 'MinT-scaled (relative covariance, full) [shipped]',
}

_DATE_FMT = '%m/%d/%y'


def load(path='forecast_data_anonymized.csv'):
    """
    Load, parse, and split the source forecast CSV into train and future sets.

    Reads the given path, parses dates, sorts chronologically, and splits on
    the 'label' column ('train' / 'future').  Also detects calendar gaps
    (missing dates) in the training period.

    Args:
        path (str): path to the CSV. Default: 'forecast_data_anonymized.csv'.
    Returns:
        train (pd.DataFrame): 1112-row training set (label == 'train').
        future (pd.DataFrame): 351-row future set (label == 'future').
        gap_info (dict): {
            'max_gap_days': int — largest gap between consecutive train dates;
            'gap_dates': list[Timestamp] — train dates where gap > 1 calendar day;
            'n_train': int; 'n_future': int.
        }
    Notes:
        Uses fillna(0) on the date diff so the first row is included in the
        index without dropping it (dropna would misalign the boolean mask).
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format=_DATE_FMT)
    df = df.sort_values('date').reset_index(drop=True)

    train = df[df['label'] == 'train'].reset_index(drop=True)
    future = df[df['label'] == 'future'].reset_index(drop=True)

    gaps = train['date'].diff().dt.days.fillna(0)
    gap_info = {
        'max_gap_days': int(gaps.max()),
        'gap_dates': train['date'][gaps > 1].tolist(),
        'n_train': len(train),
        'n_future': len(future),
    }
    return train, future, gap_info


def pred_array(data):
    """
    Extract base forecast values as a 2-D NumPy array.

    Args:
        data (pd.DataFrame): train or future DataFrame with columns {s}_pred
            for each s in SERIES.
    Returns:
        np.ndarray, shape (n, 7): base forecasts in SERIES order.
            Columns: [aggregate, cohort_A, cohort_B, A1, A2, B1, B2].
    """
    return np.column_stack([data[s + '_pred'].values for s in SERIES])


def actual_array(data):
    """
    Extract observed actual values as a 2-D NumPy array.

    Args:
        data (pd.DataFrame): train or future DataFrame with columns {s}_actual
            for each s in SERIES.
    Returns:
        np.ndarray, shape (n, 7): actuals in SERIES order.
            Columns: [aggregate, cohort_A, cohort_B, A1, A2, B1, B2].
    """
    return np.column_stack([data[s + '_actual'].values for s in SERIES])
