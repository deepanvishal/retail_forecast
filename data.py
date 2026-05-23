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

_DATE_FMT = '%m/%d/%y'


def load(path='forecast_data_anonymized.csv'):
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
    return np.column_stack([data[s + '_pred'].values for s in SERIES])


def actual_array(data):
    return np.column_stack([data[s + '_actual'].values for s in SERIES])
