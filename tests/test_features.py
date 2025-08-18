import pandas as pd
from src.features.build_features import add_rolling_mean, add_rolling_slope

def test_add_rolling_mean():
    df = pd.DataFrame({
        'unit': [1]*6,
        'cycle': list(range(1, 7)),
        'sensor_12': [0, 1, 2, 3, 4, 5]
    })
    df = add_rolling_mean(df, ['sensor_12'], window=3)
    assert 'sensor_12_rollmean' in df.columns
    expected = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    assert all(abs(a-b) < 1e-5 for a, b in zip(df['sensor_12_rollmean'][1:5], expected[1:5]))

def test_add_rolling_slope():
    df = pd.DataFrame({
        'unit': [1]*6,
        'cycle': list(range(1,7)),
        'sensor_7': [1,2,3,4,5,6]
    })
    df = add_rolling_slope(df, ['sensor_7'], window=3)
    assert 'sensor_7_slope' in df.columns
    # slope should be close to 1 for this synthetic data
    slopes = df['sensor_7_slope'].dropna().values
    for s in slopes:
        assert abs(s - 1) < 1e-5
