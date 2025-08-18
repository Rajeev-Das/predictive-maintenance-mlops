import numpy as np
import pandas as pd
from src.utils.utils import create_sequences, create_test_sequences

def test_create_sequences():
    df = pd.DataFrame({
        'unit': [1]*10,
        'cycle': list(range(1, 11)),
        'feature_a': range(10),
        'RUL': list(reversed(range(10)))
    })
    # Build sequences of 5
    X, y = create_sequences(df, 5, ['feature_a'])
    assert X.shape == (6, 5, 1)
    assert y.shape == (6,)
    # Check that the first sequence contains features 0-4 and the corresponding RUL is 5
    np.testing.assert_array_equal(X[0].flatten(), np.array([0, 1, 2, 3, 4]))
    assert y[0] == 5
    # Check that the last sequence contains features 5-9 and the corresponding RUL is 0
    np.testing.assert_array_equal(X[-1].flatten(), np.array([5, 6, 7, 8, 9]))
    assert y[-1] == 0

def test_create_test_sequences():
    df = pd.DataFrame({
        'unit': [2]*7,
        'cycle': list(range(1,8)),
        'feature_b': range(7)
    })
    X, units = create_test_sequences(df, 5, ['feature_b'])
    assert X.shape == (1, 5, 1)
    assert units == [2]
    # Should be last 5 cycles
    assert (X.flatten() == np.array([2,3,4,5,6])).all()
