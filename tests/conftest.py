import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def test_data():
    data_path = PROJECT_ROOT / 'data' / 'FairCVtest' / 'data' / 'FairCVdb.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data['Profiles Test']
    y_true_blind = (data['Blind Labels Test'] > 0.5).astype(int)
    return X_test, y_true_blind