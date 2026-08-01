# tests/test_fairness.py
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics

# Thresholds — your fairness "gate"
DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10

def load_test_data():
    data_path = PROJECT_ROOT / 'data' / 'FairCVtest' / 'data' / 'FairCVdb.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data['Profiles Test']
    y_true_blind = (data['Blind Labels Test'] > 0.5).astype(int)
    return X_test, y_true_blind

def test_gender_fairness():
    X_test, y_true = load_test_data()
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"

def test_ethnicity_fairness():
    X_test, y_true = load_test_data()
    ethnicity = X_test[:, 0]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter_ethnicity.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, ethnicity)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"


def test_fair_model_passes_gender():
    X_test, y_true = load_test_data()
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"

def test_fair_model_passes_ethnicity():
    X_test, y_true = load_test_data()
    ethnicity = X_test[:, 0]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, ethnicity)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"