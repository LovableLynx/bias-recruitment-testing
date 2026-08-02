# tests/test_deployment_gate.py
"""
Deployment gate tests: simulate a CI/CD quality gate that blocks
biased models from deployment. Unlike test_fairness.py (which validates
detection accuracy), these tests assert models MUST pass fairness
thresholds to be considered deployable.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics

DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10

def test_gender_model_is_deployable(test_data):
    """Gate check: biased gender model should FAIL this gate (not deployable)."""
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"BLOCKED: demographic parity violation ({metrics['demographic_parity_difference']:.3f})"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"BLOCKED: equalized odds violation ({metrics['equalized_odds_difference']:.3f})"

def test_fair_model_is_deployable(test_data):
    """Gate check: fair model should PASS this gate (deployable)."""
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    assert metrics["demographic_parity_difference"] < DP_THRESHOLD, \
        f"BLOCKED: demographic parity violation ({metrics['demographic_parity_difference']:.3f})"
    assert metrics["equalized_odds_difference"] < EO_THRESHOLD, \
        f"BLOCKED: equalized odds violation ({metrics['equalized_odds_difference']:.3f})"