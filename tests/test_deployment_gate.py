# tests/test_deployment_gate.py
"""
Deployment gate tests: verify that src/deployment_gate.py's evaluate_gate()
correctly translates fairness metrics into deploy/block decisions. Unlike
test_fairness.py (which validates fairness classification directly against
a threshold), these tests exercise the gate function itself, independent of
how the metrics were computed.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics
from deployment_gate import evaluate_gate, BLOCK, DEPLOY

def test_gender_model_is_deployable(test_data):
    """Gate check: biased gender model should be BLOCKed (not deployable)."""
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    decision = evaluate_gate(metrics)
    assert decision == BLOCK, (
        f"Gate should have BLOCKED the biased model but returned {decision} "
        f"(dp={metrics['demographic_parity_difference']:.3f}, "
        f"eo={metrics['equalized_odds_difference']:.3f})"
    )

def test_ethnicity_model_is_deployable(test_data):
    """Gate check: biased ethnicity model should be BLOCKed (not deployable)."""
    X_test, y_true = test_data
    ethnicity = X_test[:, 0]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter_ethnicity.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, ethnicity)
    decision = evaluate_gate(metrics)
    assert decision == BLOCK, (
        f"Gate should have BLOCKED the biased model but returned {decision} "
        f"(dp={metrics['demographic_parity_difference']:.3f}, "
        f"eo={metrics['equalized_odds_difference']:.3f})"
    )

def test_reference_model_is_deployable(test_data):
    """Gate check: reference (blind-label) model should be DEPLOYable."""
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')
    metrics = compute_fairness_metrics(model, X_test, y_true, gender)
    decision = evaluate_gate(metrics)
    assert decision == DEPLOY, (
        f"Gate should have DEPLOYed the reference model but returned {decision} "
        f"(dp={metrics['demographic_parity_difference']:.3f}, "
        f"eo={metrics['equalized_odds_difference']:.3f})"
    )
