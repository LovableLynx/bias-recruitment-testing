import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics

DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10
RESULTS_PATH = PROJECT_ROOT / 'tests' / 'results.json'

def log_result(test_name, metrics, dp_pass, eo_pass):
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
    results[test_name] = {
        "demographic_parity_difference": float(metrics["demographic_parity_difference"]),
        "equalized_odds_difference": float(metrics["equalized_odds_difference"]),
        "dp_pass": bool(dp_pass),
        "eo_pass": bool(eo_pass)
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

def run_fairness_check(test_name, model_path, X_test, y_true, sensitive_features):
    model = load_model(model_path)
    metrics = compute_fairness_metrics(model, X_test, y_true, sensitive_features)
    dp_pass = metrics["demographic_parity_difference"] < DP_THRESHOLD
    eo_pass = metrics["equalized_odds_difference"] < EO_THRESHOLD
    log_result(test_name, metrics, dp_pass, eo_pass)
    return metrics, dp_pass, eo_pass

def test_gender_fairness(test_data):
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("gender_biased", model_path, X_test, y_true, gender)
    assert dp_pass, f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert eo_pass, f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"

def test_ethnicity_fairness(test_data):
    X_test, y_true = test_data
    ethnicity = X_test[:, 0]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter_ethnicity.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("ethnicity_biased", model_path, X_test, y_true, ethnicity)
    assert dp_pass, f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert eo_pass, f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"

def test_fair_model_passes_gender(test_data):
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("fair_gender", model_path, X_test, y_true, gender)
    assert dp_pass, f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert eo_pass, f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"

def test_fair_model_passes_ethnicity(test_data):
    X_test, y_true = test_data
    ethnicity = X_test[:, 0]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("fair_ethnicity", model_path, X_test, y_true, ethnicity)
    assert dp_pass, f"Demographic parity failed: {metrics['demographic_parity_difference']:.3f}"
    assert eo_pass, f"Equalized odds failed: {metrics['equalized_odds_difference']:.3f}"