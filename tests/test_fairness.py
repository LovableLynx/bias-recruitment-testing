import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics, compute_performance_metrics

DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10
RESULTS_PATH = PROJECT_ROOT / 'tests' / 'results.json'

def log_result(test_name, metrics, dp_pass, eo_pass):
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
    entry = {k: float(v) for k, v in metrics.items()}
    entry["dp_pass"] = bool(dp_pass)
    entry["eo_pass"] = bool(eo_pass)
    results[test_name] = entry
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

def run_fairness_check(test_name, model_path, X_test, y_true, sensitive_features):
    model = load_model(model_path)
    fairness_metrics = compute_fairness_metrics(model, X_test, y_true, sensitive_features)
    perf_metrics = compute_performance_metrics(model, X_test, y_true)
    dp_pass = fairness_metrics["demographic_parity_difference"] < DP_THRESHOLD
    eo_pass = fairness_metrics["equalized_odds_difference"] < EO_THRESHOLD
    log_result(test_name, {**fairness_metrics, **perf_metrics}, dp_pass, eo_pass)
    return fairness_metrics, dp_pass, eo_pass

def test_gender_fairness(test_data):
    """Confirms the assertion suite detects known gender bias in the baseline model."""
    X_test, y_true = test_data
    gender = X_test[:, 1]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("gender_biased", model_path, X_test, y_true, gender)
    assert not dp_pass, (
        f"Expected demographic parity violation on biased model, "
        f"but disparity ({metrics['demographic_parity_difference']:.3f}) was within threshold"
    )
    assert not eo_pass, (
        f"Expected equalized odds violation on biased model, "
        f"but disparity ({metrics['equalized_odds_difference']:.3f}) was within threshold"
    )

def test_ethnicity_fairness(test_data):
    """Confirms the assertion suite detects known ethnicity bias in the baseline model."""
    X_test, y_true = test_data
    ethnicity = X_test[:, 0]
    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter_ethnicity.pkl'
    metrics, dp_pass, eo_pass = run_fairness_check("ethnicity_biased", model_path, X_test, y_true, ethnicity)
    assert not dp_pass, (
        f"Expected demographic parity violation on biased model, "
        f"but disparity ({metrics['demographic_parity_difference']:.3f}) was within threshold"
    )
    assert not eo_pass, (
        f"Expected equalized odds violation on biased model, "
        f"but disparity ({metrics['equalized_odds_difference']:.3f}) was within threshold"
    )
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


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_performance_metrics(model, X_test, y_true):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba)
    }