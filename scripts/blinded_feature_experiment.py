"""
Demographically blinded replication of the RQ1/RQ2 pipeline, as a
follow-up to Section 7's limitation that none of the three baseline
models are blinded to demographic features: all three (Section 3.1)
receive the raw gender (feature index 1) and ethnicity (feature index 0)
columns as input, and only their training labels differ.

This script re-trains the same three logistic regression classifiers
(same hyperparameters as src/models/*.pkl: sklearn LogisticRegression,
max_iter=1000, all other defaults) on the identical FairCVdb profiles and
label sets used throughout the paper (Section 3.1), with exactly one
change: feature indices 0 and 1 (ethnicity, gender) are dropped from the
input vector before training and before inference, so the classifier
never sees the sensitive attribute directly, for any of the three models,
including the reference model.

It then re-runs the same two checks as Sections 4.1-4.2: does each
blinded model land on the expected side of the fairness threshold given
its label construction, and does evaluate_gate() correctly translate that
into a deploy/reject decision.

This is a single re-run, not a new mutation study: it answers "does the
metric-vs-gate distinction still hold once the more obvious source of
demographic leakage is removed," not a new fault-injection experiment.
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from fairness_checks import compute_fairness_metrics
from deployment_gate import evaluate_gate, DP_THRESHOLD, EO_THRESHOLD

RESULTS_PATH = PROJECT_ROOT / "scripts" / "blinded_feature_results.json"

SENSITIVE_INDICES = [0, 1]  # ethnicity, gender


def drop_sensitive_columns(X):
    keep = [i for i in range(X.shape[1]) if i not in SENSITIVE_INDICES]
    return X[:, keep]


def train_model(X_train_blinded, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_blinded, y_train)
    return model


def main():
    data_path = PROJECT_ROOT / "data" / "FairCVtest" / "data" / "FairCVdb.npy"
    data = np.load(data_path, allow_pickle=True).item()

    X_train = data["Profiles Train"]
    X_test = data["Profiles Test"]
    X_train_blinded = drop_sensitive_columns(X_train)
    X_test_blinded = drop_sensitive_columns(X_test)
    print(f"Original feature count: {X_train.shape[1]}, blinded: {X_train_blinded.shape[1]}")

    y_blind_train = (data["Blind Labels Train"] > 0.5).astype(int)
    y_blind_test = (data["Blind Labels Test"] > 0.5).astype(int)
    y_gender_biased_train = (data["Biased Labels Train (Gender)"] > 0.5).astype(int)
    y_ethnicity_biased_train = (data["Biased Labels Train (Ethnicity)"] > 0.5).astype(int)

    gender_col_test = X_test[:, 1]
    ethnicity_col_test = X_test[:, 0]

    model_specs = [
        ("reference_blinded", y_blind_train, "gender", gender_col_test, True),
        ("reference_blinded", y_blind_train, "ethnicity", ethnicity_col_test, True),
        ("gender_biased_blinded", y_gender_biased_train, "gender", gender_col_test, False),
        ("ethnicity_biased_blinded", y_ethnicity_biased_train, "ethnicity", ethnicity_col_test, False),
    ]

    trained_cache = {}
    results = []

    print(f"\n{'model':26s} {'attribute':10s} {'dp':>8s} {'eo':>8s} {'gate':>8s} {'expected':>10s} {'ok':>5s}")
    print("-" * 80)

    for model_name, y_train, attr_name, sensitive_test_col, expect_pass in model_specs:
        cache_key = model_name + "|" + str(id(y_train))
        if model_name not in trained_cache:
            trained_cache[model_name] = train_model(X_train_blinded, y_train)
        model = trained_cache[model_name]

        y_true_test = y_blind_test  # ground-truth outcome label is always the blind target, per Section 3.1
        fairness_metrics = compute_fairness_metrics(model, X_test_blinded, y_true_test, sensitive_test_col)
        dp = fairness_metrics["demographic_parity_difference"]
        eo = fairness_metrics["equalized_odds_difference"]
        gate_decision = evaluate_gate(fairness_metrics)

        expected_gate = "DEPLOY" if expect_pass else "BLOCK"
        expected_pass_bool = expect_pass
        actual_pass = dp < DP_THRESHOLD and eo < EO_THRESHOLD
        ok = (gate_decision == expected_gate)

        print(f"{model_name:26s} {attr_name:10s} {dp:8.4f} {eo:8.4f} {gate_decision:>8s} {expected_gate:>10s} {str(ok):>5s}")
        results.append({
            "model": model_name,
            "attribute": attr_name,
            "demographic_parity_difference": float(dp),
            "equalized_odds_difference": float(eo),
            "gate_decision": gate_decision,
            "expected_gate_decision": expected_gate,
            "matches_expected": bool(ok),
        })

    all_match = all(r["matches_expected"] for r in results)
    summary = {
        "purpose": (
            "Re-run of Sections 4.1-4.2 (fairness classification, gate correctness) "
            "with feature indices 0 and 1 (ethnicity, gender) dropped from the input "
            "vector for all three models, including the reference model, addressing "
            "Section 7's limitation that no model was previously blinded to "
            "demographic features."
        ),
        "blinding_method": "feature removal (drop columns 0, 1 before training and inference)",
        "original_feature_count": int(X_train.shape[1]),
        "blinded_feature_count": int(X_train_blinded.shape[1]),
        "all_match_expected": all_match,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll {len(results)} blinded-model gate decisions match expectation: {all_match}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
