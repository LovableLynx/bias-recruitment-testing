"""
Independent hand-verification of the fairness metric formulas against
fairlearn's output, as a follow-up to the paper's Section 7 limitation:
"both libraries are treated as trusted instruments rather than re-derived
or audited."

This script does not re-implement fairlearn or aequitas. It computes
demographic parity difference and equalized odds difference directly from
their textbook definitions (group-wise positive-prediction rate and
group-wise false-positive/false-negative rate, from raw y_true/y_pred
arrays with plain numpy), independently of fairlearn's own internal
implementation, and compares the result against fairlearn's
demographic_parity_difference / equalized_odds_difference on the same
real models, same real test split, same sensitive-attribute columns used
throughout the rest of the study (Section 3.1-3.2).

This closes part of the "libraries treated as trusted instruments" gap:
it does not audit fairlearn's or aequitas's source code, but it does
confirm that an independent, from-definition computation agrees with
fairlearn's reported numbers on this study's actual data, for every
model/attribute pair used in Section 4.1 (Table 1).
"""
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from fairness_checks import load_model, compute_fairness_metrics

RESULTS_PATH = PROJECT_ROOT / "scripts" / "metric_verification_results.json"


def hand_demographic_parity_difference(y_pred, sensitive_features):
    """max_g(P(pred=1 | group=g)) - min_g(P(pred=1 | group=g)), from definition."""
    groups = np.unique(sensitive_features)
    rates = []
    for g in groups:
        mask = sensitive_features == g
        rates.append(y_pred[mask].mean())
    return max(rates) - min(rates)


def hand_equalized_odds_difference(y_true, y_pred, sensitive_features):
    """
    max over {FPR, FNR} of (max_g(rate) - min_g(rate)), from definition.
    FPR = P(pred=1 | true=0), FNR = P(pred=0 | true=1), per group.
    """
    groups = np.unique(sensitive_features)
    fpr_by_group = []
    fnr_by_group = []
    for g in groups:
        mask = sensitive_features == g
        y_t = y_true[mask]
        y_p = y_pred[mask]

        negatives = y_t == 0
        positives = y_t == 1

        fpr = y_p[negatives].mean() if negatives.sum() > 0 else 0.0
        fnr = 1 - y_p[positives].mean() if positives.sum() > 0 else 0.0

        fpr_by_group.append(fpr)
        fnr_by_group.append(fnr)

    fpr_spread = max(fpr_by_group) - min(fpr_by_group)
    fnr_spread = max(fnr_by_group) - min(fnr_by_group)
    return max(fpr_spread, fnr_spread)


def main():
    data_path = PROJECT_ROOT / "data" / "FairCVtest" / "data" / "FairCVdb.npy"
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data["Profiles Test"]
    y_true = (data["Blind Labels Test"] > 0.5).astype(int)

    cases = [
        ("gender_biased", "baseline_biased_recruiter.pkl", 1),
        ("ethnicity_biased", "baseline_biased_recruiter_ethnicity.pkl", 0),
        ("reference_gender", "baseline_fair_recruiter.pkl", 1),
        ("reference_ethnicity", "baseline_fair_recruiter.pkl", 0),
    ]

    results = []
    print(f"{'case':22s} {'metric':6s} {'fairlearn':>12s} {'hand-derived':>14s} {'abs diff':>12s} {'match':>7s}")
    print("-" * 80)

    for name, model_file, attr_idx in cases:
        model_path = PROJECT_ROOT / "src" / "models" / model_file
        model = load_model(model_path)
        sensitive_features = X_test[:, attr_idx]

        fairlearn_metrics = compute_fairness_metrics(model, X_test, y_true, sensitive_features)

        y_pred = model.predict(X_test)
        hand_dp = hand_demographic_parity_difference(y_pred, sensitive_features)
        hand_eo = hand_equalized_odds_difference(y_true, y_pred, sensitive_features)

        for metric_label, fl_val, hand_val in [
            ("dp", fairlearn_metrics["demographic_parity_difference"], hand_dp),
            ("eo", fairlearn_metrics["equalized_odds_difference"], hand_eo),
        ]:
            diff = abs(fl_val - hand_val)
            match = diff < 1e-9
            print(f"{name:22s} {metric_label:6s} {fl_val:12.6f} {hand_val:14.6f} {diff:12.2e} {str(match):>7s}")
            results.append({
                "case": name,
                "model_file": model_file,
                "sensitive_attribute_index": attr_idx,
                "metric": metric_label,
                "fairlearn_value": float(fl_val),
                "hand_derived_value": float(hand_val),
                "abs_difference": float(diff),
                "match": bool(match),
            })

    all_match = all(r["match"] for r in results)
    summary = {
        "purpose": (
            "Independent hand-verification of demographic parity difference and "
            "equalized odds difference, computed from their textbook definitions "
            "via plain numpy, against fairlearn's demographic_parity_difference and "
            "equalized_odds_difference on the same real models and real FairCVdb test "
            "split used throughout Section 4.1."
        ),
        "all_match": all_match,
        "tolerance": 1e-9,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll {len(results)} fairlearn vs. hand-derived comparisons match (tolerance 1e-9): {all_match}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
