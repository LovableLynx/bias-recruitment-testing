"""
Independent hand-verification of aequitas's disparity-ratio formulas,
extending scripts/verify_metric_formulas.py (which covers fairlearn) to
the second library used in this study (Section 3.2, test_aequitas_crossvalidation.py).

aequitas reports group disparities as ratios relative to a reference group
(e.g. female FNR / male FNR), not differences like fairlearn's
demographic_parity_difference / equalized_odds_difference. This script
computes false positive rate disparity, false negative rate disparity, and
predicted positive rate disparity directly from their ratio definitions,
independently of aequitas's own Group/Bias classes, and compares the
result against aequitas's actual output on the same two real model/
attribute cases exercised by test_aequitas_crossvalidation.py: the
gender-biased model and the reference model, both against gender
(sensitive_col_idx=1), with 'Male' as the reference group.
"""
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from aequitas_checks import run_aequitas_audit
from fairness_checks import load_model

RESULTS_PATH = PROJECT_ROOT / "scripts" / "aequitas_verification_results.json"

GROUP_MAP = {0: "Male", 1: "Female"}
REF_GROUP = "Male"


def hand_disparity_ratios(model_path, X_test, y_true, sensitive_col_idx):
    """
    Compute fpr_disparity, fnr_disparity, ppr_disparity from their ratio
    definitions, independently of aequitas's Group/Bias classes.

    fpr, fnr are standard per-group rates: fpr = FP/(FP+TN), fnr = FN/(FN+TP).
    ppr is NOT a per-group positive-prediction rate: aequitas defines it as
    each group's predicted-positive count divided by the total number of
    predicted positives across ALL groups combined (k = sum of predicted
    positives dataset-wide; see aequitas/group.py, "k = Total number of
    predicted positives in sample"), i.e. each group's *share* of all
    positive predictions, not P(pred=1 | group). This was confirmed by
    reading aequitas's own source rather than assumed.
    """
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    attr = X_test[:, sensitive_col_idx].astype(int)

    groups = {"Male": attr == 0, "Female": attr == 1}
    total_predicted_positives = (y_pred == 1).sum()  # k, dataset-wide

    rates = {}
    for label, mask in groups.items():
        y_t = y_true[mask]
        y_p = y_pred[mask]
        negatives = y_t == 0
        positives = y_t == 1
        fpr = y_p[negatives].mean() if negatives.sum() > 0 else 0.0
        fnr = 1 - y_p[positives].mean() if positives.sum() > 0 else 0.0
        group_predicted_positives = (y_p == 1).sum()
        ppr = group_predicted_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
        rates[label] = {"fpr": fpr, "fnr": fnr, "ppr": ppr}

    ref = rates[REF_GROUP]
    female = rates["Female"]
    return {
        "fpr_disparity": female["fpr"] / ref["fpr"] if ref["fpr"] != 0 else float("nan"),
        "fnr_disparity": female["fnr"] / ref["fnr"] if ref["fnr"] != 0 else float("nan"),
        "ppr_disparity": female["ppr"] / ref["ppr"] if ref["ppr"] != 0 else float("nan"),
    }


def main():
    data_path = PROJECT_ROOT / "data" / "FairCVtest" / "data" / "FairCVdb.npy"
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data["Profiles Test"]
    y_true = (data["Blind Labels Test"] > 0.5).astype(int)

    cases = [
        ("gender_biased", "baseline_biased_recruiter.pkl"),
        ("reference_gender", "baseline_fair_recruiter.pkl"),
    ]

    results = []
    print(f"{'case':18s} {'metric':16s} {'aequitas':>12s} {'hand-derived':>14s} {'abs diff':>12s} {'match':>7s}")
    print("-" * 82)

    for name, model_file in cases:
        model_path = PROJECT_ROOT / "src" / "models" / model_file

        aequitas_result = run_aequitas_audit(
            model_path, X_test, y_true, sensitive_col_idx=1, attribute_name="gender",
            group_map=GROUP_MAP, ref_group=REF_GROUP,
        )
        female_row = aequitas_result[aequitas_result["attribute_value"] == "Female"].iloc[0]

        hand = hand_disparity_ratios(model_path, X_test, y_true, sensitive_col_idx=1)

        for metric_label in ["fpr_disparity", "fnr_disparity", "ppr_disparity"]:
            aeq_val = float(female_row[metric_label])
            hand_val = float(hand[metric_label])
            diff = abs(aeq_val - hand_val)
            match = diff < 1e-6
            print(f"{name:18s} {metric_label:16s} {aeq_val:12.6f} {hand_val:14.6f} {diff:12.2e} {str(match):>7s}")
            results.append({
                "case": name,
                "model_file": model_file,
                "metric": metric_label,
                "aequitas_value": aeq_val,
                "hand_derived_value": hand_val,
                "abs_difference": diff,
                "match": bool(match),
            })

    all_match = all(r["match"] for r in results)
    summary = {
        "purpose": (
            "Independent hand-verification of aequitas's fpr_disparity, "
            "fnr_disparity, and ppr_disparity, computed as group-relative-to-"
            "reference-group rate ratios via plain numpy, against aequitas's "
            "own Group/Bias output, on the same two real model cases "
            "(gender_biased, reference_gender) exercised by "
            "test_aequitas_crossvalidation.py."
        ),
        "reference_group": REF_GROUP,
        "all_match": all_match,
        "tolerance": 1e-6,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll {len(results)} aequitas vs. hand-derived comparisons match (tolerance 1e-6): {all_match}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
