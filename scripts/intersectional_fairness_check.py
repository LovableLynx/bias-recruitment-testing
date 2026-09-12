"""
Intersectional fairness check, addressing part of Section 7's first
limitation: "the study uses a single synthetic dataset with known,
injected bias and only two fairness metrics and two protected attributes,
so behavior on naturally occurring bias and generalization to
intersectional criteria or additional attributes remain untested."

This does not add a new protected attribute (FairCVdb only provides
gender and ethnicity); it instead recomputes demographic parity and
equalized odds across the six gender x ethnicity intersectional
subgroups (2 genders x 3 ethnicity categories), rather than one
protected attribute at a time as Sections 3.2/4.1 do, for all three
real trained models. This tests whether a subgroup can be masked by the
single-attribute analysis: a model could satisfy demographic parity
separately on gender and on ethnicity while still treating a specific
gender x ethnicity combination worse than the rest, a form of
disparity single-attribute analysis cannot detect by construction.

Both metrics are computed the same way as scripts/verify_metric_formulas.py
(max-group-rate minus min-group-rate across the six intersectional
groups), not via fairlearn's sensitive_features argument, since fairlearn's
demographic_parity_difference already accepts only one sensitive-feature
array; combining gender and ethnicity into a single six-valued group label
is how fairlearn itself recommends handling intersectional attributes, but
this script computes it directly rather than depending on that fairlearn
usage pattern.
"""
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from fairness_checks import load_model

RESULTS_PATH = PROJECT_ROOT / "scripts" / "intersectional_fairness_results.json"

ETHNICITY_LABELS = {0: "Eth0", 1: "Eth1", 2: "Eth2"}
GENDER_LABELS = {0: "Male", 1: "Female"}
DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10


def intersectional_group_labels(X):
    ethnicity = X[:, 0].astype(int)
    gender = X[:, 1].astype(int)
    return np.array([f"{GENDER_LABELS[g]}-{ETHNICITY_LABELS[e]}" for g, e in zip(gender, ethnicity)])


def demographic_parity_spread(y_pred, groups):
    labels = np.unique(groups)
    rates = {g: y_pred[groups == g].mean() for g in labels}
    spread = max(rates.values()) - min(rates.values())
    return spread, rates


def equalized_odds_spread(y_true, y_pred, groups):
    labels = np.unique(groups)
    fpr_by_group, fnr_by_group = {}, {}
    for g in labels:
        mask = groups == g
        y_t, y_p = y_true[mask], y_pred[mask]
        negatives, positives = y_t == 0, y_t == 1
        fpr_by_group[g] = y_p[negatives].mean() if negatives.sum() > 0 else 0.0
        fnr_by_group[g] = 1 - y_p[positives].mean() if positives.sum() > 0 else 0.0
    fpr_spread = max(fpr_by_group.values()) - min(fpr_by_group.values())
    fnr_spread = max(fnr_by_group.values()) - min(fnr_by_group.values())
    return max(fpr_spread, fnr_spread), fpr_by_group, fnr_by_group


def main():
    data_path = PROJECT_ROOT / "data" / "FairCVtest" / "data" / "FairCVdb.npy"
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data["Profiles Test"]
    y_true = (data["Blind Labels Test"] > 0.5).astype(int)
    groups = intersectional_group_labels(X_test)

    group_counts = {g: int((groups == g).sum()) for g in np.unique(groups)}
    print("Intersectional subgroup sizes (n=4800 total):")
    for g, n in sorted(group_counts.items()):
        print(f"  {g}: {n}")

    models = [
        ("reference", "baseline_fair_recruiter.pkl"),
        ("gender_biased", "baseline_biased_recruiter.pkl"),
        ("ethnicity_biased", "baseline_biased_recruiter_ethnicity.pkl"),
    ]

    results = []
    print(f"\n{'model':18s} {'intersect_dp':>13s} {'intersect_eo':>13s} {'dp_pass':>8s} {'eo_pass':>8s}")
    print("-" * 65)

    for name, model_file in models:
        model = load_model(PROJECT_ROOT / "src" / "models" / model_file)
        y_pred = model.predict(X_test)

        dp_spread, dp_rates = demographic_parity_spread(y_pred, groups)
        eo_spread, fpr_by_group, fnr_by_group = equalized_odds_spread(y_true, y_pred, groups)

        dp_pass = dp_spread < DP_THRESHOLD
        eo_pass = eo_spread < EO_THRESHOLD

        print(f"{name:18s} {dp_spread:13.4f} {eo_spread:13.4f} {str(dp_pass):>8s} {str(eo_pass):>8s}")

        results.append({
            "model": name,
            "model_file": model_file,
            "intersectional_demographic_parity_spread": float(dp_spread),
            "intersectional_equalized_odds_spread": float(eo_spread),
            "dp_pass_at_0.10": bool(dp_pass),
            "eo_pass_at_0.10": bool(eo_pass),
            "group_sizes": group_counts,
            "positive_prediction_rate_by_group": {g: float(r) for g, r in dp_rates.items()},
            "fpr_by_group": {g: float(r) for g, r in fpr_by_group.items()},
            "fnr_by_group": {g: float(r) for g, r in fnr_by_group.items()},
        })

    summary = {
        "purpose": (
            "Demographic parity and equalized odds recomputed across the six "
            "gender x ethnicity intersectional subgroups, rather than one "
            "protected attribute at a time (Sections 3.2, 4.1), for all three "
            "real trained models on the real FairCVdb test split."
        ),
        "threshold": 0.10,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
