# tests/test_compliance_gate.py
"""
Compliance gate tests on real fixtures: src/compliance_gate.py's
evaluate_compliance_gate() applied to per-group selection counts from the
three trained FairCVdb models. Ground truth: each biased model selects its
disadvantaged group at well under four-fifths of the advantaged group's rate
(gender-biased model by gender, ethnicity-biased model by ethnicity) and must
be BLOCKed; the reference model must be DEPLOYed.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_selection_counts
from compliance_gate import evaluate_compliance_gate, BLOCK, DEPLOY


def test_gender_model_fails_four_fifths(test_data):
    X_test, _ = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl')
    counts = compute_selection_counts(model, X_test, gender)
    decision = evaluate_compliance_gate(counts)
    assert decision == BLOCK, f"Gate should have BLOCKED the gender-biased model but returned {decision} ({counts})"


def test_ethnicity_model_fails_four_fifths(test_data):
    X_test, _ = test_data
    ethnicity = X_test[:, 0]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter_ethnicity.pkl')
    counts = compute_selection_counts(model, X_test, ethnicity)
    decision = evaluate_compliance_gate(counts)
    assert decision == BLOCK, f"Gate should have BLOCKED the ethnicity-biased model but returned {decision} ({counts})"


def test_reference_model_passes_four_fifths(test_data):
    X_test, _ = test_data
    gender = X_test[:, 1]
    model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')
    counts = compute_selection_counts(model, X_test, gender)
    decision = evaluate_compliance_gate(counts)
    assert decision == DEPLOY, f"Gate should have DEPLOYed the reference model but returned {decision} ({counts})"
