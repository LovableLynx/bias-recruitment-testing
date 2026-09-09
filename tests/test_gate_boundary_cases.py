# tests/test_gate_boundary_cases.py
"""
Direct unit tests of src/deployment_gate.py's evaluate_gate(), using
synthetic (demographic_parity_difference, equalized_odds_difference) pairs
rather than metrics computed from a trained model.

These exist because the real trained models used in test_deployment_gate.py
happen to breach both fairness metrics simultaneously whenever they breach
either one, which means those tests alone cannot distinguish "block if
either metric breaches threshold" from "block only if both metrics breach
threshold" -- the OR/AND combinator confound discussed in the paper's
Section 4.3 (mutant M2). The boundary cases here isolate each metric so
that OR-vs-AND, and the exact-threshold edge, are actually exercised by a
committed test rather than only by the mutation harness's synthetic-fixture
experiment (scripts/mutation_test_gate.py).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from deployment_gate import evaluate_gate, BLOCK, DEPLOY, DP_THRESHOLD, EO_THRESHOLD


def test_gate_blocks_when_only_dp_breaches():
    metrics = {"demographic_parity_difference": 0.146, "equalized_odds_difference": 0.05}
    assert evaluate_gate(metrics) == BLOCK


def test_gate_blocks_when_only_eo_breaches():
    metrics = {"demographic_parity_difference": 0.05, "equalized_odds_difference": 0.607}
    assert evaluate_gate(metrics) == BLOCK


def test_gate_blocks_when_both_breach():
    metrics = {"demographic_parity_difference": 0.146, "equalized_odds_difference": 0.607}
    assert evaluate_gate(metrics) == BLOCK


def test_gate_deploys_when_neither_breaches():
    metrics = {"demographic_parity_difference": 0.02, "equalized_odds_difference": 0.03}
    assert evaluate_gate(metrics) == DEPLOY


def test_gate_blocks_at_exact_dp_threshold():
    metrics = {"demographic_parity_difference": DP_THRESHOLD, "equalized_odds_difference": 0.0}
    assert evaluate_gate(metrics) == BLOCK, "Threshold comparison is >=, so the exact threshold value must BLOCK"


def test_gate_blocks_at_exact_eo_threshold():
    metrics = {"demographic_parity_difference": 0.0, "equalized_odds_difference": EO_THRESHOLD}
    assert evaluate_gate(metrics) == BLOCK, "Threshold comparison is >=, so the exact threshold value must BLOCK"


def test_gate_deploys_just_below_both_thresholds():
    metrics = {"demographic_parity_difference": DP_THRESHOLD - 0.001, "equalized_odds_difference": EO_THRESHOLD - 0.001}
    assert evaluate_gate(metrics) == DEPLOY
