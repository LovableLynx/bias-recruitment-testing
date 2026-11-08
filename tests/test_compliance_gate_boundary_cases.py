# tests/test_compliance_gate_boundary_cases.py
"""
Direct unit tests of src/compliance_gate.py on synthetic selection counts,
isolating the exact four-fifths edge, more than two groups, and each
fail-closed branch, which the three real models alone may not separate.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from compliance_gate import evaluate_compliance_gate, BLOCK, DEPLOY, MIN_GROUP_SIZE


def test_gate_blocks_below_four_fifths():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (39, 100)}) == BLOCK


def test_gate_deploys_at_exact_four_fifths():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (40, 100)}) == DEPLOY, "Exactly four-fifths is not less than four-fifths"


def test_gate_deploys_above_four_fifths():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (45, 100)}) == DEPLOY


def test_gate_uses_lowest_of_three_groups():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (45, 100), "c": (30, 100)}) == BLOCK


def test_gate_blocks_on_small_group():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (10, MIN_GROUP_SIZE - 1)}) == BLOCK, "Too few candidates must fail closed"


def test_gate_blocks_on_single_group():
    assert evaluate_compliance_gate({"a": (50, 100)}) == BLOCK, "One group cannot be compared"


def test_gate_blocks_when_nobody_selected():
    assert evaluate_compliance_gate({"a": (0, 100), "b": (0, 100)}) == BLOCK, "No selections must fail closed"


# ---------------------------------------------------------------------------
# Added after the first mutation run (scripts/cross_gate_mutation_results_run1.json
# and mutation_study/cosmic_ray_dump_compliance.jsonl): each test below closes a
# gap that a surviving hand-selected or tool-generated mutant exposed. See paper
# Section 4.8.
# ---------------------------------------------------------------------------

def test_gate_blocks_on_small_group_even_when_ratio_passes():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (14, 29)}) == BLOCK, "A small group must fail closed even if its ratio would pass"


def test_gate_deploys_at_exact_minimum_group_size():
    assert evaluate_compliance_gate({"a": (15, 30), "b": (14, 30)}) == DEPLOY, "Policy: 30 candidates is enough"


def test_gate_compares_rates_not_counts():
    assert evaluate_compliance_gate({"a": (45, 50), "b": (90, 100)}) == DEPLOY, "Equal rates in groups of different size must DEPLOY"


def test_gate_deploys_three_groups_within_four_fifths():
    assert evaluate_compliance_gate({"a": (50, 100), "b": (45, 100), "c": (42, 100)}) == DEPLOY
