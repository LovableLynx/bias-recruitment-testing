# tests/test_security_gate_boundary_cases.py
"""
Direct unit tests of src/security_gate.py on synthetic findings, isolating
each branch of the policy (single-condition breach, the high-count limit,
waiver expiry on the evaluation date, fail-closed on a missing report) that
the three real dependency sets alone may not separate.
"""
import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from security_gate import evaluate_security_gate, BLOCK, DEPLOY, MAX_HIGH

TODAY = datetime.date(2026, 11, 8)


def finding(fid, severity):
    return {"id": fid, "package": "example", "version": "1.0", "severity": severity}


def test_gate_blocks_when_only_critical_present():
    report = {"findings": [finding("GHSA-c", "CRITICAL")]}
    assert evaluate_security_gate(report, {}, TODAY) == BLOCK


def test_gate_blocks_when_high_exceeds_limit():
    report = {"findings": [finding(f"GHSA-h{i}", "HIGH") for i in range(MAX_HIGH + 1)]}
    assert evaluate_security_gate(report, {}, TODAY) == BLOCK


def test_gate_deploys_when_only_moderate_and_low():
    report = {"findings": [finding("GHSA-m", "MODERATE"), finding("GHSA-l", "LOW")]}
    assert evaluate_security_gate(report, {}, TODAY) == DEPLOY


def test_gate_deploys_with_high_count_equal_to_limit():
    report = {"findings": [finding("GHSA-h1", "HIGH"), finding("GHSA-h2", "HIGH")]}
    assert evaluate_security_gate(report, {}, TODAY, max_high=2) == DEPLOY, "A HIGH count equal to the limit must not BLOCK"


def test_gate_honours_waiver_expiring_today():
    report = {"findings": [finding("GHSA-c", "CRITICAL")]}
    assert evaluate_security_gate(report, {"GHSA-c": "2026-11-08"}, TODAY) == DEPLOY, "A waiver is in force through its expiry date"


def test_gate_ignores_expired_waiver():
    report = {"findings": [finding("GHSA-c", "CRITICAL")]}
    assert evaluate_security_gate(report, {"GHSA-c": "2026-11-07"}, TODAY) == BLOCK


def test_gate_blocks_on_missing_report():
    assert evaluate_security_gate(None, {}, TODAY) == BLOCK, "A missing scan must fail closed"


# ---------------------------------------------------------------------------
# Added after the first mutation run (scripts/cross_gate_mutation_results_run1.json
# and mutation_study/cosmic_ray_dump_security.jsonl): each test below closes a
# gap that a surviving tool-generated mutant exposed. See paper Section 4.8.
# ---------------------------------------------------------------------------

def test_gate_deploys_with_high_count_below_limit():
    report = {"findings": [finding("GHSA-h1", "HIGH")]}
    assert evaluate_security_gate(report, {}, TODAY, max_high=2) == DEPLOY, "A HIGH count below the limit must not BLOCK"


def test_gate_blocks_on_one_high_with_default_limit():
    report = {"findings": [finding("GHSA-h1", "HIGH")]}
    assert evaluate_security_gate(report, {}, TODAY) == BLOCK, "Policy: the default limit is zero HIGH findings"


def test_gate_honours_waiver_expiring_later():
    report = {"findings": [finding("GHSA-c", "CRITICAL")]}
    assert evaluate_security_gate(report, {"GHSA-c": "2027-01-31"}, TODAY) == DEPLOY


def test_gate_uses_the_given_evaluation_date():
    report = {"findings": [finding("GHSA-c", "CRITICAL")]}
    later = datetime.date(2030, 1, 1)
    assert evaluate_security_gate(report, {"GHSA-c": "2029-12-31"}, later) == BLOCK, "Waiver expiry is judged on the date passed in, not the system clock"


def test_real_critical_findings_alone_are_blocked():
    import json
    report = json.loads((PROJECT_ROOT / 'data' / 'security_scans' / 'scan_legacy_imaging.json').read_text(encoding='utf-8'))
    critical_only = {"findings": [f for f in report["findings"] if f["severity"] == "CRITICAL"]}
    assert critical_only["findings"], "fixture must contain CRITICAL findings"
    assert evaluate_security_gate(critical_only, {}, TODAY) == BLOCK
