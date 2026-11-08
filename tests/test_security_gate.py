# tests/test_security_gate.py
"""
Security gate tests on real fixtures: src/security_gate.py's
evaluate_security_gate() applied to frozen OSV scans of three real pinned
dependency sets (data/security_scans/, produced by
scripts/snapshot_osv_scans.py). Ground truth is fixed by the policy: a set
with any CRITICAL advisory, or any HIGH advisory, must be BLOCKed; this
project's own CI dependencies, which had no advisories on the snapshot date,
must be DEPLOYed.
"""
import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
from security_gate import evaluate_security_gate, BLOCK, DEPLOY


def load_scan(name):
    report = json.loads((PROJECT_ROOT / 'data' / 'security_scans' / f'scan_{name}.json').read_text(encoding='utf-8'))
    return report, datetime.date.fromisoformat(report['snapshot_date'])


def test_legacy_imaging_set_is_blocked():
    """CRITICAL advisories present: must BLOCK."""
    report, today = load_scan('legacy_imaging')
    decision = evaluate_security_gate(report, waivers={}, today=today)
    assert decision == BLOCK, f"Gate should have BLOCKED legacy_imaging but returned {decision}"


def test_legacy_web_set_is_blocked():
    """HIGH advisories but no CRITICAL: must still BLOCK (MAX_HIGH = 0)."""
    report, today = load_scan('legacy_web')
    decision = evaluate_security_gate(report, waivers={}, today=today)
    assert decision == BLOCK, f"Gate should have BLOCKED legacy_web but returned {decision}"


def test_project_ci_set_is_deployable():
    """No advisories on the snapshot date: must DEPLOY."""
    report, today = load_scan('project_ci')
    decision = evaluate_security_gate(report, waivers={}, today=today)
    assert decision == DEPLOY, f"Gate should have DEPLOYed project_ci but returned {decision}"
