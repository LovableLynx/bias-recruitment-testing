"""
Security gate: translates a dependency-vulnerability scan into a deploy/block
decision.

Second case study for the paper's cross-domain replication (Section 4.8).
Like src/deployment_gate.py, this is a plain function independent of any test
framework, so its enforcement logic can be called and mutated on its own,
separately from tests/test_security_gate.py and
tests/test_security_gate_boundary_cases.py.

Policy (a common shape for CI vulnerability gates):
  - BLOCK if the scan report is missing or malformed (fail closed);
  - a finding is ignored only while a waiver for its id is in force
    (waiver expiry date on or after the evaluation date);
  - BLOCK if any remaining finding is CRITICAL, or if the number of remaining
    HIGH findings exceeds MAX_HIGH;
  - otherwise DEPLOY. MODERATE and LOW findings never block on their own.
"""
import datetime

MAX_HIGH = 0

BLOCK = "BLOCK"
DEPLOY = "DEPLOY"


def is_waived(finding, waivers, today):
    """True if `finding` has a waiver whose expiry date is on or after `today`."""
    expiry = waivers.get(finding["id"])
    if expiry is None:
        return False
    return datetime.date.fromisoformat(expiry) >= today


def evaluate_security_gate(report, waivers=None, today=None, max_high=MAX_HIGH):
    """Return BLOCK or DEPLOY for a scan report of the form {"findings": [...]}."""
    if report is None or "findings" not in report:
        return BLOCK
    waivers = waivers or {}
    today = today or datetime.date.today()
    active = [f for f in report["findings"] if not is_waived(f, waivers, today)]
    critical = sum(1 for f in active if f["severity"] == "CRITICAL")
    high = sum(1 for f in active if f["severity"] == "HIGH")
    if critical > 0 or high > max_high:
        return BLOCK
    return DEPLOY
