"""
Deployment gate: translates fairness metrics into a deploy/block decision.

This is deliberately a plain function independent of any test framework,
so that its enforcement logic can be exercised, called, and mutated on its
own — separately from the pytest assertions in tests/test_deployment_gate.py
that check its behavior against known fixtures.
"""

DP_THRESHOLD = 0.10
EO_THRESHOLD = 0.10

BLOCK = "BLOCK"
DEPLOY = "DEPLOY"


def evaluate_gate(metrics, dp_threshold=DP_THRESHOLD, eo_threshold=EO_THRESHOLD):
    """Return BLOCK if either fairness metric meets or exceeds its threshold, else DEPLOY."""
    if (
        metrics["demographic_parity_difference"] >= dp_threshold
        and metrics["equalized_odds_difference"] >= eo_threshold
    ):
        return BLOCK
    return DEPLOY
