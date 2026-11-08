"""
Compliance gate: translates per-group selection counts into a deploy/block
decision under the "four-fifths rule" for adverse impact (US Uniform
Guidelines on Employee Selection Procedures, 29 CFR 1607.4(D)): a group whose
selection rate is less than four-fifths (80%) of the highest group's rate is
generally regarded as evidence of adverse impact.

Third case study for the paper's cross-domain replication (Section 4.8).
Like src/deployment_gate.py, this is a plain function independent of any test
framework, so its enforcement logic can be called and mutated on its own.

Policy:
  - BLOCK if fewer than two groups are supplied, if any group has fewer than
    MIN_GROUP_SIZE candidates, or if no group has any selections (fail closed:
    the ratio cannot be relied on);
  - BLOCK if the lowest group selection rate divided by the highest is less
    than IMPACT_RATIO_THRESHOLD (0.8); a ratio of exactly 0.8 is not "less than
    four-fifths" and so does not block;
  - otherwise DEPLOY.

The minimum-group-size rule is this project's own policy choice, not part of
29 CFR 1607.4(D).
"""

IMPACT_RATIO_THRESHOLD = 0.8
MIN_GROUP_SIZE = 30

BLOCK = "BLOCK"
DEPLOY = "DEPLOY"


def impact_ratio(group_counts):
    """Lowest selection rate divided by highest, from {group: (selected, total)}."""
    rates = [selected / total for selected, total in group_counts.values()]
    return min(rates) / max(rates)


def evaluate_compliance_gate(group_counts, ratio_threshold=IMPACT_RATIO_THRESHOLD, min_group_size=MIN_GROUP_SIZE):
    """Return BLOCK or DEPLOY for {group: (selected, total)} selection counts."""
    if len(group_counts) < 2:
        return BLOCK
    if any(total < min_group_size for _, total in group_counts.values()):
        return BLOCK
    if max(selected for selected, _ in group_counts.values()) == 0:
        return BLOCK
    if impact_ratio(group_counts) < ratio_threshold:
        return BLOCK
    return DEPLOY
