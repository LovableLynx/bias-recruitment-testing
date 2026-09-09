
# Bias as a Test Case: A Software Testing Framework for Continuous Fairness Assertions in AI Recruitment Pipelines

Proof-of-concept research project exploring fairness detection as an automated,
CI-integrated software testing artifact — treating bias checks as pytest
assertions rather than standalone data-science analysis.

## Overview

This project trains baseline recruitment-scoring models on the [FairCVdb](https://github.com/BiDAlab/FairCVtest)
dataset (Peña et al., 2020) — one trained on gender-biased labels, one on
ethnicity-biased labels, one on blind-label ("reference") labels — then wraps
fairness metrics (demographic parity, equalized odds) in an automated pytest
suite that runs on every push via GitHub Actions. All three models are trained
on the same feature set, which includes the raw gender/ethnicity columns; only
the training *labels* differ in how they were constructed. See the paper for
why this matters and why the blind-label model is called a "reference" model
rather than "fair."

Three things are tested:

1. **Fairness classification** (`tests/test_fairness.py`) — confirms the metrics
   correctly classify known-biased models as biased and the reference model as
   satisfying the study's fairness threshold on known fixtures. This does not
   independently verify the correctness of the underlying `fairlearn`/`aequitas` metric implementations
   themselves, which are treated as trusted instruments.
2. **Deployment gate** (`src/deployment_gate.py`, tested by
   `tests/test_deployment_gate.py`) — `evaluate_gate(metrics)` is a standalone
   function, independent of any test framework, that returns `BLOCK` or `DEPLOY`.
   Separating it from the tests that check it means the gate's own logic can be
   mutated on its own (see below), not just the assertions around it. This is
   CI, not CD: no deployment job is actually gated on this workflow's result.
3. **Mutation testing** (`scripts/mutation_test_gate.py`) — injects representative
   faults either into `deployment_gate.py` itself (checked by the real,
   unmodified tests) or into the tests (checked against the real, unmodified
   gate), and records whether each is caught; see `scripts/mutation_results.json`
   for results.

## Project structure

```
bias-recruitment-testing/
├── data/FairCVtest/       # FairCVdb dataset (via Git LFS)
├── src/
│   ├── fairness_checks.py # fairness + performance metric helpers
│   ├── deployment_gate.py # evaluate_gate(): metrics -> BLOCK / DEPLOY
│   └── models/             # saved baseline models (.pkl)
├── notebooks/               # exploratory data analysis
├── scripts/
│   └── mutation_test_gate.py  # fault-injection tests of the gate and its test suite
├── tests/
│   ├── conftest.py          # shared test-data fixture
│   ├── test_fairness.py     # Claim 1: fairness classification
│   └── test_deployment_gate.py  # Claim 2: deployment gate
└── .github/workflows/       # CI pipeline definition
```

## Dataset

[FairCVdb](https://github.com/BiDAlab/FairCVtest) — 24,000 synthetic resume
profiles with blind-label and biased-label (gender/ethnicity) scores, tracked
via Git LFS. "Blind" describes how the training label was constructed (without
a demographic penalty term), not which features the model sees at inference
time — all profiles, including those scored by the blind-label model, still
include raw gender/ethnicity columns as input features.

## Fairness metrics

- **Demographic parity difference** — selection-rate gap across groups
- **Equalized odds difference** — error-rate (FPR/FNR) gap across groups
- Threshold: 0.10 for both metrics

## Running locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pytest tests/ -v
```

## Status

Proof-of-concept, part of pre-PhD research for a Software Engineering PhD
proposal on continuous bias detection in AI recruitment systems.

## Citation

Peña, A., Serna, I., Morales, A., & Fierrez, J. (2020). Bias in Multimodal
AI: Testbed for Fair Automatic Recruitment. *CVPR Workshops*.
