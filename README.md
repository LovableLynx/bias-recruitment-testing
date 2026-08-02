
# Bias as a Test Case: A Software Testing Framework for Continuous Fairness Assertions in AI Recruitment Pipelines

Proof-of-concept research project exploring fairness detection as an automated,
CI/CD-integrated software testing artifact — treating bias checks as pytest
assertions rather than standalone data-science analysis.

## Overview

This project trains baseline recruitment-scoring models on the [FairCVdb](https://github.com/BiDAlab/FairCVtest)
dataset (Peña et al., 2020) — one trained on gender-biased labels, one on
ethnicity-biased labels, one on unbiased ("blind") labels — then wraps
fairness metrics (demographic parity, equalized odds) in an automated pytest
suite that runs on every push via GitHub Actions.

Two claims are tested:

1. **Detection accuracy** (`tests/test_fairness.py`) — confirms the metrics
   correctly identify known-biased models as biased and known-fair models as fair.
2. **Deployment gate** (`tests/test_deployment_gate.py`) — simulates a CI/CD
   quality gate that blocks a biased model from passing, the way a failed
   unit test blocks a bad deploy.

## Project structure

```
bias-recruitment-testing/
├── data/FairCVtest/       # FairCVdb dataset (via Git LFS)
├── src/
│   ├── fairness_checks.py # fairness + performance metric helpers
│   └── models/             # saved baseline models (.pkl)
├── notebooks/               # exploratory data analysis
├── tests/
│   ├── conftest.py          # shared test-data fixture
│   ├── test_fairness.py     # Claim 1: detection accuracy
│   └── test_deployment_gate.py  # Claim 2: deployment gate
└── .github/workflows/       # CI pipeline definition
```

## Dataset

[FairCVdb](https://github.com/BiDAlab/FairCVtest) — 24,000 synthetic resume
profiles with blind (unbiased) and biased (gender/ethnicity) scores, tracked
via Git LFS.

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
