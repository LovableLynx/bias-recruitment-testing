
# Testing the Fairness Gate: Separating Fairness Classification from Enforcement Correctness in Continuous Fairness Testing for AI Recruitment Pipelines

Proof-of-concept research project exploring fairness detection as an automated,
CI-integrated software testing artifact, treating bias checks as pytest
assertions rather than standalone data-science analysis.

## Overview

This project trains baseline recruitment-scoring models on the [FairCVdb](https://github.com/BiDAlab/FairCVtest)
dataset (Peña et al., 2020), one trained on gender-biased labels, one on
ethnicity-biased labels, one on blind-label ("reference") labels, then wraps
fairness metrics (demographic parity, equalized odds) in an automated pytest
suite that runs on every push via GitHub Actions. All three models are trained
on the same feature set, which includes the raw gender/ethnicity columns; only
the training *labels* differ in how they were constructed. See the paper for
why this matters and why the blind-label model is called a "reference" model
rather than "fair."

**What is tested (numbering follows the paper's own Results sections):**

- **4.1 Fairness classification** (`tests/test_fairness.py`): confirms the
  metrics correctly classify known-biased models as biased and the reference
  model as satisfying the study's fairness threshold on known fixtures.
  Extended by `scripts/verify_metric_formulas.py` and
  `scripts/verify_aequitas_formulas.py`, which re-derive fairlearn's and
  aequitas's metric formulas from their textbook definitions using plain array
  arithmetic and compare the result against each library's own output, rather
  than treating either library as an unchecked black box.
- **4.2 Deployment-gate enforcement** (`src/deployment_gate.py`, tested by
  `tests/test_deployment_gate.py` and `tests/test_gate_boundary_cases.py`):
  `evaluate_gate(metrics)` is a standalone function, independent of any test
  framework, that returns `BLOCK` or `DEPLOY`. Separating it from the tests
  that check it means the gate's own logic can be mutated on its own (see
  below), not just the assertions around it. `test_gate_boundary_cases.py`
  adds direct unit tests on synthetic metric values (single-metric breach,
  exact-threshold edge) that the three real trained models alone don't
  exercise.
- **4.3 Mutation testing** (`scripts/mutation_test_gate.py`): injects seven
  hand-selected faults either into `deployment_gate.py` itself (checked by
  the real, unmodified tests) or into the tests (checked against the real,
  unmodified gate), and records whether each is caught; see
  `scripts/mutation_results.json` for results.
- **4.4 Tool-generated mutation testing**: a `cosmic-ray` pass against
  `deployment_gate.py` alone (20 automatically generated mutants; results in
  `mutation_study/cosmic_ray_results.txt`), checking that the hand-selected
  mutants above weren't cherry-picked.
- **4.5 Systematic test-suite-adequacy sweep**
  (`scripts/systematic_test_adequacy_mutants.py`): a fixed, content-blind
  mutation rule applied mechanically to every test function in the suite (30
  candidates, 23 applicable), extending the hand-selected test-suite mutants
  above to the full test-suite fault space; results in
  `scripts/systematic_test_adequacy_results.json`.
- **4.6 Feature-blinded and intersectional re-evaluation**
  (`scripts/blinded_feature_experiment.py`,
  `scripts/intersectional_fairness_check.py`): retrains all three models with
  the sensitive-attribute columns removed, and recomputes both fairness
  metrics across gender-by-ethnicity subgroups rather than one attribute at a
  time.
- **4.7 Real enforcement check for the CI-only gap**: for most of this project
  the workflow was CI only, with no downstream job gated on the suite's
  result. A `deploy-approval` job was added that runs only when the
  fairness/gate suite passes; verified with a real run where it was correctly
  skipped after a deliberately injected fault (see the workflow file).
- **Section 8, CI/mutation-testing cost** (`scripts/measure_ci_cost.py`,
  `scripts/measure_ci_cost_repeated.py`): real wall-clock timing of the
  committed suite versus the two mutation-testing passes above, supporting
  the paper's recommendation to run mutation testing as a periodic audit
  rather than on every commit.

## Project structure

```
bias-recruitment-testing/
├── data/FairCVtest/       # FairCVdb dataset (via Git LFS)
├── src/
│   ├── fairness_checks.py # fairness + performance metric helpers
│   ├── aequitas_checks.py # aequitas disparity-ratio audit helper
│   ├── deployment_gate.py # evaluate_gate(): metrics -> BLOCK / DEPLOY
│   └── models/             # saved baseline models (.pkl)
├── notebooks/               # exploratory data analysis
├── scripts/
│   ├── mutation_test_gate.py               # 4.3: hand-selected fault injection
│   ├── mutation_results.json               #      raw per-mutant results
│   ├── systematic_test_adequacy_mutants.py # 4.5: rule-generated test-suite sweep
│   ├── systematic_test_adequacy_results.json
│   ├── verify_metric_formulas.py           # 4.1: fairlearn formula re-derivation
│   ├── metric_verification_results.json
│   ├── verify_aequitas_formulas.py         # 4.1: aequitas formula re-derivation
│   ├── aequitas_verification_results.json
│   ├── blinded_feature_experiment.py       # 4.6: feature-blinded retraining
│   ├── blinded_feature_results.json
│   ├── intersectional_fairness_check.py    # 4.6: gender x ethnicity subgroup check
│   ├── intersectional_fairness_results.json
│   ├── measure_ci_cost.py                  # Section 8: single-run CI/mutation timing
│   ├── measure_ci_cost_repeated.py         #            repeated-run timing (mean/range)
│   └── ci_cost_results.json, ci_cost_repeated_results.json
├── mutation_study/
│   ├── cr_config.toml            # 4.4: cosmic-ray configuration
│   └── cosmic_ray_results.txt    #      raw tool-generated mutation results
├── tests/
│   ├── conftest.py                 # shared test-data fixture
│   ├── test_fairness.py            # fairness classification
│   ├── test_deployment_gate.py     # deployment-gate enforcement (real models)
│   ├── test_gate_boundary_cases.py # deployment-gate enforcement (synthetic boundary cases)
│   └── test_aequitas_crossvalidation.py  # independent aequitas cross-check
└── .github/workflows/       # CI pipeline: fairness-check + deploy-approval (4.7)
```

## Dataset

[FairCVdb](https://github.com/BiDAlab/FairCVtest): 24,000 synthetic resume
profiles with blind-label and biased-label (gender/ethnicity) scores, tracked
via Git LFS. "Blind" describes how the training label was constructed (without
a demographic penalty term), not which features the model sees at inference
time; all profiles, including those scored by the blind-label model, still
include raw gender/ethnicity columns as input features.

## Fairness metrics

- **Demographic parity difference**: selection-rate gap across groups
- **Equalized odds difference**: error-rate (FPR/FNR) gap across groups
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

## License

The code in this repository (`src/`, `tests/`, `scripts/`, and the CI
workflow) is released under the [MIT License](LICENSE). This covers only
code written for this project; it does not cover third-party dependencies
such as fairlearn, aequitas, scikit-learn, or the FairCVdb dataset, each of
which is separately licensed by its own authors and installed via
`requirements.txt` rather than vendored into this repository.
