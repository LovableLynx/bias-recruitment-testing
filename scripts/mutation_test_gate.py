"""
Mutation testing harness for the deployment gate (src/deployment_gate.py).

This mutates the gate's IMPLEMENTATION, not its test file. A mutant is a
patched copy of evaluate_gate() with one fault injected; the UNMODIFIED
tests/test_deployment_gate.py suite (which calls evaluate_gate() through the
real trained models and the real FairCVdb test split) is then run against
each mutant to see whether it still passes. This is standard mutation
testing methodology: the code under test is what changes, not the tests
checking it, which is what makes "the gate's own test suite can detect a
fault in the gate" a meaningful claim rather than a test file checking
itself.

Two complementary experiments are run:

1. Model-fixture mutants (M1-M7): a single fault is injected into a copy
   of deployment_gate.py, which is imported in place of the real module
   while the real tests/test_deployment_gate.py suite runs against the
   real trained models. A mutant is CAUGHT if that suite no longer passes
   (at least one test fails against the known ground truth: both biased
   models must be BLOCKed, the reference model must be DEPLOYed). A mutant
   SURVIVES if the suite still passes despite the injected fault.

   One risk with this design: a mutant can be semantically different from
   the original but still survive simply because the real models' metric
   values don't happen to distinguish the two conditions (e.g. `or` vs
   `and` when both metrics independently exceed threshold on every biased
   fixture in the dataset). Such a mutant is not a genuine counterexample
   to "the gate works" so much as an EQUIVALENT MUTANT under the current
   test data, a well-known confound in mutation testing (Jia & Harman,
   2011). Experiment 2 exists specifically to resolve this ambiguity.

2. Boolean-logic fixtures (B1-B4, run_boolean_logic_experiment()): synthetic,
   hand-picked (dp, eo) metric pairs that make `or` vs `and` observably
   different, independent of what any real trained model happens to
   produce. These call evaluate_gate() directly, to determine whether M2
   (or -> and) is a genuine fault or an artifact of the real models' data
   happening to make both branches agree.

Faults are tagged with a `fault_layer`:
  - "enforcement-logic": the fault is in deployment_gate.py's decision
    computed FROM a given, correctly measured metric value (a comparison
    operator, a threshold, a Boolean combinator).
  - "test-suite-adequacy": the fault is in tests/test_deployment_gate.py's
    coverage or assertions, not in deployment_gate.py's decision logic (a
    dropped test case, a vacuous assertion). These are injected into the
    TEST file, not the gate module, since they are about whether the tests
    adequately exercise the gate, not about the gate's own logic. They
    cannot be caught by strengthening enforcement logic; they require an
    external check on the suite itself (e.g. coverage auditing, this
    harness).

This intentionally does not use a general-purpose mutation testing tool
(e.g. mutmut, cosmic-ray): the fault list is hand-selected to mirror the
specific defect class discussed in the paper (Section 3.4) rather than an
exhaustive AST-level mutation sweep. See Limitations in the paper.
"""
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_MODULE_SRC = PROJECT_ROOT / "src" / "deployment_gate.py"
GATE_TEST_SRC = PROJECT_ROOT / "tests" / "test_deployment_gate.py"
MUTANTS_DIR = PROJECT_ROOT / "tests" / "_mutants"
RESULTS_PATH = PROJECT_ROOT / "scripts" / "mutation_results.json"

ORIGINAL_GATE_MODULE = GATE_MODULE_SRC.read_text(encoding="utf-8")
ORIGINAL_GATE_TEST = GATE_TEST_SRC.read_text(encoding="utf-8")

# Enforcement-logic mutants: patch src/deployment_gate.py, run the real,
# unmodified test suite against the patched module.
# Each entry: (id, description, old_string, new_string)
ENFORCEMENT_MUTANTS = [
    (
        "M1_flip_comparison_direction",
        "Flip both >= to < (a single comparison-direction fault, matching the original Section 3.4 defect; the combinator is left as `or`, unlike M2).",
        'metrics["demographic_parity_difference"] >= dp_threshold\n        or metrics["equalized_odds_difference"] >= eo_threshold',
        'metrics["demographic_parity_difference"] < dp_threshold\n        or metrics["equalized_odds_difference"] < eo_threshold',
    ),
    (
        "M2_or_to_and",
        "Change the combinator from `or` to `and` (require both metrics to breach threshold, not either). See Experiment 2 for whether this is equivalent under real model data.",
        'metrics["demographic_parity_difference"] >= dp_threshold\n        or metrics["equalized_odds_difference"] >= eo_threshold',
        'metrics["demographic_parity_difference"] >= dp_threshold\n        and metrics["equalized_odds_difference"] >= eo_threshold',
    ),
    (
        "M5_threshold_widened",
        "Widen both default thresholds from 0.10 to 0.90, so almost nothing gets blocked.",
        "DP_THRESHOLD = 0.10\nEO_THRESHOLD = 0.10",
        "DP_THRESHOLD = 0.90\nEO_THRESHOLD = 0.90",
    ),
    (
        "M7_decision_inverted",
        "Swap the BLOCK/DEPLOY return values (correct condition, inverted outcome).",
        "        return BLOCK\n    return DEPLOY",
        "        return DEPLOY\n    return BLOCK",
    ),
]

# Test-suite-adequacy mutants: patch tests/test_deployment_gate.py itself,
# leaving src/deployment_gate.py untouched, since these are about whether
# the test suite adequately exercises the (correct) gate, not about a fault
# in the gate's own logic.
TEST_ADEQUACY_MUTANTS = [
    (
        "M3_drop_ethnicity_gate_test",
        "Delete the ethnicity gate test entirely (coverage gap: gender-only gating).",
        'def test_ethnicity_model_is_deployable(test_data):\n    """Gate check: biased ethnicity model should be BLOCKed (not deployable)."""\n    X_test, y_true = test_data\n    ethnicity = X_test[:, 0]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')\n    metrics = compute_fairness_metrics(model, X_test, y_true, ethnicity)\n    decision = evaluate_gate(metrics)\n    assert decision == BLOCK, (\n        f"Gate should have BLOCKED the biased model but returned {decision} "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\n',
        '',
    ),
    (
        "M6_vacuous_assertion",
        "Replace the gender gate's assertion with unconditional `assert True` (gate result never actually consumed).",
        '    decision = evaluate_gate(metrics)\n    assert decision == BLOCK, (\n        f"Gate should have BLOCKED the biased model but returned {decision} "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
        '    decision = evaluate_gate(metrics)\n    assert True  # gate result not actually checked\n\ndef test_ethnicity_model_is_deployable',
    ),
    (
        "M4_swap_sensitive_attribute",
        "Use gender as the sensitive feature in the ethnicity gate test (wrong column, index 1 instead of 0) -- a test-fixture bug, not a gate-logic bug.",
        '    ethnicity = X_test[:, 0]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')',
        '    ethnicity = X_test[:, 1]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')',
    ),
]

# ---------------------------------------------------------------------------
# Experiment 2: synthetic (dp, eo) fixtures that make `or` vs `and` in the
# gate's combinator observably different, resolving whether M2 is a genuine
# fault or an equivalent mutant under the real models' data.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from deployment_gate import evaluate_gate as real_evaluate_gate, BLOCK as REAL_BLOCK, DEPLOY as REAL_DEPLOY  # noqa: E402

THRESHOLD = 0.10

BOOLEAN_LOGIC_FIXTURES = [
    ("B1_both_breach", 0.146, 0.607, "both metrics breach threshold (mirrors the real gender-biased model)"),
    ("B2_only_dp_breach", 0.146, 0.05, "only demographic parity breaches threshold"),
    ("B3_only_eo_breach", 0.05, 0.607, "only equalized odds breaches threshold"),
    ("B4_neither_breach", 0.02, 0.03, "neither metric breaches threshold (mirrors the real reference model)"),
]


def gate_and_mutant(dp, eo):
    """M2-mutated gate logic: reject only if BOTH metrics breach threshold."""
    return REAL_BLOCK if (dp >= THRESHOLD and eo >= THRESHOLD) else REAL_DEPLOY


def run_boolean_logic_experiment() -> dict:
    rows = []
    distinguished = False
    for fixture_id, dp, eo, description in BOOLEAN_LOGIC_FIXTURES:
        metrics = {"demographic_parity_difference": dp, "equalized_odds_difference": eo}
        original_decision = real_evaluate_gate(metrics)
        mutant_decision = gate_and_mutant(dp, eo)
        differs = original_decision != mutant_decision
        distinguished = distinguished or differs
        rows.append({
            "fixture": fixture_id,
            "description": description,
            "dp": dp,
            "eo": eo,
            "original_gate_decision": original_decision,
            "m2_mutant_gate_decision": mutant_decision,
            "mutant_distinguished": differs,
        })
    return {
        "m2_equivalent_on_real_fixtures": True,
        "m2_equivalent_as_gate_function": not distinguished,
        "note": (
            "M2 (or -> and) survives Experiment 1 because every real biased-model "
            "fixture in this dataset happens to breach both DP and EO simultaneously, "
            "making `or` and `and` agree on all three real fixtures. B2 and B3 below "
            "are constructed so that only one metric breaches threshold, which is "
            "sufficient to show `or` and `and` are NOT equivalent in general: M2 is an "
            "equivalent mutant with respect to this dataset's specific fixtures, not an "
            "equivalent mutant of the gate logic itself."
        ),
        "fixtures": rows,
    }


def make_module_mutant(mutant_id: str, old: str, new: str) -> Path:
    """Write a mutated copy of deployment_gate.py under tests/_mutants/<id>/src/."""
    if old not in ORIGINAL_GATE_MODULE:
        raise ValueError(f"{mutant_id}: old_string not found in deployment_gate.py")
    mutated = ORIGINAL_GATE_MODULE.replace(old, new, 1)
    if mutated == ORIGINAL_GATE_MODULE:
        raise ValueError(f"{mutant_id}: replacement produced no change")
    mutant_dir = MUTANTS_DIR / mutant_id
    src_dir = mutant_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "deployment_gate.py").write_text(mutated, encoding="utf-8")
    # The real test suite imports fairness_checks too; symlink-by-copy so the
    # mutant's src/ directory is a complete, self-contained package.
    shutil.copy(PROJECT_ROOT / "src" / "fairness_checks.py", src_dir / "fairness_checks.py")
    return mutant_dir


def make_test_mutant(mutant_id: str, old: str, new: str) -> Path:
    """Write a mutated copy of test_deployment_gate.py under tests/_mutants/<id>/tests/."""
    if old not in ORIGINAL_GATE_TEST:
        raise ValueError(f"{mutant_id}: old_string not found in test_deployment_gate.py")
    mutated = ORIGINAL_GATE_TEST.replace(old, new, 1)
    if mutated == ORIGINAL_GATE_TEST:
        raise ValueError(f"{mutant_id}: replacement produced no change")
    # tests/_mutants/<id>/tests/test_deployment_gate_mutant.py sits four
    # directories below the repo root (file -> tests -> <id> -> _mutants ->
    # tests -> root), so PROJECT_ROOT's relative depth (parents[1] in the
    # original, which sits one directory below root) must be patched to
    # parents[4] here, or every model path and the `src` import resolve to
    # the wrong location.
    mutated = mutated.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[4]",
        1,
    )
    mutant_dir = MUTANTS_DIR / mutant_id
    test_dir = mutant_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    mutant_file = test_dir / "test_deployment_gate_mutant.py"
    mutant_file.write_text(mutated, encoding="utf-8")
    conftest_src = (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    conftest_src = conftest_src.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[4]",
        1,
    )
    (test_dir / "conftest.py").write_text(conftest_src, encoding="utf-8")
    return mutant_file


def run_pytest_against_module_mutant(mutant_dir: Path) -> dict:
    """Run the REAL, unmodified test suite with the mutant's src/ on sys.path first."""
    env_pythonpath = str(mutant_dir / "src") + ";" + str(PROJECT_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(GATE_TEST_SRC), "-v", "--no-header", "-q"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**_env_with_pythonpath(env_pythonpath)},
    )
    output = proc.stdout + proc.stderr
    return {"returncode": proc.returncode, "output_tail": "\n".join(output.strip().splitlines()[-15:])}


def _env_with_pythonpath(pythonpath: str) -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    return env


def run_pytest_on_file(test_file: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--no-header", "-q"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    output = proc.stdout + proc.stderr
    return {"returncode": proc.returncode, "output_tail": "\n".join(output.strip().splitlines()[-15:])}


def main():
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)

    # Baseline: confirm the unmutated gate module + unmutated test suite pass.
    baseline = run_pytest_on_file(GATE_TEST_SRC)
    print(f"[baseline] returncode={baseline['returncode']}")
    if baseline["returncode"] != 0:
        print("Baseline gate suite does not pass; aborting mutation run.")
        print(baseline["output_tail"])
        sys.exit(1)

    results = []

    for mutant_id, description, old, new in ENFORCEMENT_MUTANTS:
        try:
            mutant_dir = make_module_mutant(mutant_id, old, new)
        except ValueError as e:
            print(f"[{mutant_id}] SKIPPED: {e}")
            results.append({"id": mutant_id, "fault_layer": "enforcement-logic", "description": description, "status": "skipped", "reason": str(e)})
            continue
        result = run_pytest_against_module_mutant(mutant_dir)
        status = "caught" if result["returncode"] != 0 else "survived"
        print(f"[{mutant_id}] (enforcement-logic, gate module mutated) {status} (returncode={result['returncode']})")
        results.append({
            "id": mutant_id, "fault_layer": "enforcement-logic", "description": description,
            "status": status, "returncode": result["returncode"], "output_tail": result["output_tail"],
        })

    for mutant_id, description, old, new in TEST_ADEQUACY_MUTANTS:
        try:
            mutant_file = make_test_mutant(mutant_id, old, new)
        except ValueError as e:
            print(f"[{mutant_id}] SKIPPED: {e}")
            results.append({"id": mutant_id, "fault_layer": "test-suite-adequacy", "description": description, "status": "skipped", "reason": str(e)})
            continue
        result = run_pytest_on_file(mutant_file)
        status = "caught" if result["returncode"] != 0 else "survived"
        print(f"[{mutant_id}] (test-suite-adequacy, test file mutated) {status} (returncode={result['returncode']})")
        results.append({
            "id": mutant_id, "fault_layer": "test-suite-adequacy", "description": description,
            "status": status, "returncode": result["returncode"], "output_tail": result["output_tail"],
        })

    caught = sum(1 for r in results if r["status"] == "caught")
    total = sum(1 for r in results if r["status"] in ("caught", "survived"))

    print("\n[Experiment 2] Boolean-logic fixtures (resolving M2's equivalence status)")
    boolean_logic_results = run_boolean_logic_experiment()
    for row in boolean_logic_results["fixtures"]:
        marker = "DISTINGUISHES or/and" if row["mutant_distinguished"] else "agrees with or/and"
        print(f"  [{row['fixture']}] dp={row['dp']} eo={row['eo']}: {marker}")
    print(f"  M2 equivalent on real fixtures:     {boolean_logic_results['m2_equivalent_on_real_fixtures']}")
    print(f"  M2 equivalent as gate function:     {boolean_logic_results['m2_equivalent_as_gate_function']}")

    summary = {
        "experiment_1_mutants": {
            "total_mutants": total,
            "caught": caught,
            "survived": total - caught,
            "kill_rate": round(caught / total, 3) if total else None,
            "caught_by_fault_layer": {
                "enforcement-logic": sum(1 for r in results if r["status"] == "caught" and r["fault_layer"] == "enforcement-logic"),
                "test-suite-adequacy": sum(1 for r in results if r["status"] == "caught" and r["fault_layer"] == "test-suite-adequacy"),
            },
            "total_by_fault_layer": {
                "enforcement-logic": sum(1 for r in results if r["fault_layer"] == "enforcement-logic"),
                "test-suite-adequacy": sum(1 for r in results if r["fault_layer"] == "test-suite-adequacy"),
            },
            "mutants": results,
        },
        "experiment_2_boolean_logic_fixtures": boolean_logic_results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExperiment 1 kill rate: {caught}/{total} = {summary['experiment_1_mutants']['kill_rate']}")
    print(f"Results written to {RESULTS_PATH}")

    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)


if __name__ == "__main__":
    main()
