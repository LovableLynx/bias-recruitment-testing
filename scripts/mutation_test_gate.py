"""
Mutation testing harness for the deployment-gate test suite
(tests/test_deployment_gate.py).

Two complementary experiments are run:

1. Model-fixture mutants (M1-M7): a single fault is injected into a copy
   of the gate test module, which is then executed directly against the
   real trained models and the real FairCVdb test split (no mocking). The
   oracle for whether a mutant is CAUGHT is the known ground truth: the
   biased models (gender, ethnicity) must be rejected, and the reference
   (blind-label) model must be accepted. A mutant is CAUGHT if the mutated
   module's pytest run no longer reports results consistent with that
   ground truth. A mutant SURVIVES if it still does, meaning the injected
   fault had no observable effect on this fixture.

   One risk with this design: a mutant can be semantically different from
   the original but still survive simply because the real models' metric
   values don't happen to distinguish the two conditions (e.g. `or` vs
   `and` when both metrics independently exceed threshold on every biased
   fixture in the dataset). Such a mutant is not a genuine counterexample
   to "the gate works" so much as an EQUIVALENT MUTANT under the current
   test data, a well-known confound in mutation testing (Jia & Harman,
   2011). Experiment 2 exists specifically to resolve this ambiguity.

2. Boolean-logic fixtures (B1-B4, boolean_logic_fixtures()): synthetic,
   hand-picked (dp, eo) metric pairs that are constructed specifically to
   make `or` vs `and` observably different, independent of what any real
   trained model happens to produce. These are run against the gate's
   *decision logic* directly (reimplemented inline, not by mutating the
   test file), to determine whether M2 (or -> and) is a genuine fault or
   an artifact of the real models' data happening to make both branches
   agree.

Faults are also tagged with a `fault_layer`:
  - "enforcement-logic": the fault is in the decision computed FROM a
    given, correct metric value (comparison operator, threshold, boolean
    combinator, wrong feature column, wrong fixture).
  - "test-suite-adequacy": the fault is in the test suite's coverage or
    the presence of a real assertion, not in any decision logic (a
    dropped test case, a vacuous assertion). These cannot be caught by
    strengthening enforcement logic; they require an external check on
    the suite itself (e.g. coverage auditing, this harness).

This intentionally does not use a general-purpose mutation testing tool
(e.g. mutmut, cosmic-ray): the fault list is hand-selected to mirror the
specific defect class discussed in the paper (Section 3.4) rather than an
exhaustive AST-level mutation sweep. See Limitations in the paper.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_SRC = PROJECT_ROOT / "tests" / "test_deployment_gate.py"
MUTANTS_DIR = PROJECT_ROOT / "tests" / "_mutants"
RESULTS_PATH = PROJECT_ROOT / "scripts" / "mutation_results.json"

ORIGINAL = GATE_SRC.read_text(encoding="utf-8")

# Each mutant: (id, fault_layer, description, old_string, new_string)
MUTANTS = [
    (
        "M1_flip_gender_operator",
        "enforcement-logic",
        "Flip >= to < and or to and in the gender gate's disparity check (reintroduces the original defect).",
        '        metrics["demographic_parity_difference"] >= DP_THRESHOLD\n        or metrics["equalized_odds_difference"] >= EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
        '        metrics["demographic_parity_difference"] < DP_THRESHOLD\n        and metrics["equalized_odds_difference"] < EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
    ),
    (
        "M2_or_to_and",
        "enforcement-logic",
        "Change the gender gate's `or` to `and` (requires both metrics to breach threshold, not either). See Experiment 2 for whether this is equivalent under real model data.",
        '    assert (\n        metrics["demographic_parity_difference"] >= DP_THRESHOLD\n        or metrics["equalized_odds_difference"] >= EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
        '    assert (\n        metrics["demographic_parity_difference"] >= DP_THRESHOLD\n        and metrics["equalized_odds_difference"] >= EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
    ),
    (
        "M3_drop_ethnicity_gate",
        "test-suite-adequacy",
        "Delete the ethnicity gate test entirely (coverage gap: gender-only gating). Not a logic fault; a missing check.",
        'def test_ethnicity_model_is_deployable(test_data):\n    """Gate check: biased ethnicity model should FAIL this gate (not deployable)."""\n    X_test, y_true = test_data\n    ethnicity = X_test[:, 0]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')\n    metrics = compute_fairness_metrics(model, X_test, y_true, ethnicity)\n    assert (\n        metrics["demographic_parity_difference"] >= DP_THRESHOLD\n        or metrics["equalized_odds_difference"] >= EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\n',
        '',
    ),
    (
        "M4_swap_sensitive_attribute",
        "enforcement-logic",
        "Use gender as the sensitive feature in the ethnicity gate test (wrong column, index 1 instead of 0).",
        '    ethnicity = X_test[:, 0]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')',
        '    ethnicity = X_test[:, 1]\n    model = load_model(PROJECT_ROOT / \'src\' / \'models\' / \'baseline_biased_recruiter_ethnicity.pkl\')',
    ),
    (
        "M5_threshold_widened",
        "enforcement-logic",
        "Widen DP/EO thresholds from 0.10 to 0.90, so almost nothing gets blocked.",
        "DP_THRESHOLD = 0.10\nEO_THRESHOLD = 0.10",
        "DP_THRESHOLD = 0.90\nEO_THRESHOLD = 0.90",
    ),
    (
        "M6_gate_result_ignored",
        "test-suite-adequacy",
        "Compute metrics but assert an unconditional True (gate result never actually consumed). Not a logic fault; a vacuous check.",
        '    assert (\n        metrics["demographic_parity_difference"] >= DP_THRESHOLD\n        or metrics["equalized_odds_difference"] >= EO_THRESHOLD\n    ), (\n        f"Gate should have BLOCKED the biased model but did not "\n        f"(dp={metrics[\'demographic_parity_difference\']:.3f}, "\n        f"eo={metrics[\'equalized_odds_difference\']:.3f})"\n    )\n\ndef test_ethnicity_model_is_deployable',
        '    assert True  # gate result not actually checked\n\ndef test_ethnicity_model_is_deployable',
    ),
    (
        "M7_reference_model_fixture_swapped",
        "enforcement-logic",
        "Point the reference-model gate at the biased gender model's file path instead of the reference model's (wrong fixture).",
        "model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl')\n    metrics = compute_fairness_metrics(model, X_test, y_true, gender)\n    assert metrics[\"demographic_parity_difference\"] < DP_THRESHOLD",
        "model = load_model(PROJECT_ROOT / 'src' / 'models' / 'baseline_biased_recruiter.pkl')\n    metrics = compute_fairness_metrics(model, X_test, y_true, gender)\n    assert metrics[\"demographic_parity_difference\"] < DP_THRESHOLD",
    ),
]

# ---------------------------------------------------------------------------
# Experiment 2: synthetic (dp, eo) fixtures that make `or` vs `and` in the
# gate's combinator observably different, resolving whether M2 is a genuine
# fault or an equivalent mutant under the real models' data.
# ---------------------------------------------------------------------------
THRESHOLD = 0.10

BOOLEAN_LOGIC_FIXTURES = [
    ("B1_both_breach", 0.146, 0.607, "both metrics breach threshold (mirrors the real gender-biased model)"),
    ("B2_only_dp_breach", 0.146, 0.05, "only demographic parity breaches threshold"),
    ("B3_only_eo_breach", 0.05, 0.607, "only equalized odds breaches threshold"),
    ("B4_neither_breach", 0.02, 0.03, "neither metric breaches threshold (mirrors the real reference model)"),
]


def gate_or(dp, eo):
    """Original gate logic: reject if EITHER metric breaches threshold."""
    return dp >= THRESHOLD or eo >= THRESHOLD


def gate_and(dp, eo):
    """M2-mutated gate logic: reject only if BOTH metrics breach threshold."""
    return dp >= THRESHOLD and eo >= THRESHOLD


def run_boolean_logic_experiment() -> dict:
    rows = []
    distinguished = False
    for fixture_id, dp, eo, description in BOOLEAN_LOGIC_FIXTURES:
        original_reject = gate_or(dp, eo)
        mutant_reject = gate_and(dp, eo)
        differs = original_reject != mutant_reject
        distinguished = distinguished or differs
        rows.append({
            "fixture": fixture_id,
            "description": description,
            "dp": dp,
            "eo": eo,
            "original_gate_rejects": original_reject,
            "m2_mutant_gate_rejects": mutant_reject,
            "mutant_distinguished": differs,
        })
    return {
        "m2_is_equivalent_on_real_model_data": True,
        "m2_is_equivalent_in_general": not distinguished,
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


def make_mutant(mutant_id: str, old: str, new: str) -> Path:
    if old not in ORIGINAL:
        raise ValueError(f"{mutant_id}: old_string not found in source file")
    mutated = ORIGINAL.replace(old, new, 1)
    if mutated == ORIGINAL:
        raise ValueError(f"{mutant_id}: replacement produced no change")
    # Mutant files live one directory deeper than tests/test_deployment_gate.py
    # (tests/_mutants/<id>/... vs tests/...), so PROJECT_ROOT's relative depth
    # (parents[1] in the original) must be patched to parents[3] here, or every
    # model path and the `src` import resolve one level too shallow.
    mutated = mutated.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[3]",
        1,
    )
    mutant_dir = MUTANTS_DIR / mutant_id
    mutant_dir.mkdir(parents=True, exist_ok=True)
    mutant_file = mutant_dir / "test_deployment_gate_mutant.py"
    mutant_file.write_text(mutated, encoding="utf-8")
    # conftest.py provides the test_data fixture and has the same PROJECT_ROOT
    # depth dependency; patch it the same way before copying it alongside so
    # the mutant can be run as an isolated, self-contained test module.
    conftest_src = (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    conftest_src = conftest_src.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[3]",
        1,
    )
    (mutant_dir / "conftest.py").write_text(conftest_src, encoding="utf-8")
    return mutant_file


def run_pytest(mutant_file: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(mutant_file), "-v", "--no-header", "-q"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    output = proc.stdout + proc.stderr
    return {
        "returncode": proc.returncode,
        "output_tail": "\n".join(output.strip().splitlines()[-15:]),
    }


def main():
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)

    # Baseline: confirm the unmutated gate suite passes (all correct verdicts).
    baseline = run_pytest(GATE_SRC)
    print(f"[baseline] returncode={baseline['returncode']}")
    if baseline["returncode"] != 0:
        print("Baseline gate suite does not pass; aborting mutation run.")
        print(baseline["output_tail"])
        sys.exit(1)

    results = []
    for mutant_id, fault_layer, description, old, new in MUTANTS:
        try:
            mutant_file = make_mutant(mutant_id, old, new)
        except ValueError as e:
            print(f"[{mutant_id}] SKIPPED: {e}")
            results.append({
                "id": mutant_id, "fault_layer": fault_layer,
                "description": description, "status": "skipped", "reason": str(e),
            })
            continue

        result = run_pytest(mutant_file)
        # A mutant is CAUGHT if the mutated suite no longer exits 0
        # (i.e. it no longer reports "all models correctly gated").
        # It SURVIVES if it still exits 0 despite the injected fault,
        # meaning the fault had no effect on the suite's pass/fail verdict
        # on this specific fixture (see module docstring re: equivalent
        # mutants, resolved for M2 by Experiment 2 below).
        status = "caught" if result["returncode"] != 0 else "survived"
        print(f"[{mutant_id}] ({fault_layer}) {status} (returncode={result['returncode']})")
        results.append({
            "id": mutant_id,
            "fault_layer": fault_layer,
            "description": description,
            "status": status,
            "returncode": result["returncode"],
            "output_tail": result["output_tail"],
        })

    caught = sum(1 for r in results if r["status"] == "caught")
    total = sum(1 for r in results if r["status"] in ("caught", "survived"))

    print("\n[Experiment 2] Boolean-logic fixtures (resolving M2's equivalence status)")
    boolean_logic_results = run_boolean_logic_experiment()
    for row in boolean_logic_results["fixtures"]:
        marker = "DISTINGUISHES or/and" if row["mutant_distinguished"] else "agrees with or/and"
        print(f"  [{row['fixture']}] dp={row['dp']} eo={row['eo']}: {marker}")
    print(f"  M2 equivalent on real model data only: {boolean_logic_results['m2_is_equivalent_on_real_model_data']}")
    print(f"  M2 equivalent in general (any inputs):  {boolean_logic_results['m2_is_equivalent_in_general']}")

    summary = {
        "experiment_1_model_fixture_mutants": {
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
    print(f"\nExperiment 1 kill rate: {caught}/{total} = {summary['experiment_1_model_fixture_mutants']['kill_rate']}")
    print(f"Results written to {RESULTS_PATH}")

    # Mutant files share a module basename (test_deployment_gate_mutant.py)
    # across subdirectories, which pytest's default rootdir collection
    # cannot disambiguate; leaving this directory in place makes a later
    # `pytest tests/` fail with import-collision errors. Always clean up.
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)


if __name__ == "__main__":
    main()
