"""
Systematic (rule-generated) test-suite-adequacy mutants for the deployment
gate's test suite, run as a follow-up to scripts/mutation_test_gate.py.

Context: the paper's original test-suite-adequacy mutants (M3, M4, M6) were
hand-selected by the author, which is a threat to validity -- the author
both designed the gate and chose which faults to inject into its tests, so
nothing in the original evaluation was independently adversarial. This
script closes part of that gap for the test-suite-adequacy side (the
enforcement-logic side is already addressed by scripts/mutation_test_gate.py's
cosmic-ray pass in Section 4.4, which is genuinely tool-generated).

THE RULE (fixed before checking outcomes, applied mechanically to every test
function in tests/test_deployment_gate.py and tests/test_gate_boundary_cases.py,
with no author discretion about which functions get which mutant):

  (a) delete-test:      remove the entire test function from the suite
  (b) vacuous-assertion: replace the test's assertion statement(s) with
                          `assert True`
  (c) wrong-fixture-column: if the test reads a sensitive-attribute column
                          index (X_test[:, 0] or X_test[:, 1]), swap it to
                          the other index; skipped for tests with no such
                          column read

This mechanically generates 23 candidate mutants across the 10 test
functions in the two files (3 in test_deployment_gate.py, 7 in
test_gate_boundary_cases.py). The paper's original M3, M4, and M6 are each
a member of this larger set (M3 = delete-test on
test_ethnicity_model_is_deployable; M4 = wrong-fixture-column on the same
test; M6 = vacuous-assertion on test_gender_model_is_deployable), which is
reported as a sanity check, not tuned for -- the rule was fixed first and
applied to all 10 functions uniformly, not selected to reproduce M3/M4/M6.
"""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
MUTANTS_DIR = TESTS_DIR / "_systematic_mutants"
RESULTS_PATH = PROJECT_ROOT / "scripts" / "systematic_test_adequacy_results.json"

TARGET_FILES = ["test_deployment_gate.py", "test_gate_boundary_cases.py"]


def get_test_function_names(source: str) -> list[str]:
    tree = ast.parse(source)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]


def get_function_source_span(source: str, func_name: str) -> tuple[int, int]:
    """Return (start_line, end_line) 1-indexed, inclusive, for a top-level function."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.lineno, node.end_lineno
    raise ValueError(f"function {func_name} not found")


def apply_delete_test(source: str, func_name: str) -> str:
    start, end = get_function_source_span(source, func_name)
    lines = source.splitlines(keepends=True)
    # also drop a single leading blank line before the function, if present,
    # to avoid leaving double-blank artifacts; not semantically important.
    new_lines = lines[: start - 1] + lines[end:]
    return "".join(new_lines)


def apply_vacuous_assertion(source: str, func_name: str) -> str:
    start, end = get_function_source_span(source, func_name)
    lines = source.splitlines(keepends=True)
    func_lines = lines[start - 1 : end]
    func_src = "".join(func_lines)
    tree = ast.parse(func_src)
    func_node = tree.body[0]
    indent = " " * (func_node.body[0].col_offset) if func_node.body else "    "
    # Find the first Assert node's line range within the function and replace
    # every Assert statement's source line(s) with a single `assert True`.
    assert_nodes = [n for n in ast.walk(func_node) if isinstance(n, ast.Assert)]
    if not assert_nodes:
        raise ValueError(f"{func_name}: no assert statement found")
    func_line_offset = start - 1  # 0-indexed offset of function's first line within `lines`
    # Replace from the first assert's start line through its end line (assert
    # statements here may span multiple lines with a message argument).
    first_assert = assert_nodes[0]
    a_start = func_line_offset + first_assert.lineno
    a_end = func_line_offset + first_assert.end_lineno
    replacement = f"{indent}assert True  # mutated: assertion body removed\n"
    new_lines = lines[: a_start - 1] + [replacement] + lines[a_end:]
    return "".join(new_lines)


def apply_wrong_fixture_column(source: str, func_name: str) -> str | None:
    start, end = get_function_source_span(source, func_name)
    lines = source.splitlines(keepends=True)
    func_src = "".join(lines[start - 1 : end])
    if "X_test[:, 0]" in func_src:
        mutated_func = func_src.replace("X_test[:, 0]", "X_test[:, 1]", 1)
    elif "X_test[:, 1]" in func_src:
        mutated_func = func_src.replace("X_test[:, 1]", "X_test[:, 0]", 1)
    else:
        return None  # rule does not apply to this test
    new_lines = lines[: start - 1] + [mutated_func] + lines[end:]
    return "".join(new_lines)


def make_mutant_file(target_filename: str, mutated_source: str, mutant_id: str) -> Path:
    mutant_dir = MUTANTS_DIR / mutant_id / "tests"
    mutant_dir.mkdir(parents=True, exist_ok=True)
    mutated_source = mutated_source.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[4]",
        1,
    )
    out_path = mutant_dir / target_filename.replace(".py", "_mutant.py")
    out_path.write_text(mutated_source, encoding="utf-8")
    conftest_src = (TESTS_DIR / "conftest.py").read_text(encoding="utf-8")
    conftest_src = conftest_src.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[4]",
        1,
    )
    (mutant_dir / "conftest.py").write_text(conftest_src, encoding="utf-8")
    return out_path


def run_pytest_on_file(test_file: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    output = proc.stdout + proc.stderr
    return {"returncode": proc.returncode, "output_tail": "\n".join(output.strip().splitlines()[-10:])}


def main():
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)

    results = []

    for filename in TARGET_FILES:
        src_path = TESTS_DIR / filename
        original_source = src_path.read_text(encoding="utf-8")
        test_names = get_test_function_names(original_source)
        print(f"\n=== {filename}: {len(test_names)} test functions ===")

        for func_name in test_names:
            # (a) delete-test
            mutant_id = f"{filename[:-3]}__{func_name}__delete_test"
            try:
                mutated = apply_delete_test(original_source, func_name)
                mutant_file = make_mutant_file(filename, mutated, mutant_id)
                result = run_pytest_on_file(mutant_file)
                status = "caught" if result["returncode"] != 0 else "survived"
                print(f"  [{mutant_id}] {status}")
                results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                 "mutation_type": "delete-test", "status": status,
                                 "returncode": result["returncode"]})
            except Exception as e:
                print(f"  [{mutant_id}] ERROR: {e}")
                results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                 "mutation_type": "delete-test", "status": "error", "error": str(e)})

            # (b) vacuous-assertion
            mutant_id = f"{filename[:-3]}__{func_name}__vacuous_assertion"
            try:
                mutated = apply_vacuous_assertion(original_source, func_name)
                mutant_file = make_mutant_file(filename, mutated, mutant_id)
                result = run_pytest_on_file(mutant_file)
                status = "caught" if result["returncode"] != 0 else "survived"
                print(f"  [{mutant_id}] {status}")
                results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                 "mutation_type": "vacuous-assertion", "status": status,
                                 "returncode": result["returncode"]})
            except Exception as e:
                print(f"  [{mutant_id}] ERROR: {e}")
                results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                 "mutation_type": "vacuous-assertion", "status": "error", "error": str(e)})

            # (c) wrong-fixture-column (only where applicable)
            mutant_id = f"{filename[:-3]}__{func_name}__wrong_fixture_column"
            try:
                mutated = apply_wrong_fixture_column(original_source, func_name)
                if mutated is None:
                    print(f"  [{mutant_id}] N/A (no sensitive-attribute column read)")
                    results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                     "mutation_type": "wrong-fixture-column", "status": "not-applicable"})
                else:
                    mutant_file = make_mutant_file(filename, mutated, mutant_id)
                    result = run_pytest_on_file(mutant_file)
                    status = "caught" if result["returncode"] != 0 else "survived"
                    print(f"  [{mutant_id}] {status}")
                    results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                     "mutation_type": "wrong-fixture-column", "status": status,
                                     "returncode": result["returncode"]})
            except Exception as e:
                print(f"  [{mutant_id}] ERROR: {e}")
                results.append({"id": mutant_id, "file": filename, "test_function": func_name,
                                 "mutation_type": "wrong-fixture-column", "status": "error", "error": str(e)})

    applicable = [r for r in results if r["status"] in ("caught", "survived")]
    caught = sum(1 for r in applicable if r["status"] == "caught")
    total = len(applicable)

    # Cross-check against the paper's original hand-picked M3/M4/M6.
    original_overlap = {
        "M3_drop_ethnicity_gate_test": "test_deployment_gate__test_ethnicity_model_is_deployable__delete_test",
        "M4_swap_sensitive_attribute": "test_deployment_gate__test_ethnicity_model_is_deployable__wrong_fixture_column",
        "M6_vacuous_assertion": "test_deployment_gate__test_gender_model_is_deployable__vacuous_assertion",
    }
    overlap_check = {}
    for orig_id, mechanical_id in original_overlap.items():
        match = next((r for r in results if r["id"] == mechanical_id), None)
        overlap_check[orig_id] = {
            "matches_mechanical_candidate": mechanical_id,
            "mechanical_status": match["status"] if match else "NOT FOUND",
        }

    summary = {
        "rule": {
            "delete_test": "remove the entire test function",
            "vacuous_assertion": "replace the test's assertion(s) with `assert True`",
            "wrong_fixture_column": "swap X_test[:, 0] <-> X_test[:, 1] where such a read exists; not applicable otherwise",
        },
        "total_candidates_generated": len(results),
        "applicable_candidates": total,
        "caught": caught,
        "survived": total - caught,
        "kill_rate": round(caught / total, 3) if total else None,
        "overlap_with_original_hand_picked_mutants": overlap_check,
        "all_results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Total mechanically generated candidates: {len(results)}")
    print(f"Applicable (excludes not-applicable): {total}")
    print(f"Caught: {caught}/{total} = {summary['kill_rate']}")
    print(f"Survived: {total - caught}")
    print(f"\nOverlap check against original hand-picked M3/M4/M6:")
    for orig_id, info in overlap_check.items():
        print(f"  {orig_id} -> {info['matches_mechanical_candidate']}: {info['mechanical_status']}")
    print(f"\nResults written to {RESULTS_PATH}")

    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)


if __name__ == "__main__":
    main()
