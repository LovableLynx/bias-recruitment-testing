"""
Cross-gate replication of the mutation-testing study (paper Section 4.8).

The original study (scripts/mutation_test_gate.py,
scripts/systematic_test_adequacy_mutants.py, mutation_study/) evaluated one
gate: the fairness gate in src/deployment_gate.py. This script applies the
SAME procedure, with the mutant classes and rules fixed before any outcome was
inspected, to three gates in three domains:

  fairness    src/deployment_gate.py   tests/test_deployment_gate.py
                                        tests/test_gate_boundary_cases.py
  security    src/security_gate.py     tests/test_security_gate.py
                                        tests/test_security_gate_boundary_cases.py
  compliance  src/compliance_gate.py   tests/test_compliance_gate.py
                                        tests/test_compliance_gate_boundary_cases.py

Part A, enforcement-logic mutants. Four fault classes, one mutant each per
gate, mirroring the original M1, M2, M5 and M7:
  X1 flip comparison direction    X2 weaken the combinator
  X5 widen the threshold          X7 swap the BLOCK/DEPLOY outcome
Each mutant patches a copy of the gate module and is run twice against the
real, unmodified tests: (i) the real-fixture test file only, as in the
original Experiment 1, and (ii) the real-fixture plus boundary-case files.

Part B, test-suite-adequacy sweep. The original fixed rule (Section 4.5),
applied mechanically to every test function in both test files of each gate:
  (a) delete-test        remove the test function
  (b) vacuous-assertion  replace the first assertion with `assert True`
  (c) wrong-input        make the test read a different real input:
        fairness, compliance: swap X_test[:, 0] <-> X_test[:, 1]
        security: load the next scan in the fixed cyclic order
                  legacy_imaging -> legacy_web -> project_ci -> legacy_imaging
      not applicable to tests that read no real input.
Each mutated test file runs against the correct, unmodified gate. As in the
original study, a test-suite mutant is "caught" only if the mutated file no
longer passes.

The tool-generated pass (cosmic-ray) for each gate is run separately; see
mutation_study/cr_config_*.toml and scripts/run_cosmic_ray_all.py.

Results: scripts/cross_gate_mutation_results.json

Run history. The first full run, with the mutant classes and rules fixed
beforehand, is kept unchanged in scripts/cross_gate_mutation_results_run1.json.
In that run the compliance gate's X2 mutant (any -> all in the minimum-group-size
check) survived even with the boundary-case tests, because the boundary test for
a too-small group also failed the four-fifths ratio, so the ratio check blocked
it anyway. One isolating boundary test was then added
(test_gate_blocks_on_small_group_even_when_ratio_passes), together with
tests closing the gaps exposed by surviving cosmic-ray mutants. Re-running:
  python scripts/cross_gate_mutation_study.py --gate compliance
  python scripts/cross_gate_mutation_study.py --gate security
write scripts/cross_gate_mutation_results_followup_<gate>.json (run 2, after
the added tests; the tests added are marked in the boundary-case files).
"""
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = PROJECT_ROOT / "tests" / "_mutants_xgate"
RESULTS_PATH = PROJECT_ROOT / "scripts" / "cross_gate_mutation_results.json"
ROOT_LINE = "PROJECT_ROOT = Path(__file__).resolve().parents[1]"

GATES = {
    "fairness": {
        "module": "deployment_gate.py",
        "real_tests": "test_deployment_gate.py",
        "boundary_tests": "test_gate_boundary_cases.py",
        "enforcement_mutants": [
            ("X1_flip_comparison",
             'metrics["demographic_parity_difference"] >= dp_threshold\n        or metrics["equalized_odds_difference"] >= eo_threshold',
             'metrics["demographic_parity_difference"] < dp_threshold\n        or metrics["equalized_odds_difference"] < eo_threshold'),
            ("X2_weaken_combinator",
             '>= dp_threshold\n        or metrics',
             '>= dp_threshold\n        and metrics'),
            ("X5_widen_threshold",
             "DP_THRESHOLD = 0.10\nEO_THRESHOLD = 0.10",
             "DP_THRESHOLD = 0.90\nEO_THRESHOLD = 0.90"),
            ("X7_swap_outcome",
             "        return BLOCK\n    return DEPLOY",
             "        return DEPLOY\n    return BLOCK"),
        ],
    },
    "security": {
        "module": "security_gate.py",
        "real_tests": "test_security_gate.py",
        "boundary_tests": "test_security_gate_boundary_cases.py",
        "enforcement_mutants": [
            ("X1_flip_comparison",
             "if critical > 0 or high > max_high:",
             "if critical <= 0 or high <= max_high:"),
            ("X2_weaken_combinator",
             "if critical > 0 or high > max_high:",
             "if critical > 0 and high > max_high:"),
            ("X5_widen_threshold",
             "MAX_HIGH = 0\n",
             "MAX_HIGH = 1000\n"),
            ("X7_swap_outcome",
             "high > max_high:\n        return BLOCK\n    return DEPLOY",
             "high > max_high:\n        return DEPLOY\n    return BLOCK"),
        ],
    },
    "compliance": {
        "module": "compliance_gate.py",
        "real_tests": "test_compliance_gate.py",
        "boundary_tests": "test_compliance_gate_boundary_cases.py",
        "enforcement_mutants": [
            ("X1_flip_comparison",
             "if impact_ratio(group_counts) < ratio_threshold:",
             "if impact_ratio(group_counts) >= ratio_threshold:"),
            ("X2_weaken_combinator",
             "if any(total < min_group_size",
             "if all(total < min_group_size"),
            ("X5_widen_threshold",
             "IMPACT_RATIO_THRESHOLD = 0.8\n",
             "IMPACT_RATIO_THRESHOLD = 0.1\n"),
            ("X7_swap_outcome",
             "< ratio_threshold:\n        return BLOCK\n    return DEPLOY",
             "< ratio_threshold:\n        return DEPLOY\n    return BLOCK"),
        ],
    },
}

SCAN_ORDER = ["legacy_imaging", "legacy_web", "project_ci"]


def pinned(source: str) -> str:
    """Point a copied test file's PROJECT_ROOT at the real repository root."""
    if ROOT_LINE not in source:
        raise ValueError("PROJECT_ROOT line not found")
    return source.replace(ROOT_LINE, f"PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})", 1)


def run_pytest(files, pythonpath=None) -> int:
    import os
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    proc = subprocess.run([sys.executable, "-m", "pytest", *map(str, files), "-q", "--no-header", "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    return proc.returncode


def copy_tests(dest: Path, files: dict) -> list:
    """Write {filename: source} plus conftest into dest; return the test paths."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "conftest.py").write_text(pinned((PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")), encoding="utf-8")
    out = []
    for name, src in files.items():
        p = dest / name
        p.write_text(pinned(src), encoding="utf-8")
        out.append(p)
    return out


# ---- Part B rule ----------------------------------------------------------
def func_span(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(name)


def test_names(source):
    return [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def rule_delete(source, name):
    node = func_span(source, name)
    lines = source.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1] + lines[node.end_lineno:])


def rule_vacuous(source, name):
    node = func_span(source, name)
    asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
    if not asserts:
        raise ValueError(f"{name}: no assert")
    a = asserts[0]
    lines = source.splitlines(keepends=True)
    indent = " " * a.col_offset
    return "".join(lines[: a.lineno - 1] + [f"{indent}assert True  # mutated\n"] + lines[a.end_lineno:])


def rule_wrong_input(source, name):
    node = func_span(source, name)
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1: node.end_lineno])
    if "X_test[:, 0]" in body:
        new = body.replace("X_test[:, 0]", "X_test[:, 1]", 1)
    elif "X_test[:, 1]" in body:
        new = body.replace("X_test[:, 1]", "X_test[:, 0]", 1)
    else:
        m = re.search(r"load_scan\('(\w+)'\)", body)
        if not m:
            return None
        nxt = SCAN_ORDER[(SCAN_ORDER.index(m.group(1)) + 1) % len(SCAN_ORDER)]
        new = body.replace(m.group(0), f"load_scan('{nxt}')", 1)
    return "".join(lines[: node.lineno - 1] + [new] + lines[node.end_lineno:])


RULES = [("delete-test", rule_delete), ("vacuous-assertion", rule_vacuous), ("wrong-input", rule_wrong_input)]


def main():
    global RESULTS_PATH
    only = sys.argv[sys.argv.index("--gate") + 1] if "--gate" in sys.argv else None
    part_a_only = "--part-a-only" in sys.argv
    if "--out" in sys.argv:
        RESULTS_PATH = PROJECT_ROOT / sys.argv[sys.argv.index("--out") + 1]
    elif only or part_a_only:
        RESULTS_PATH = PROJECT_ROOT / "scripts" / f"cross_gate_mutation_results_followup_{only or 'all'}{'_part_a' if part_a_only else ''}.json"
    if WORK.exists():
        shutil.rmtree(WORK)
    tests_dir = PROJECT_ROOT / "tests"
    results = {}
    for gate, cfg in GATES.items():
        if only and gate != only:
            continue
        real_src = (tests_dir / cfg["real_tests"]).read_text(encoding="utf-8")
        bnd_src = (tests_dir / cfg["boundary_tests"]).read_text(encoding="utf-8")
        module_src = (PROJECT_ROOT / "src" / cfg["module"]).read_text(encoding="utf-8")

        base = copy_tests(WORK / gate / "baseline", {cfg["real_tests"]: real_src, cfg["boundary_tests"]: bnd_src})
        if run_pytest(base) != 0:
            sys.exit(f"{gate}: baseline suite does not pass")

        part_a = []
        for mid, old, new in cfg["enforcement_mutants"]:
            if module_src.count(old) != 1:
                sys.exit(f"{gate} {mid}: pattern must occur exactly once, found {module_src.count(old)}")
            mdir = WORK / gate / mid / "src"
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / cfg["module"]).write_text(module_src.replace(old, new, 1), encoding="utf-8")
            pp = str(mdir)
            real_only = "caught" if run_pytest([base[0]], pp) != 0 else "survived"
            with_bnd = "caught" if run_pytest(base, pp) != 0 else "survived"
            print(f"[{gate}] {mid}: real-fixture tests {real_only}; with boundary tests {with_bnd}")
            part_a.append({"id": mid, "old": old, "new": new, "real_fixture_tests": real_only, "with_boundary_tests": with_bnd})

        part_b = []
        for fname, src in (() if part_a_only else ((cfg["real_tests"], real_src), (cfg["boundary_tests"], bnd_src))):
            for tname in test_names(src):
                for rule, fn in RULES:
                    mutated = fn(src, tname)
                    rec = {"file": fname, "test_function": tname, "rule": rule}
                    if mutated is None:
                        rec["status"] = "not-applicable"
                    else:
                        paths = copy_tests(WORK / gate / "sweep" / f"{tname}__{rule}", {fname: mutated})
                        rec["status"] = "caught" if run_pytest(paths) != 0 else "survived"
                        if rec["status"] == "caught" and rule != "wrong-input" and run_pytest(paths) == 0:
                            # Deleting a test or emptying an assertion cannot make a file fail;
                            # a failure that does not repeat is a run glitch, not a result.
                            rec["status"] = "survived"
                            rec["note"] = "first run failed once and passed on an immediate re-run (transient)"
                    print(f"[{gate}] {fname}::{tname} {rule}: {rec['status']}")
                    part_b.append(rec)

        app = [r for r in part_b if r["status"] != "not-applicable"]
        results[gate] = {
            "module": f"src/{cfg['module']}",
            "test_files": [f"tests/{cfg['real_tests']}", f"tests/{cfg['boundary_tests']}"],
            "part_a_enforcement_mutants": part_a,
            "part_a_summary": {
                "caught_real_fixture_tests_only": sum(r["real_fixture_tests"] == "caught" for r in part_a),
                "caught_with_boundary_tests": sum(r["with_boundary_tests"] == "caught" for r in part_a),
                "total": len(part_a),
            },
            "part_b_sweep": part_b,
            "part_b_summary": {
                "candidates": len(part_b),
                "applicable": len(app),
                "caught": sum(r["status"] == "caught" for r in app),
                "by_rule": {rule: {"applicable": sum(r["rule"] == rule for r in app),
                                   "caught": sum(r["rule"] == rule and r["status"] == "caught" for r in app)}
                            for rule, _ in RULES},
            },
        }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    for gate, r in results.items():
        print(gate, r["part_a_summary"], r["part_b_summary"])
    shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
