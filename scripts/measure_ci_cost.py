"""
Wall-clock cost of running this study's checks continuously, as a follow-up
to Section 6's future-work item "the CI-time cost of running fairness and
mutation checks alongside the existing suite."

This measures three real, already-existing costs on this machine, rather
than estimating them:
  1. The committed test suite (all 16 tests, four modules) as it runs on
     every push via the GitHub Actions workflow described in Section 3.3.
  2. The cosmic-ray tool-generated mutation pass from Section 4.4 (20
     enforcement-logic mutants).
  3. The systematic rule-generated mutation pass from Section 4.5 (30
     candidate test-suite-adequacy mutants).

Mutation testing is not something a project would run on every commit
(the numbers below make clear why); it is included here to give an honest
comparison between "per-commit CI cost" and "periodic mutation-audit cost."
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "scripts" / "ci_cost_results.json"


def timed_run(label, cmd, cwd=PROJECT_ROOT, display_cmd=None):
    shown = display_cmd or " ".join(cmd)
    print(f"\nRunning: {label}")
    print(f"  $ {shown}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    print(f"  -> {elapsed:.2f}s (returncode {proc.returncode})")
    return {
        "label": label,
        "command": shown,
        "seconds": round(elapsed, 2),
        "returncode": proc.returncode,
    }


def main():
    results = []

    # 1. The committed test suite, exactly as CI runs it.
    results.append(timed_run(
        "Full committed test suite (16 tests, 4 modules)",
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        display_cmd="pytest tests/ -q --no-header",
    ))

    # 2. The Section 4.4 tool-generated mutation pass (cosmic-ray, 20 mutants).
    #    Only run if cosmic-ray is installed and configured; otherwise report
    #    the already-recorded result from mutation_study/cosmic_ray_results.txt.
    cosmic_ray_log = PROJECT_ROOT / "mutation_study" / "cosmic_ray_results.txt"
    session_db = PROJECT_ROOT / "mutation_study" / "session.sqlite"
    toml_config = PROJECT_ROOT / "mutation_study" / "cr_config.toml"
    import shutil as _shutil
    _candidates = [
        Path.home() / "AppData" / "Local" / "Packages"
        / "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0" / "LocalCache"
        / "local-packages" / "Python311" / "Scripts" / "cosmic-ray.exe",
    ]
    cosmic_ray_exe = next((c for c in _candidates if c.exists()), None)
    if cosmic_ray_exe is None:
        found_on_path = _shutil.which("cosmic-ray")
        cosmic_ray_exe = Path(found_on_path) if found_on_path else Path("cosmic-ray")
    if toml_config.exists():
        if session_db.exists():
            session_db.unlink()
        start = time.perf_counter()
        init_proc = subprocess.run(
            [str(cosmic_ray_exe), "init", str(toml_config), str(session_db)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        if init_proc.returncode != 0:
            print(f"  cosmic-ray init failed: {init_proc.stderr[-500:]}")
        proc = subprocess.run(
            [str(cosmic_ray_exe), "exec", str(toml_config), str(session_db)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  cosmic-ray exec stderr tail: {proc.stderr[-500:]}")
        elapsed = time.perf_counter() - start
        print(f"\ncosmic-ray pass (20 mutants): {elapsed:.2f}s (returncode {proc.returncode})")
        results.append({
            "label": "Section 4.4 cosmic-ray pass (20 enforcement-logic mutants)",
            "command": "cosmic-ray init+exec",
            "seconds": round(elapsed, 2),
            "returncode": proc.returncode,
        })
    else:
        print(f"\ncosmic-ray config not found at {toml_config}; skipping re-run, "
              f"see {cosmic_ray_log} for the recorded Section 4.4 result (20/20 killed).")
        results.append({
            "label": "Section 4.4 cosmic-ray pass (20 enforcement-logic mutants)",
            "command": "not re-run this session",
            "seconds": None,
            "note": f"see {cosmic_ray_log.name} for recorded outcome",
        })

    # 3. The Section 4.5 systematic test-suite-adequacy sweep (30 candidates).
    results.append(timed_run(
        "Section 4.5 systematic sweep (30 candidate mutants, 23 applicable)",
        [sys.executable, "scripts/systematic_test_adequacy_mutants.py"],
        display_cmd="python scripts/systematic_test_adequacy_mutants.py",
    ))

    committed_suite_seconds = results[0]["seconds"]
    summary = {
        "purpose": (
            "Real wall-clock cost, on this machine, of (1) the committed test "
            "suite as it runs on every push, versus (2)-(3) the two mutation-"
            "testing passes from Sections 4.4-4.5, which are periodic audits "
            "rather than per-commit checks."
        ),
        "runs": results,
        "committed_suite_seconds": committed_suite_seconds,
        "note": (
            "The committed suite (16 tests) is what actually runs on every push "
            "in the GitHub Actions workflow (Section 3.3); the two mutation "
            "passes are one-off validation experiments (Sections 4.4-4.5), not "
            "part of the per-commit CI loop, which is why their cost is reported "
            "separately rather than added to the per-commit figure."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Committed test suite (per-commit CI cost): {committed_suite_seconds}s")
    for r in results[1:]:
        print(f"{r['label']}: {r.get('seconds')}s")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
