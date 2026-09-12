"""
Repeats the committed-test-suite and mutation-pass timings from
measure_ci_cost.py multiple times and reports mean/min/max, since a single
wall-clock run is not a defensible number for the paper on its own
(machine load and filesystem caching vary run to run).
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "scripts" / "ci_cost_repeated_results.json"

N_SUITE_RUNS = 5
N_MUTATION_RUNS = 3  # mutation passes are expensive; fewer repeats


def timed(cmd, cwd=PROJECT_ROOT):
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return elapsed, proc.returncode


def summarize(label, times):
    return {
        "label": label,
        "n_runs": len(times),
        "seconds_each": [round(t, 2) for t in times],
        "mean_seconds": round(mean(times), 2),
        "min_seconds": round(min(times), 2),
        "max_seconds": round(max(times), 2),
    }


def main():
    results = {}

    print(f"Committed test suite: {N_SUITE_RUNS} runs")
    suite_times = []
    for i in range(N_SUITE_RUNS):
        elapsed, rc = timed([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"])
        print(f"  run {i+1}: {elapsed:.2f}s (rc={rc})")
        suite_times.append(elapsed)
    results["committed_test_suite"] = summarize(
        "Full committed test suite (16 tests, 4 modules)", suite_times
    )

    print(f"\nSection 4.5 systematic sweep: {N_MUTATION_RUNS} runs")
    sweep_times = []
    for i in range(N_MUTATION_RUNS):
        elapsed, rc = timed([sys.executable, "scripts/systematic_test_adequacy_mutants.py"])
        print(f"  run {i+1}: {elapsed:.2f}s (rc={rc})")
        sweep_times.append(elapsed)
    results["section_4_5_systematic_sweep"] = summarize(
        "Section 4.5 systematic sweep (30 candidate mutants, 23 applicable)", sweep_times
    )

    print(f"\nSection 4.4 cosmic-ray pass: {N_MUTATION_RUNS} runs")
    cosmic_ray_exe = (
        Path.home() / "AppData" / "Local" / "Packages"
        / "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0" / "LocalCache"
        / "local-packages" / "Python311" / "Scripts" / "cosmic-ray.exe"
    )
    toml_config = PROJECT_ROOT / "mutation_study" / "cr_config.toml"
    session_db = PROJECT_ROOT / "mutation_study" / "session.sqlite"

    cosmic_times = []
    if cosmic_ray_exe.exists() and toml_config.exists():
        for i in range(N_MUTATION_RUNS):
            if session_db.exists():
                session_db.unlink()
            start = time.perf_counter()
            subprocess.run([str(cosmic_ray_exe), "init", str(toml_config), str(session_db)],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
            proc = subprocess.run([str(cosmic_ray_exe), "exec", str(toml_config), str(session_db)],
                                   cwd=PROJECT_ROOT, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            print(f"  run {i+1}: {elapsed:.2f}s (rc={proc.returncode})")
            cosmic_times.append(elapsed)
        results["section_4_4_cosmic_ray_pass"] = summarize(
            "Section 4.4 cosmic-ray pass (20 enforcement-logic mutants)", cosmic_times
        )
    else:
        print("  cosmic-ray executable or config not found; skipping repeated timing for this pass.")
        results["section_4_4_cosmic_ray_pass"] = {"note": "not re-run; see scripts/ci_cost_results.json for a single-run figure"}

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    for key, r in results.items():
        if "mean_seconds" in r:
            print(f"{r['label']}: mean {r['mean_seconds']}s, range [{r['min_seconds']}, {r['max_seconds']}]s over {r['n_runs']} runs")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
