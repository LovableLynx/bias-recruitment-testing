# scripts/measure_overhead.py
import time
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))
from fairness_checks import load_model, compute_fairness_metrics, compute_performance_metrics

def time_run(label, fn, n=5):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"{label}: avg {avg*1000:.1f}ms over {n} runs")
    return avg

def main():
    data_path = PROJECT_ROOT / 'data' / 'FairCVtest' / 'data' / 'FairCVdb.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X_test = data['Profiles Test']
    y_true = (data['Blind Labels Test'] > 0.5).astype(int)
    sensitive_features = X_test[:, 1]

    model_path = PROJECT_ROOT / 'src' / 'models' / 'baseline_fair_recruiter.pkl'
    model = load_model(model_path)

    # warm-up call, not timed
    model.predict(X_test)

    bare_inference = time_run("Bare .predict() only", lambda: model.predict(X_test))
    with_perf = time_run("Predict + performance metrics", lambda: compute_performance_metrics(model, X_test, y_true))
    with_fairness = time_run("Full fairness check", lambda: compute_fairness_metrics(model, X_test, y_true, sensitive_features))

    print(f"\nOverhead of fairness check vs bare inference: {(with_fairness - bare_inference)*1000:.1f}ms")

if __name__ == "__main__":
    main()