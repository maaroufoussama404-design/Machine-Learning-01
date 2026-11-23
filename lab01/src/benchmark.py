import os
import sys
import csv
import time
import statistics
import random
from pathlib import Path

import numpy as np
import psutil
import joblib
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

def peak_memory_delta(function):
    import threading

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / (1024 ** 2)
    peak = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                current = process.memory_info().rss / (1024 ** 2)
                peak[0] = max(peak[0], current)
            except psutil.Error:
                pass
            stop.wait(0.01)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        function()
    finally:
        stop.set()
        thread.join()

    return max(0.0, peak[0] - baseline)

results = []
saved_models = {}

for name, factory in models.items():
    print(f"\nRunning {name}...")

    # Warm-up
    factory().fit(X_train, y_train)

    # Five training measurements
    training_times = []
    final_model = None
    for _ in range(5):
        model = factory()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_times.append(time.perf_counter() - start)
        final_model = model

    training_median = statistics.median(training_times)

    # Accuracy
    predictions = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Single-sample inference: 100 repetitions
    sample = X_test[:1]
    final_model.predict(sample)
    inference_times = []
    for _ in range(100):
        start = time.perf_counter_ns()
        final_model.predict(sample)
        inference_times.append((time.perf_counter_ns() - start) / 1_000_000)

    inference_median = statistics.median(inference_times)

    # Memory measurements
    def train_once():
        factory().fit(X_train, y_train)

    train_memory = peak_memory_delta(train_once)

    def infer_once():
        for _ in range(100):
            final_model.predict(sample)

    inference_memory = peak_memory_delta(infer_once)
    peak_memory = max(train_memory, inference_memory)

    # Save model
    filename = "logistic_regression.joblib" if name == "Logistic Regression" else "random_forest.joblib"
    model_path = RESULTS / filename
    joblib.dump(final_model, model_path)
    model_bytes = model_path.stat().st_size
    saved_models[name] = final_model

    results.append({
        "model": name,
        "accuracy": accuracy,
        "training_median_seconds": training_median,
        "inference_median_ms": inference_median,
        "model_size_bytes": model_bytes,
        "model_size_KB": model_bytes / 1024,
        "training_peak_memory_delta_MB": train_memory,
        "inference_peak_memory_delta_MB": inference_memory,
        "peak_memory_delta_MB": peak_memory
    })

# Required combined model file
joblib.dump(saved_models, RESULTS / "model.joblib")

# Accuracy CSV
with open(RESULTS / "baseline_accuracy.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "test_accuracy"])
    for r in results:
        writer.writerow([r["model"], f'{r["accuracy"]:.4f}'])

# Full measurements CSV
with open(RESULTS / "measurements.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Resource budget assessment
def assess(r, target):
    latency = r["inference_median_ms"]
    size_mb = r["model_size_bytes"] / (1024 ** 2)
    memory = r["peak_memory_delta_MB"]

    if target == "Cloud":
        return "YES*" if latency <= 100 and size_mb <= 500 else "NO"
    if target == "Edge":
        return "YES" if latency <= 50 and size_mb <= 50 and memory <= 1024 else "NO"
    if target == "Mobile":
        return "YES" if latency <= 20 and size_mb <= 10 and memory <= 256 else "NO"
    return "YES" if latency <= 10 and r["model_size_bytes"] <= 100 * 1024 and memory <= 0.25 else "NO"

with open(RESULTS / "deployment_fit.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "Cloud", "Edge", "Mobile", "TinyML"])
    for r in results:
        writer.writerow([r["model"]] + [
            assess(r, target) for target in ["Cloud", "Edge", "Mobile", "TinyML"]
        ])

# Generate report using actual measurements
lines = [
    "# Lab 1: Environment and First System Measurements",
    "",
    "## 1. Goal",
    "Train and benchmark Logistic Regression and Random Forest on the Breast Cancer Wisconsin dataset.",
    "",
    "## 2. Method",
    "- Dataset: sklearn Breast Cancer Wisconsin.",
    "- Split: 70/30 stratified, random_state=42.",
    "- LogisticRegression(max_iter=1000, random_state=42).",
    "- RandomForestClassifier(n_estimators=100, random_state=42).",
    "- Training: one warm-up and five measured runs; median reported.",
    "- Inference: 100 single-sample predictions; median reported.",
    "- Memory: peak process memory increase sampled using memory_profiler.",
    "- Model serialization: joblib.",
    "",
    "## 3. Results",
    "",
    "| Model | Accuracy | Training median (s) | Inference median (ms) | Size (KB) | Peak memory increase (MB) |",
    "|---|---:|---:|---:|---:|---:|"
]

for r in results:
    lines.append(
        f'| {r["model"]} | {r["accuracy"]:.4f} | '
        f'{r["training_median_seconds"]:.6f} | '
        f'{r["inference_median_ms"]:.6f} | '
        f'{r["model_size_KB"]:.2f} | '
        f'{r["peak_memory_delta_MB"]:.3f} |'
    )

lines += [
    "",
    "### Deployment budget assessment",
    "",
    "| Model | Cloud | Edge | Mobile | TinyML |",
    "|---|---|---|---|---|"
]

for r in results:
    lines.append("| " + r["model"] + " | " + " | ".join(
        assess(r, target) for target in ["Cloud", "Edge", "Mobile", "TinyML"]
    ) + " |")

lines += [
    "",
    "*Cloud YES assumes a host with at least 1 GB RAM, as specified in the manual. Other assessments use the measured process memory increase, latency and serialized model size. These are estimates, not hardware deployment tests.*",
    "",
    "## 4. Conclusions - REVIEW AND REWRITE IN YOUR OWN WORDS",
]

best = max(results, key=lambda r: r["accuracy"])
fastest = min(results, key=lambda r: r["inference_median_ms"])
smallest = min(results, key=lambda r: r["model_size_bytes"])

lines += [
    f"1. {best['model']} achieved the higher test accuracy ({best['accuracy']:.4f}) in this experiment.",
    f"2. {fastest['model']} had the lowest median single-sample inference latency ({fastest['inference_median_ms']:.4f} ms).",
    f"3. {smallest['model']} produced the smaller serialized model ({smallest['model_size_KB']:.2f} KB), which is relevant to deployment constraints.",
    "",
    "## 5. Reproducibility",
    "Random seeds were set to 42. See results/versions.txt for package versions and results/ for raw measurements.",
    "",
    "IMPORTANT: Rewrite the three conclusions in your own words before submission. Review the measurements and make sure you can explain them."
]

(BASE / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("\nLab 1 experiment completed.")
print("Results:", RESULTS)
print("Report:", BASE / "report.md")
print("\n", "\n".join(lines))