"""Step 22: prediction latency on this CPU.

Times the saved TF-IDF + Logistic Regression model on 1,000 random 2024
test narratives, one complaint at a time (prediction only, and prediction
plus top-5 contributing words), and as one batch.
Statistics only: no narrative text is printed or saved.
"""
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

DATA = Path("data/processed/complaints_clean.parquet")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")
CLF_PATH = Path("models/logreg_classifier.joblib")
OUT = Path("reports/latency.txt")
N = 1000
WARMUP = 20
SEED = 42


def main():
    t0 = time.perf_counter()
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    load_s = time.perf_counter() - t0
    terms = np.array(vec.get_feature_names_out())
    print(f"Models loaded in {load_s:.2f} s", flush=True)

    df = pd.read_parquet(DATA, columns=["split", "text"])
    test = df[df["split"] == "test"].sample(n=N, random_state=SEED)
    texts = test["text"].tolist()
    median_words = int(test["text"].str.split().str.len().median())

    def predict_a(text):
        x = vec.transform([text])
        k = int(clf.predict_proba(x)[0].argmax())
        return k, x

    def predict_b(text):
        k, x = predict_a(text)
        coo = x.tocoo()
        contrib = coo.data * clf.coef_[k][coo.col]
        order = np.argsort(contrib)[::-1][:5]
        return k, [str(terms[coo.col[i]]) for i in order]

    for t in texts[:WARMUP]:
        predict_b(t)

    def time_each(fn):
        times, results = [], []
        for t in texts:
            t0 = time.perf_counter()
            r = fn(t)
            times.append(time.perf_counter() - t0)
            results.append(r)
        return np.array(times) * 1000, results

    ms_a, res_a = time_each(predict_a)
    ms_b, res_b = time_each(predict_b)
    print("One-at-a-time timing done", flush=True)

    t0 = time.perf_counter()
    batch_k = clf.predict_proba(vec.transform(texts)).argmax(axis=1)
    batch_s = time.perf_counter() - t0

    single_k = np.array([r[0] for r in res_a])
    assert (single_k == batch_k).all(), "single and batch predictions differ"
    assert (single_k == np.array([r[0] for r in res_b])).all()
    print("Check passed: one-at-a-time and batch predictions agree", flush=True)

    def row(name, ms):
        return {
            "version": name,
            "median_ms": round(float(np.median(ms)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "p99_ms": round(float(np.percentile(ms, 99)), 2),
            "max_ms": round(float(ms.max()), 2),
        }

    table = pd.DataFrame(
        [
            row("prediction only", ms_a),
            row("prediction + top 5 words", ms_b),
        ]
    )
    per_row_ms = batch_s / N * 1000
    print(table.to_string(index=False), flush=True)
    print(
        f"Batch of {N}: {batch_s:.2f} s total, {per_row_ms:.2f} ms per "
        f"complaint ({N / batch_s:,.0f} per second)",
        flush=True,
    )

    lines = [
        "Latency: TF-IDF + Logistic Regression, time split model (Step 21)",
        f"Machine: {platform.processor()}, {os.cpu_count()} logical CPUs, "
        f"{platform.system()} {platform.release()}",
        f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}",
        f"Sample: {N} random 2024 test narratives (seed {SEED}), median "
        f"length {median_words} words, {WARMUP} warm-up runs not timed",
        f"Model load time: {load_s:.2f} s",
        "",
        "One complaint at a time, raw text to predicted class (milliseconds):",
        table.to_string(index=False),
        "",
        f"Batch of {N}: {batch_s:.2f} s total, {per_row_ms:.2f} ms per "
        f"complaint ({N / batch_s:,.0f} per second)",
        "One-at-a-time and batch predictions agree on all rows.",
        "Timings vary between runs and depend on the machine and narrative length.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/latency.txt")


if __name__ == "__main__":
    main()