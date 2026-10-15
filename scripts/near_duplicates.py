"""Step 18: how many 2024 test narratives have a near-identical training twin?

Uses rare-word TF-IDF vectors (very common words are dropped, so two
narratives only look similar if they share distinctive wording) and
cosine similarity. Statistics only: no narrative text is printed or saved.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
OUT_SIM = Path("data/processed/test_sample_similarity.parquet")
REPORT = Path("reports/near_duplicates.txt")

SEED = 42
SAMPLE = 20_000
CHUNK = 250
THRESHOLDS = [0.95, 0.90, 0.80]


def main():
    df = pd.read_parquet(
        DATA, columns=["complaint_id", "label", "split", "text"]
    )
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    probs = pd.read_parquet(PROBS).set_index("complaint_id")
    labels = [c for c in probs.columns if c != "label"]
    assert len(labels) == 9, "expected 9 classes"
    probs = probs.loc[test["complaint_id"]]
    assert (probs["label"].to_numpy() == test["label"].to_numpy()).all()
    test["pred"] = np.array(labels)[probs[labels].to_numpy().argmax(axis=1)]

    sample = test.sample(n=SAMPLE, random_state=SEED).reset_index(drop=True)
    print(
        f"Train: {len(train):,} rows; test sample: {len(sample):,} rows",
        flush=True,
    )

    vec = TfidfVectorizer(
        min_df=3, max_df=0.05, sublinear_tf=True, dtype=np.float32
    )
    t0 = time.time()
    x_train = vec.fit_transform(train["text"].tolist())
    print(
        f"Rare-word TF-IDF fitted in {time.time() - t0:.0f} s: "
        f"{len(vec.vocabulary_):,} features",
        flush=True,
    )
    x_train_t = x_train.T.tocsr()
    x_test = vec.transform(sample["text"].tolist())
    no_words = int((x_test.getnnz(axis=1) == 0).sum())

    # Self-check: a training narrative must be most similar to itself
    s = x_train[:100] @ x_train_t
    self_sim = s.diagonal()
    nonzero = x_train[:100].getnnz(axis=1) > 0
    assert (self_sim[nonzero] > 0.999).all(), "self-similarity check failed"
    print("Self-check passed", flush=True)

    n_chunks = (SAMPLE + CHUNK - 1) // CHUNK
    max_sim = np.zeros(SAMPLE, dtype=np.float32)
    nn_idx = np.zeros(SAMPLE, dtype=np.int64)
    t0 = time.time()
    for k in range(n_chunks):
        lo, hi = k * CHUNK, min((k + 1) * CHUNK, SAMPLE)
        s = x_test[lo:hi] @ x_train_t
        max_sim[lo:hi] = s.max(axis=1).toarray().ravel()
        nn_idx[lo:hi] = np.asarray(s.argmax(axis=1)).ravel()
        if k == 0 or (k + 1) % 10 == 0:
            print(
                f"chunk {k + 1} of {n_chunks} done "
                f"({time.time() - t0:.0f} s so far)",
                flush=True,
            )
    assert max_sim.max() < 1.001, "similarity above 1 is impossible"

    sample["max_sim"] = max_sim
    has_nn = max_sim > 0
    sample["nn_id"] = np.where(
        has_nn, train["complaint_id"].to_numpy()[nn_idx], ""
    )
    sample["nn_label"] = np.where(
        has_nn, train["label"].to_numpy()[nn_idx], ""
    )
    OUT_SIM.parent.mkdir(parents=True, exist_ok=True)
    sample[
        ["complaint_id", "label", "pred", "max_sim", "nn_id", "nn_label"]
    ].to_parquet(OUT_SIM, index=False)

    # Share of the sample with a twin at each threshold
    rows = []
    for t in THRESHOLDS:
        hit = sample["max_sim"] >= t
        n = int(hit.sum())
        same = (sample.loc[hit, "nn_label"] == sample.loc[hit, "label"]).mean()
        ok = (sample.loc[hit, "pred"] == sample.loc[hit, "label"]).mean()
        rows.append(
            {
                "similarity_at_least": t,
                "narratives": n,
                "pct_of_sample": round(n / len(sample) * 100, 1),
                "twin_has_same_label_pct": round(same * 100, 1) if n else None,
                "model_correct_pct": round(ok * 100, 1) if n else None,
            }
        )
    thr_table = pd.DataFrame(rows)

    # Model scores with and without twins, same sample
    everyone = pd.Series(True, index=sample.index)
    subsets = [
        ("all sampled test narratives", everyone),
        ("twins at 0.90 or more removed", sample["max_sim"] < 0.90),
        ("twins at 0.80 or more removed", sample["max_sim"] < 0.80),
    ]
    score_rows = []
    for name, keep in subsets:
        part = sample[keep]
        macro, acc, _ = summarise(part["label"], part["pred"], labels)
        score_rows.append(
            {
                "subset": name,
                "rows": len(part),
                "macro_f1": round(macro, 4),
                "accuracy": round(acc, 4),
            }
        )
    score_table = pd.DataFrame(score_rows)

    per_class = (
        sample.assign(twin=sample["max_sim"] >= 0.90)
        .groupby("label")
        .agg(rows=("twin", "size"), with_twin=("twin", "sum"))
    )
    per_class["pct_with_twin"] = (
        per_class["with_twin"] / per_class["rows"] * 100
    ).round(1)

    quant = pd.Series(max_sim).quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(3)

    lines = [
        "Near-duplicate check: 2024 test sample vs the full training split",
        f"Test sample: {len(sample):,} narratives (seed {SEED}); "
        f"training narratives compared against: {len(train):,}",
        "Vectors: TF-IDF, single words, words in more than 5% of narratives "
        "dropped, cosine similarity",
        f"Sampled narratives with no usable words after filtering "
        f"(similarity 0, cannot be judged): {no_words:,}",
        "",
        "Highest similarity to any training narrative, percentiles of the sample:",
        quant.to_string(),
        "",
        "Share of the sample with a twin at each threshold:",
        thr_table.to_string(index=False),
        "",
        "Model scores on the same sample, with and without twins:",
        score_table.to_string(index=False),
        "",
        "Share of each class with a twin at 0.90 or more:",
        per_class.to_string(),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/near_duplicates.txt")


if __name__ == "__main__":
    main()