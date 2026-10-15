"""Step 20b: repeat the random split with other seeds (1 and 2).

Same design as Step 19: same rows, same per-class counts in train/val/test
as the time split, same TF-IDF and Logistic Regression settings.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
OUT = Path("reports/random_split_seeds.txt")
SEEDS = [1, 2]
SPLITS = ["train", "val", "test"]


def run_seed(df, labels, counts, seed):
    rng = np.random.default_rng(seed)
    label_arr = df["label"].to_numpy()
    rsplit = np.full(len(df), "", dtype=object)
    for lab in labels:
        pos = rng.permutation(np.flatnonzero(label_arr == lab))
        n_tr = int(counts.loc[lab, "train"])
        n_va = int(counts.loc[lab, "val"])
        rsplit[pos[:n_tr]] = "train"
        rsplit[pos[n_tr:n_tr + n_va]] = "val"
        rsplit[pos[n_tr + n_va:]] = "test"
    assert (rsplit != "").all()
    train = df[rsplit == "train"]
    test = df[rsplit == "test"]

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True
    )
    x_train = vec.fit_transform(train["text"].tolist())
    clf = LogisticRegression(C=1.0, max_iter=300)
    t0 = time.time()
    clf.fit(x_train, train["label"])
    n_iter = int(clf.n_iter_[0])
    print(f"seed {seed}: fitted in {time.time() - t0:.0f} s "
          f"({n_iter} iterations)", flush=True)
    assert list(clf.classes_) == labels

    pred = clf.predict(vec.transform(test["text"].tolist()))
    macro, acc, _ = summarise(test["label"], pred, labels)
    print(f"seed {seed}: test macro-F1 {macro:.4f}, accuracy {acc:.4f}",
          flush=True)
    return macro, acc, n_iter


def main():
    df = pd.read_parquet(DATA, columns=["complaint_id", "label", "split", "text"])
    labels = sorted(df["label"].unique())
    counts = df.groupby(["label", "split"]).size().unstack()[SPLITS]

    rows = []
    for seed in SEEDS:
        macro, acc, n_iter = run_seed(df, labels, counts, seed)
        rows.append({"seed": seed, "test_macro_f1": round(macro, 4),
                     "test_accuracy": round(acc, 4), "iterations": n_iter})
    res = pd.DataFrame(rows)

    lines = [
        "Random split repeated with other seeds (same design as Step 19)",
        res.to_string(index=False),
        "",
        "Seed 42 (Step 19): macro-F1 0.7400, accuracy 0.8433",
        "Time split (Step 17): macro-F1 0.7446, accuracy 0.8498",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/random_split_seeds.txt")


if __name__ == "__main__":
    main()