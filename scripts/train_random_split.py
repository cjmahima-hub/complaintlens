"""Step 19: random split vs time split.

Same 380,002 narratives and the SAME number of rows per class in
train / val / test as the time split. The only difference: each
class's rows are shuffled and dealt out at random, ignoring dates.
Same TF-IDF and Logistic Regression settings as Step 16.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
TIME_PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
OUT = Path("reports/random_vs_time_split.txt")
SEED = 42
SPLITS = ["train", "val", "test"]


def main():
    df = pd.read_parquet(
        DATA, columns=["complaint_id", "label", "split", "text"]
    )
    print(f"Loaded {len(df):,} rows", flush=True)
    labels = sorted(df["label"].unique())
    assert len(labels) == 9, "expected 9 classes"

    # Rows per class in each time-split part: the random split copies these
    counts = df.groupby(["label", "split"]).size().unstack()[SPLITS]

    # Deal each class's rows out at random, ignoring dates
    rng = np.random.default_rng(SEED)
    label_arr = df["label"].to_numpy()
    rsplit = np.full(len(df), "", dtype=object)
    for lab in labels:
        pos = rng.permutation(np.flatnonzero(label_arr == lab))
        n_tr = int(counts.loc[lab, "train"])
        n_va = int(counts.loc[lab, "val"])
        n_te = int(counts.loc[lab, "test"])
        assert n_tr + n_va + n_te == len(pos)
        rsplit[pos[:n_tr]] = "train"
        rsplit[pos[n_tr:n_tr + n_va]] = "val"
        rsplit[pos[n_tr + n_va:]] = "test"
    assert (rsplit != "").all(), "some rows were not assigned"
    df["rsplit"] = rsplit

    new_counts = df.groupby(["label", "rsplit"]).size().unstack()[SPLITS]
    assert (new_counts.to_numpy() == counts.to_numpy()).all()
    print("Split check passed: per-class counts match the time split", flush=True)

    # How mixed is the random test set? (original split = time period)
    mix = pd.crosstab(df["rsplit"], df["split"], normalize="index") * 100
    mix = mix.loc[SPLITS, SPLITS].round(1)
    mix.index = ["random train", "random val", "random test"]
    mix.columns = ["from 2017-2022", "from 2023", "from 2024"]

    parts = {s: df[df["rsplit"] == s] for s in SPLITS}

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=200_000,
        sublinear_tf=True,
    )
    t0 = time.time()
    x_train = vec.fit_transform(parts["train"]["text"].tolist())
    t_vec = time.time() - t0
    print(
        f"TF-IDF fitted in {t_vec:.0f} s: {len(vec.vocabulary_):,} features",
        flush=True,
    )

    clf = LogisticRegression(C=1.0, max_iter=300)
    t0 = time.time()
    clf.fit(x_train, parts["train"]["label"])
    t_clf = time.time() - t0
    n_iter = int(clf.n_iter_[0])
    print(
        f"Logistic Regression fitted in {t_clf:.0f} s ({n_iter} iterations)",
        flush=True,
    )
    assert list(clf.classes_) == labels

    # Time-split reference, recomputed from the Step 17 probabilities
    tp = pd.read_parquet(TIME_PROBS)
    cls = [c for c in tp.columns if c not in ("complaint_id", "label")]
    assert cls == labels
    t_pred = np.array(cls)[tp[cls].to_numpy().argmax(axis=1)]
    t_macro, t_acc, t_table = summarise(tp["label"], t_pred, labels)

    results = {}
    tables = {}
    for split in ["val", "test"]:
        part = parts[split]
        pred = clf.predict(vec.transform(part["text"].tolist()))
        macro, acc, table = summarise(part["label"], pred, labels)
        assert table["support"].sum() == len(part), "support does not add up"
        results[split] = (macro, acc)
        tables[split] = table
        print(
            f"random split, {split}: macro-F1 {macro:.4f}, accuracy {acc:.4f}",
            flush=True,
        )

    compare = pd.DataFrame(
        [
            {
                "split_method": "time (train 2017-22, test 2024)",
                "test_macro_f1": round(t_macro, 4),
                "test_accuracy": round(t_acc, 4),
            },
            {
                "split_method": "random (same rows, same class counts)",
                "test_macro_f1": round(results["test"][0], 4),
                "test_accuracy": round(results["test"][1], 4),
            },
        ]
    )
    per_class = pd.DataFrame(
        {
            "time_split_f1": t_table["f1-score"],
            "random_split_f1": tables["test"]["f1-score"],
        }
    )
    per_class["random_minus_time"] = (
        per_class["random_split_f1"] - per_class["time_split_f1"]
    ).round(3)

    lines = [
        "Random split vs time split: same 380,002 narratives, same rows per "
        f"class in each part (seed {SEED})",
        "Model: TF-IDF (1-2 word n-grams, min_df=5, max 200,000 features, "
        "sublinear tf) + Logistic Regression (C=1.0), as in Step 16",
        f"Random split fit time: TF-IDF {t_vec:.0f} s, Logistic Regression "
        f"{t_clf:.0f} s, iterations {n_iter} (limit 300)",
        "",
        "Which time period do the random parts come from? (% of each part)",
        mix.to_string(),
        "",
        "Test results, time split vs random split:",
        compare.to_string(index=False),
        "",
        "Random split, validation: macro-F1 "
        f"{results['val'][0]:.4f}, accuracy {results['val'][1]:.4f}",
        "",
        "Per-class F1 on the test part:",
        per_class.to_string(),
        "",
        "Per-class results, random split, test part:",
        tables["test"].to_string(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/random_vs_time_split.txt")


if __name__ == "__main__":
    main()