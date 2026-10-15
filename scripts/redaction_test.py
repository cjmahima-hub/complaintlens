"""Step 20: do XXXX redaction tokens act as a shortcut?

Measures how common redaction tokens (runs of 2+ X) are, then retrains the
same TF-IDF + Logistic Regression on the time split with those tokens
removed and compares with the Step 17 result.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
TIME_PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
OUT = Path("reports/redaction_test.txt")
REDACT = re.compile(r"\bx{2,}\b", flags=re.IGNORECASE)
SPLITS = ["train", "val", "test"]


def main():
    df = pd.read_parquet(
        DATA, columns=["complaint_id", "label", "split", "text"]
    )
    print(f"Loaded {len(df):,} rows", flush=True)
    labels = sorted(df["label"].unique())
    assert len(labels) == 9, "expected 9 classes"

    df["has_x"] = df["text"].str.contains(REDACT)
    df["clean"] = df["text"].str.replace(REDACT, " ", regex=True)
    assert not df["clean"].str.contains(REDACT).any(), "tokens left behind"
    print("Redaction tokens removed", flush=True)

    by_split = (
        df.groupby("split")["has_x"].mean().loc[SPLITS] * 100
    ).round(1)
    test = df[df["split"] == "test"]
    by_class = (
        test.groupby("label")["has_x"].mean() * 100
    ).round(1)

    train = df[df["split"] == "train"]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=200_000,
        sublinear_tf=True,
    )
    t0 = time.time()
    x_train = vec.fit_transform(train["clean"].tolist())
    t_vec = time.time() - t0
    print(
        f"TF-IDF fitted in {t_vec:.0f} s: {len(vec.vocabulary_):,} features",
        flush=True,
    )

    clf = LogisticRegression(C=1.0, max_iter=300)
    t0 = time.time()
    clf.fit(x_train, train["label"])
    t_clf = time.time() - t0
    n_iter = int(clf.n_iter_[0])
    print(
        f"Logistic Regression fitted in {t_clf:.0f} s ({n_iter} iterations)",
        flush=True,
    )
    assert list(clf.classes_) == labels

    test = test.reset_index(drop=True)
    test["pred_clean"] = clf.predict(vec.transform(test["clean"].tolist()))

    # Time-split reference, recomputed from the Step 17 probabilities
    tp = pd.read_parquet(TIME_PROBS).set_index("complaint_id")
    cls = [c for c in tp.columns if c != "label"]
    assert cls == labels
    assert len(tp) == len(test), "test sizes differ"
    tp = tp.loc[test["complaint_id"]]
    assert (tp["label"].to_numpy() == test["label"].to_numpy()).all()
    test["pred_base"] = np.array(cls)[tp[cls].to_numpy().argmax(axis=1)]

    b_macro, b_acc, b_table = summarise(test["label"], test["pred_base"], labels)
    c_macro, c_acc, c_table = summarise(test["label"], test["pred_clean"], labels)
    assert c_table["support"].sum() == len(test), "support does not add up"
    print(f"original text:  macro-F1 {b_macro:.4f}, accuracy {b_acc:.4f}", flush=True)
    print(f"tokens removed: macro-F1 {c_macro:.4f}, accuracy {c_acc:.4f}", flush=True)

    compare = pd.DataFrame(
        [
            {
                "model": "original text (Step 17)",
                "test_macro_f1": round(b_macro, 4),
                "test_accuracy": round(b_acc, 4),
            },
            {
                "model": "redaction tokens removed",
                "test_macro_f1": round(c_macro, 4),
                "test_accuracy": round(c_acc, 4),
            },
        ]
    )

    rows = []
    for name, keep in [
        ("test narratives WITH tokens", test["has_x"]),
        ("test narratives WITHOUT tokens", ~test["has_x"]),
    ]:
        part = test[keep]
        bm, ba, _ = summarise(part["label"], part["pred_base"], labels)
        cm, ca, _ = summarise(part["label"], part["pred_clean"], labels)
        rows.append(
            {
                "subset": name,
                "rows": len(part),
                "orig_macro_f1": round(bm, 4),
                "clean_macro_f1": round(cm, 4),
                "orig_accuracy": round(ba, 4),
                "clean_accuracy": round(ca, 4),
            }
        )
    subset_table = pd.DataFrame(rows)

    per_class = pd.DataFrame(
        {
            "orig_f1": b_table["f1-score"],
            "clean_f1": c_table["f1-score"],
        }
    )
    per_class["clean_minus_orig"] = (
        per_class["clean_f1"] - per_class["orig_f1"]
    ).round(3)

    lines = [
        "Redaction test: do XXXX tokens act as a shortcut? "
        "(tokens = runs of 2+ X, any case)",
        "Model: TF-IDF (1-2 word n-grams, min_df=5, max 200,000 features, "
        "sublinear tf) + Logistic Regression (C=1.0), time split, as in Step 16",
        f"Fit time: TF-IDF {t_vec:.0f} s, Logistic Regression {t_clf:.0f} s, "
        f"iterations {n_iter} (limit 300)",
        "",
        "% of narratives containing a redaction token, by split:",
        by_split.to_string(),
        "",
        "% of test narratives containing a redaction token, by class:",
        by_class.to_string(),
        "",
        "Test results:",
        compare.to_string(index=False),
        "",
        "Same test set split by whether the original narrative had tokens:",
        subset_table.to_string(index=False),
        "",
        "Per-class F1 on the test part:",
        per_class.to_string(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/redaction_test.txt")


if __name__ == "__main__":
    main()