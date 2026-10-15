"""Step 16: TF-IDF + Logistic Regression.

The TF-IDF vocabulary is learned from the training split only.
Results are written to reports/tfidf_lr.txt; the fitted model
is saved to models/ (gitignored).
"""
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
MODEL_OUT = Path("models/tfidf_lr.joblib")
OUT = Path("reports/tfidf_lr.txt")


def main():
    df = pd.read_parquet(DATA, columns=["label", "split", "text"])
    print(f"Loaded {len(df):,} rows", flush=True)
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    labels = sorted(parts["train"]["label"].unique())
    assert len(labels) == 9, "expected 9 classes"

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

    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)

    lines = [
        "TF-IDF (1-2 word n-grams, min_df=5, max 200,000 features, "
        "sublinear tf) + Logistic Regression (C=1.0)",
        f"Features: {len(vec.vocabulary_):,}",
        f"Fit time: TF-IDF {t_vec:.0f} s, Logistic Regression {t_clf:.0f} s",
        f"Iterations used: {n_iter} (limit 300)",
        "",
    ]
    summary_rows = []
    detail = []
    for split in ["val", "test"]:
        part = parts[split]
        pred = pipe.predict(part["text"].tolist())
        macro, acc, table = summarise(part["label"], pred, labels)
        assert table["support"].sum() == len(part), "support does not add up"
        summary_rows.append(
            {
                "split": split,
                "macro_f1": round(macro, 4),
                "accuracy": round(acc, 4),
            }
        )
        detail.append(f"--- per-class results on {split} ---")
        detail.append(table.to_string())
        detail.append("")
        print(f"{split}: macro-F1 {macro:.4f}, accuracy {acc:.4f}", flush=True)

    lines += ["SUMMARY", pd.DataFrame(summary_rows).to_string(index=False), ""]
    lines += detail
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/tfidf_lr.txt")


if __name__ == "__main__":
    main()