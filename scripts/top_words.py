"""Step 21: top words per class, and save the model for the demo.

Retrains the Step 16 model (time split, original text), saves it to
models/, verifies it reproduces the Step 17 test results, and lists the
15 terms with the largest positive coefficient for each class.
Vocabulary terms only: no narrative text is printed or saved.
"""
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
TIME_PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
MODEL_DIR = Path("models")
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLF_PATH = MODEL_DIR / "logreg_classifier.joblib"
OUT = Path("reports/top_words.txt")
TOP_N = 15
REDACT = re.compile(r"x{2,}", flags=re.IGNORECASE)


def main():
    df = pd.read_parquet(
        DATA, columns=["complaint_id", "label", "split", "text"]
    )
    print(f"Loaded {len(df):,} rows", flush=True)
    labels = sorted(df["label"].unique())
    assert len(labels) == 9, "expected 9 classes"

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].reset_index(drop=True)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=200_000,
        sublinear_tf=True,
    )
    t0 = time.time()
    x_train = vec.fit_transform(train["text"].tolist())
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

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(clf, CLF_PATH)
    print("Model saved to models/", flush=True)

    # Reload from disk and verify against Step 17
    vec2 = joblib.load(VEC_PATH)
    clf2 = joblib.load(CLF_PATH)
    pred = clf2.predict(vec2.transform(test["text"].tolist()))
    macro, acc, table = summarise(test["label"], pred, labels)
    assert table["support"].sum() == len(test), "support does not add up"

    tp = pd.read_parquet(TIME_PROBS).set_index("complaint_id")
    cls = [c for c in tp.columns if c != "label"]
    assert cls == labels
    assert len(tp) == len(test), "test sizes differ"
    tp = tp.loc[test["complaint_id"]]
    assert (tp["label"].to_numpy() == test["label"].to_numpy()).all()
    step17_pred = np.array(cls)[tp[cls].to_numpy().argmax(axis=1)]
    agree = float((pred == step17_pred).mean())
    print(
        f"reloaded model: macro-F1 {macro:.4f}, accuracy {acc:.4f}; "
        f"agrees with Step 17 predictions on {agree * 100:.2f}% of rows",
        flush=True,
    )

    # Top terms per class
    terms = np.array(vec2.get_feature_names_out())
    coef = clf2.coef_
    assert coef.shape == (len(labels), len(terms))
    lines = [
        "Top terms per class: largest positive coefficients",
        "Model: TF-IDF (1-2 word n-grams, min_df=5, max 200,000 features, "
        "sublinear tf) + Logistic Regression (C=1.0), time split, as in Step 16",
        f"Reloaded model on the test part: macro-F1 {macro:.4f}, "
        f"accuracy {acc:.4f} (Step 17: 0.7446 / 0.8498)",
        f"Predictions agree with Step 17 on {agree * 100:.2f}% of test rows",
        "Coefficients are relative to the other classes; they show what the "
        "model associates with a class, not why a complaint belongs to it.",
        "",
    ]
    redact_counts = {}
    for i, lab in enumerate(labels):
        top = np.argsort(coef[i])[::-1][:TOP_N]
        names = [str(terms[j]) for j in top]
        redact_counts[lab] = sum(bool(REDACT.search(n)) for n in names)
        lines.append(lab)
        for j, name in zip(top, names):
            lines.append(f"    {coef[i, j]:7.2f}  {name}")
        lines.append("")
    lines.append(f"Top-{TOP_N} terms containing a redaction token, by class:")
    lines.append(pd.Series(redact_counts).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/top_words.txt")


if __name__ == "__main__":
    main()