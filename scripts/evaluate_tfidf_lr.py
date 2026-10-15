"""Step 17: confusion matrix, error analysis and saved probabilities.

Loads the fitted model from models/ and never refits it. Writes
statistics and a picture only: no narrative text goes into reports/.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from complaintlens.evaluation import summarise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/processed/complaints_clean.parquet")
MODEL = Path("models/tfidf_lr.joblib")
PROBS = {
    "val": Path("data/processed/tfidf_lr_probs_val.parquet"),
    "test": Path("data/processed/tfidf_lr_probs_test.parquet"),
}
REPORT = Path("reports/error_analysis_tfidf_lr.txt")
PNG = Path("reports/confusion_matrix_tfidf_lr.png")

SHORT = {
    "Checking or savings account": "Checking/savings",
    "Credit card or prepaid card": "Credit card",
    "Credit reporting": "Credit reporting",
    "Debt collection": "Debt collection",
    "Money transfer or virtual currency": "Money transfer",
    "Mortgage": "Mortgage",
    "Payday, title, or personal loan": "Payday/personal",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
}


def save_heatmap(pct, names, path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("TF-IDF + Logistic Regression, 2024 test (% of each true class)")
    for i in range(len(names)):
        for j in range(len(names)):
            v = pct[i, j]
            ax.text(
                j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if v > 50 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    assert MODEL.stat().st_size > 0, "model file is empty"
    pipe = joblib.load(MODEL)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    labels = list(clf.classes_)
    assert len(labels) == 9, "expected 9 classes"

    df = pd.read_parquet(
        DATA, columns=["complaint_id", "label", "split", "text"]
    )
    results = {}
    for split in ["val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        t0 = time.time()
        x = vec.transform(part["text"].tolist())
        probs = clf.predict_proba(x)
        pred = clf.classes_[probs.argmax(axis=1)]
        assert (pred == clf.predict(x)).all(), "argmax disagrees with predict"
        macro, acc, _ = summarise(part["label"], pred, labels)
        print(
            f"{split}: macro-F1 {macro:.4f}, accuracy {acc:.4f} "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )
        out = pd.DataFrame(probs, columns=labels)
        out.insert(0, "label", part["label"].to_numpy())
        out.insert(0, "complaint_id", part["complaint_id"].to_numpy())
        out.to_parquet(PROBS[split], index=False)
        results[split] = (part, pred, macro, acc)

    part, pred, macro, acc = results["test"]
    y = part["label"].to_numpy()
    cm = confusion_matrix(y, pred, labels=labels)
    assert cm.sum() == len(part), "confusion matrix does not add up"
    pct = cm / cm.sum(axis=1, keepdims=True) * 100

    names = [SHORT.get(lab, lab) for lab in labels]
    idx = [f"{i} {n}" for i, n in enumerate(names)]
    cols = [str(i) for i in range(len(names))]
    cm_pct = pd.DataFrame(pct.round(1), index=idx, columns=cols)
    cm_cnt = pd.DataFrame(cm, index=idx, columns=cols)

    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                pairs.append(
                    {
                        "true": names[i],
                        "predicted": names[j],
                        "count": int(cm[i, j]),
                        "pct_of_true_class": round(float(pct[i, j]), 1),
                    }
                )
    top = pd.DataFrame(pairs).sort_values("count", ascending=False).head(10)

    words = pd.Series([len(t.split()) for t in part["text"]])
    bucket = pd.cut(
        words,
        bins=[0, 50, 150, 400, np.inf],
        labels=["1-50", "51-150", "151-400", "400+"],
    )
    correct = pd.Series(pred == y)
    by_len = (
        pd.DataFrame({"bucket": bucket, "correct": correct})
        .groupby("bucket", observed=True)["correct"]
        .agg(["size", "mean"])
    )
    by_len["accuracy_pct"] = (by_len["mean"] * 100).round(1)
    by_len = by_len[["size", "accuracy_pct"]]

    lines = [
        "TF-IDF + Logistic Regression: error analysis on the 2024 test split",
        f"Test macro-F1 {macro:.4f}, accuracy {acc:.4f} "
        "(must match reports/tfidf_lr.txt)",
        "",
        "Confusion matrix. Rows = TRUE class, columns = PREDICTED class "
        "(column numbers match the row numbers).",
        "Each row shows the % of that true class sent to each predicted class:",
        cm_pct.to_string(),
        "",
        "Same matrix as counts:",
        cm_cnt.to_string(),
        "",
        "Top 10 confusions (true class -> predicted class):",
        top.to_string(index=False),
        "",
        "Accuracy by narrative length (words), test split:",
        by_len.to_string(),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    save_heatmap(pct, names, PNG)
    print("Done. See reports/error_analysis_tfidf_lr.txt and the PNG next to it")


if __name__ == "__main__":
    main()