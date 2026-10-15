"""Step 17c: print a few misclassified test narratives, terminal only.

Usage: python scripts\\read_errors.py TRUE_NUMBER PREDICTED_NUMBER [HOW_MANY]
The numbers are the row numbers from reports/error_analysis_tfidf_lr.txt.
Nothing is saved. Only the narrative text is printed.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed/complaints_clean.parquet")
PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
SEED = 7
MAX_CHARS = 450


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("true_number", type=int)
    ap.add_argument("predicted_number", type=int)
    ap.add_argument("how_many", type=int, nargs="?", default=4)
    args = ap.parse_args()

    probs = pd.read_parquet(PROBS)
    labels = [c for c in probs.columns if c not in ("complaint_id", "label")]
    assert len(labels) == 9, "expected 9 classes"
    assert 0 <= args.true_number < 9 and 0 <= args.predicted_number < 9

    true_name = labels[args.true_number]
    pred_name = labels[args.predicted_number]
    pred = np.array(labels)[probs[labels].to_numpy().argmax(axis=1)]

    is_true = probs["label"].to_numpy() == true_name
    mask = is_true & (pred == pred_name)
    n_err = int(mask.sum())
    print(f"True class:      {true_name}")
    print(f"Predicted class: {pred_name}")
    print(
        f"{n_err:,} such errors in the test split "
        f"({n_err / int(is_true.sum()) * 100:.1f}% of that true class)"
    )
    if n_err == 0:
        return

    errors = probs[mask].sample(
        n=min(args.how_many, n_err), random_state=SEED
    )
    text = pd.read_parquet(DATA, columns=["complaint_id", "text"])
    text = text[text["complaint_id"].isin(set(errors["complaint_id"]))]
    errors = errors.merge(text, on="complaint_id", how="left")
    assert errors["text"].notna().all(), "text missing for some ids"

    for k, (_, row) in enumerate(errors.iterrows(), start=1):
        p_pred = row[pred_name]
        p_true = row[true_name]
        body = row["text"]
        words = len(body.split())
        print("\n" + "=" * 60)
        print(
            f"Example {k}: model gave {pred_name} {p_pred:.2f}, "
            f"true class {p_true:.2f}; {words} words"
        )
        snippet = body[:MAX_CHARS]
        snippet = snippet.encode("ascii", "replace").decode("ascii")
        print(textwrap.fill(snippet, width=70))
        if len(body) > MAX_CHARS:
            print("[...cut]")


if __name__ == "__main__":
    main()