"""Step 15: baselines that every later model must beat.

1. Majority class: always predict the most common class in the
   pre-2023 natural data (chosen without looking at val or test).
2. Length and redaction counts only: three numbers per narrative,
   no words at all. Shows how much text-free signals carry.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from complaintlens.evaluation import summarise

DATA = Path("data/processed/complaints_clean.parquet")
META = Path("data/processed/window_meta.csv")
OUT = Path("reports/baselines.txt")


def numeric_features(text):
    words = text.str.split().str.len()
    xxxx = text.str.count("XXXX")
    return pd.DataFrame(
        {
            "log_words": np.log1p(words),
            "log_xxxx": np.log1p(xxxx),
            "xxxx_per_100_words": xxxx / words * 100,
        }
    )


def main():
    df = pd.read_parquet(DATA, columns=["label", "split", "text"])
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    labels = sorted(parts["train"]["label"].unique())
    assert len(labels) == 9, "expected 9 classes"

    # Baseline 1: majority class of the pre-2023 natural data
    meta = pd.read_csv(META, usecols=["date", "label"], dtype=str)
    pre2023 = meta[meta["date"] <= "2022-12-31"]
    shares = pre2023["label"].value_counts(normalize=True)
    majority = shares.idxmax()
    print(f"Majority class: {majority} ({shares.max() * 100:.1f}%)", flush=True)

    def predict_majority(part):
        return pd.Series(majority, index=part.index)

    # Baseline 2: numeric features only
    print("Training the length + XXXX model...", flush=True)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(numeric_features(parts["train"]["text"]), parts["train"]["label"])

    def predict_numeric(part):
        return model.predict(numeric_features(part["text"]))

    baselines = {
        "Majority class": predict_majority,
        "Length + XXXX counts only": predict_numeric,
    }

    summary_rows = []
    detail = []
    for name, predict in baselines.items():
        for split in ["val", "test"]:
            part = parts[split]
            macro, acc, table = summarise(part["label"], predict(part), labels)
            summary_rows.append(
                {
                    "baseline": name,
                    "split": split,
                    "macro_f1": round(macro, 4),
                    "accuracy": round(acc, 4),
                }
            )
            detail.append(f"--- {name} on {split}: per-class results ---")
            detail.append(table.to_string())
            detail.append("")
            print(f"done: {name} on {split}", flush=True)

    summary = pd.DataFrame(summary_rows)
    lines = [
        f"Majority class (pre-2023 natural data): {majority} "
        f"({shares.max() * 100:.1f}% of those narratives)",
        "",
        "SUMMARY",
        summary.to_string(index=False),
        "",
    ] + detail
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/baselines.txt")


if __name__ == "__main__":
    main()