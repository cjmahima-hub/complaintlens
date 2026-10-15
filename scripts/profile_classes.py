"""Step 14: profile the clean dataset per class.

Prints and saves statistics only. No narrative text is written.
"""
from pathlib import Path

import pandas as pd

DATA = Path("data/processed/complaints_clean.parquet")
OUT = Path("reports/class_profile.txt")


def main():
    df = pd.read_parquet(DATA, columns=["label", "split", "text"])
    print(f"Loaded {len(df):,} rows", flush=True)

    df["n_words"] = df["text"].str.split().str.len()
    df["n_xxxx"] = df["text"].str.count("XXXX")
    df["has_xxxx"] = df["n_xxxx"] > 0
    df["xxxx_flag"] = df["has_xxxx"].astype(float)

    per_class = df.groupby("label").agg(
        rows=("text", "size"),
        median_words=("n_words", "median"),
        p95_words=("n_words", lambda s: s.quantile(0.95)),
        pct_with_xxxx=("xxxx_flag", "mean"),
        median_xxxx=("n_xxxx", "median"),
        mean_xxxx=("n_xxxx", "mean"),
    )
    per_class["pct_with_xxxx"] = (per_class["pct_with_xxxx"] * 100).round(1)
    per_class["mean_xxxx"] = per_class["mean_xxxx"].round(2)

    by_split = (
        df.groupby(["label", "split"])["xxxx_flag"].mean().unstack()
    )[["train", "val", "test"]]
    by_split = (by_split * 100).round(1)

    lines = [
        "Per class (all splits together):",
        per_class.to_string(),
        "",
        "Percent of narratives containing XXXX, by class and split:",
        by_split.to_string(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/class_profile.txt")


if __name__ == "__main__":
    main()