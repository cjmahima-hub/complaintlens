"""Stage 2a: choose which complaints go into train / val / test.

Reads the metadata from Stage 1 (no narrative text needed).
Writes data/processed/sample_ids.csv and reports/sample_counts.txt.
"""
from pathlib import Path

import pandas as pd

META = Path("data/processed/window_meta.csv")
OUT_IDS = Path("data/processed/sample_ids.csv")
OUT_REPORT = Path("reports/sample_counts.txt")

SEED = 42
TRAIN_CAP = 5000
VAL_SIZE = 50_000
TEST_SIZE = 100_000


def main():
    meta = pd.read_csv(META, dtype=str)
    print(f"Loaded {len(meta):,} rows", flush=True)

    # Train: 2017-2022, capped per class per year
    train_pool = meta[meta["year"] <= "2022"]
    parts = []
    for _, group in train_pool.groupby(["label", "year"]):
        n = min(TRAIN_CAP, len(group))
        parts.append(group.sample(n=n, random_state=SEED))
    train = pd.concat(parts).assign(split="train")

    # Validation: 2023, natural class mix
    val = (
        meta[meta["year"] == "2023"]
        .sample(n=VAL_SIZE, random_state=SEED)
        .assign(split="val")
    )

    # Test: 2024, natural class mix
    test = (
        meta[meta["year"] == "2024"]
        .sample(n=TEST_SIZE, random_state=SEED)
        .assign(split="test")
    )

    out = pd.concat([train, val, test], ignore_index=True)
    out = out[["complaint_id", "split", "label", "date", "text_hash"]]

    # Safety checks: nothing duplicated, splits in time order
    assert out["complaint_id"].is_unique, "duplicate complaint ids"
    assert out["text_hash"].is_unique, "duplicate texts across splits"
    train_max = out.loc[out["split"] == "train", "date"].max()
    val_min = out.loc[out["split"] == "val", "date"].min()
    val_max = out.loc[out["split"] == "val", "date"].max()
    test_min = out.loc[out["split"] == "test", "date"].min()
    assert train_max < val_min, "train overlaps validation in time"
    assert val_max < test_min, "validation overlaps test in time"

    OUT_IDS.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="text_hash").to_csv(OUT_IDS, index=False)

    sizes = out["split"].value_counts()
    counts = pd.crosstab(out["label"], out["split"])[["train", "val", "test"]]
    counts.loc["All"] = counts.sum()
    pct = (counts.drop(index="All") / counts.loc["All"] * 100).round(1)

    lines = [
        f"Seed: {SEED}   Train cap per class-year: {TRAIN_CAP}",
        f"Train dates: up to {train_max}",
        f"Val dates:   {val_min} to {val_max}",
        f"Test dates:  from {test_min}",
        "",
        f"Rows: train {sizes['train']:,}, val {sizes['val']:,}, test {sizes['test']:,}",
        "",
        "Counts per class and split:",
        counts.to_string(),
        "",
        "Class share within each split (%):",
        pct.to_string(),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/sample_counts.txt")


if __name__ == "__main__":
    main()