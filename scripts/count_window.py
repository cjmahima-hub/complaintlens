"""Stage 1 of the dataset build: count the cleaned window.

One chunked pass over the raw zip. Saves NO narrative text,
only metadata (id, date, label, text hash) and a counts report.
"""
from pathlib import Path

import pandas as pd

from complaintlens.labels import END_DATE, START_DATE, harmonise_product

RAW = Path("data/raw/datalumos_download.zip")
OUT_META = Path("data/processed/window_meta.csv")
OUT_REPORT = Path("reports/window_counts.txt")
COLS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Consumer complaint narrative",
]
CHUNK = 200_000

start = str(START_DATE)[:10]
end = str(END_DATE)[:10]


def main():
    pieces = []
    rows_read = 0
    with_narrative = 0
    in_window = 0

    reader = pd.read_csv(RAW, usecols=COLS, dtype=str, chunksize=CHUNK)
    for i, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)

        text = chunk["Consumer complaint narrative"].fillna("")
        chunk = chunk[text.str.strip() != ""].copy()
        with_narrative += len(chunk)

        date = chunk["Date received"].fillna("")
        chunk = chunk[(date >= start) & (date <= end)].copy()
        in_window += len(chunk)

        chunk["label"] = chunk["Product"].map(harmonise_product)
        chunk = chunk[chunk["label"].notna()].copy()

        norm = (
            chunk["Consumer complaint narrative"]
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        hashes = pd.util.hash_pandas_object(norm, index=False)

        pieces.append(
            pd.DataFrame(
                {
                    "complaint_id": chunk["Complaint ID"].to_numpy(),
                    "date": chunk["Date received"].to_numpy(),
                    "label": chunk["label"].to_numpy(),
                    "text_hash": hashes.to_numpy(),
                }
            )
        )
        print(f"chunk {i}: rows read so far {rows_read:,}", flush=True)

    meta = pd.concat(pieces, ignore_index=True)
    kept_before_dedup = len(meta)

    # Identical texts that carry different labels (label-noise signal)
    label_counts = meta.groupby("text_hash")["label"].nunique()
    conflicting_texts = int((label_counts > 1).sum())

    # Earliest copy wins
    meta = meta.sort_values(["date", "complaint_id"])
    meta = meta.drop_duplicates(subset="text_hash", keep="first")
    meta["year"] = meta["date"].str[:4]

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT_META, index=False)

    table = pd.crosstab(meta["label"], meta["year"], margins=True)
    assert table.loc["All", "All"] == len(meta)

    lines = [
        f"Window: {start} to {end}",
        f"Rows read:                         {rows_read:,}",
        f"With narrative:                    {with_narrative:,}",
        f"...and inside the date window:     {in_window:,}",
        f"...and with a mapped label:        {kept_before_dedup:,}",
        f"After exact-duplicate removal:     {len(meta):,}",
        f"Duplicates removed:                {kept_before_dedup - len(meta):,}",
        f"Duplicate texts with conflicting labels: {conflicting_texts:,}",
        "",
        "Unique narratives per class and year:",
        table.to_string(),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/window_counts.txt")


if __name__ == "__main__":
    main()