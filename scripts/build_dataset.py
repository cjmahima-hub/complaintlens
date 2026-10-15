"""Stage 2b: fetch narrative text for the sampled complaint IDs.

Reads sample_ids.csv (from Stage 2a), makes one chunked pass over the
raw zip, and writes data/processed/complaints_clean.parquet.
The Company column is never read.
"""
from pathlib import Path

import pandas as pd

from complaintlens.text import collapse_whitespace, normalise_text

RAW = Path("data/raw/datalumos_download.zip")
IDS = Path("data/processed/sample_ids.csv")
META = Path("data/processed/window_meta.csv")
OUT = Path("data/processed/complaints_clean.parquet")
OUT_REPORT = Path("reports/dataset_summary.txt")
CHUNK = 200_000
NARR = "Consumer complaint narrative"


def main():
    ids = pd.read_csv(IDS, dtype=str)
    wanted = set(ids["complaint_id"])
    print(f"Sampled ids: {len(ids):,}", flush=True)

    pieces = []
    found = 0
    rows_read = 0
    reader = pd.read_csv(
        RAW, usecols=["Complaint ID", NARR], dtype=str, chunksize=CHUNK
    )
    for i, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        hit = chunk[chunk["Complaint ID"].isin(wanted)]
        if len(hit) > 0:
            pieces.append(
                hit.rename(
                    columns={"Complaint ID": "complaint_id", NARR: "raw_text"}
                )
            )
            found += len(hit)
        print(
            f"chunk {i}: rows read {rows_read:,}, found {found:,}",
            flush=True,
        )

    texts = pd.concat(pieces, ignore_index=True)
    assert texts["complaint_id"].is_unique, "duplicate ids in raw file"
    df = ids.merge(texts, on="complaint_id", how="left", validate="one_to_one")
    assert df["raw_text"].notna().all(), "some sampled ids were not found"

    # Safety check 1: hash of each fetched text must match Stage 1.
    # Plain-Python normalisation, so the result does not depend on
    # which pandas string backend is active.
    norm = pd.Series(
        [normalise_text(t) for t in df["raw_text"]], dtype=object
    )
    new_hash = pd.util.hash_pandas_object(norm, index=False).astype(str)
    meta = pd.read_csv(META, usecols=["complaint_id", "text_hash"], dtype=str)
    old_hash = df[["complaint_id"]].merge(meta, on="complaint_id", how="left")
    bad = new_hash.to_numpy() != old_hash["text_hash"].to_numpy()
    if bad.any():
        print("Mismatching ids:", list(df.loc[bad, "complaint_id"].head(10)))
    assert not bad.any(), f"{int(bad.sum())} texts do not match Stage 1 hashes"
    print("Hash check passed: every id has the right text", flush=True)

    # Final text: whitespace collapsed, case and XXXX tokens kept
    df["text"] = [collapse_whitespace(t) for t in df["raw_text"]]
    assert (df["text"].str.len() > 0).all(), "empty text found"

    df = df[["complaint_id", "date", "label", "split", "text"]]
    df = df.sort_values(["date", "complaint_id"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    back = pd.read_parquet(OUT)
    assert len(back) == len(df), "parquet round-trip lost rows"
    print(f"Saved {len(back):,} rows to {OUT}", flush=True)

    # Summary report (no narrative text is written to it)
    df["n_words"] = df["text"].str.split().str.len()
    df["has_xxxx"] = df["text"].str.contains("XXXX", regex=False)
    summary = df.groupby("split").agg(
        rows=("text", "size"),
        median_words=("n_words", "median"),
        p95_words=("n_words", lambda s: s.quantile(0.95)),
        max_words=("n_words", "max"),
        pct_with_xxxx=("has_xxxx", "mean"),
    )
    summary["pct_with_xxxx"] = (summary["pct_with_xxxx"] * 100).round(1)

    lines = [
        f"Total rows: {len(df):,}",
        "Columns: " + ", ".join(back.columns),
        "",
        "Per split:",
        summary.to_string(),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Done. See reports/dataset_summary.txt")


if __name__ == "__main__":
    main()