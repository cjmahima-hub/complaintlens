"""First look at the raw CFPB complaints CSV, read in chunks to save memory."""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_PATH = Path("reports/raw_data_summary.txt")
CHUNK_SIZE = 200_000
DATE_COL = "Date received"
PRODUCT_COL = "Product"
TEXT_COL = "Consumer complaint narrative"


def pick_file() -> Path:
    """Use the path given on the command line, or the only CSV in data/raw."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if len(csv_files) != 1:
        print(f"Expected exactly one .csv in {RAW_DIR}, found {len(csv_files)}.")
        print("Files currently in that folder:")
        for f in sorted(RAW_DIR.iterdir()):
            print("  ", f.name)
        sys.exit(1)
    return csv_files[0]


def main() -> None:
    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    path = pick_file()
    emit(f"File: {path.name} ({path.stat().st_size / 1e6:,.1f} MB)")

    header = pd.read_csv(path, nrows=0)
    emit(f"\nColumns ({len(header.columns)}):")
    for i, col in enumerate(header.columns, start=1):
        emit(f"  {i:>2}. {col}")

    missing = [c for c in (DATE_COL, PRODUCT_COL, TEXT_COL) if c not in header.columns]
    if missing:
        emit(f"\nExpected columns not found: {missing}")
        emit("Paste the column list above and we will adjust the script.")
        sys.exit(1)

    total_rows = 0
    rows_with_text = 0
    all_counts = pd.Series(dtype="float64")
    text_counts = pd.Series(dtype="float64")
    first_date = last_date = last_text_date = None
    bad_dates = 0
    date_examples = None
    example_text = None

    for chunk in pd.read_csv(
        path, usecols=[DATE_COL, PRODUCT_COL, TEXT_COL], dtype=str, chunksize=CHUNK_SIZE
    ):
        total_rows += len(chunk)
        has_text = chunk[TEXT_COL].fillna("").str.strip() != ""
        rows_with_text += int(has_text.sum())

        all_counts = all_counts.add(chunk[PRODUCT_COL].value_counts(), fill_value=0)
        text_counts = text_counts.add(
            chunk.loc[has_text, PRODUCT_COL].value_counts(), fill_value=0
        )

        dates = pd.to_datetime(chunk[DATE_COL], errors="coerce", format="mixed")
        bad_dates += int(dates.isna().sum())
        if date_examples is None:
            date_examples = chunk[DATE_COL].head(3).tolist()
        if dates.notna().any():
            lo, hi = dates.min(), dates.max()
            first_date = lo if first_date is None else min(first_date, lo)
            last_date = hi if last_date is None else max(last_date, hi)
        if (has_text & dates.notna()).any():
            hi_text = dates[has_text].max()
            last_text_date = hi_text if last_text_date is None else max(last_text_date, hi_text)

        if example_text is None and has_text.any():
            example_text = chunk.loc[has_text, TEXT_COL].iloc[0]

    emit(f"\nRows in file: {total_rows:,}")
    emit(f"Rows with a narrative: {rows_with_text:,} ({rows_with_text / total_rows:.1%})")
    emit(f"Date range, all rows: {first_date} to {last_date}")
    emit(f"Latest date among rows WITH a narrative: {last_text_date}")
    emit(f"Raw date examples: {date_examples} (unparseable dates: {bad_dates:,})")

    table = (
        pd.DataFrame({"all_rows": all_counts, "with_narrative": text_counts})
        .fillna(0)
        .astype(int)
    )
    table["share_of_narratives_%"] = (
        table["with_narrative"] / table["with_narrative"].sum() * 100
    ).round(1)
    table = table.sort_values("with_narrative", ascending=False)
    emit(f"\nProduct categories: {len(table)}")
    emit(table.to_string())

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary saved to {REPORT_PATH}")

    if example_text:
        print("\nExample narrative (first 300 characters):")
        print(example_text[:300])


if __name__ == "__main__":
    main()