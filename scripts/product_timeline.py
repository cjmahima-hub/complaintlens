"""Show when each Product label was used, among complaints that have a narrative."""

import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/raw/datalumos_download.zip")
REPORT_PATH = Path("reports/product_timeline.txt")
CHUNK_SIZE = 200_000
DATE_COL = "Date received"
PRODUCT_COL = "Product"
TEXT_COL = "Consumer complaint narrative"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    year_parts = []
    bound_parts = []
    total_rows = 0

    for chunk in pd.read_csv(
        path, usecols=[DATE_COL, PRODUCT_COL, TEXT_COL], dtype=str, chunksize=CHUNK_SIZE
    ):
        total_rows += len(chunk)
        print(f"  processed {total_rows:,} rows...", flush=True)

        has_text = chunk[TEXT_COL].fillna("").str.strip() != ""
        chunk = chunk.loc[has_text, [DATE_COL, PRODUCT_COL]].copy()
        chunk["date"] = pd.to_datetime(chunk[DATE_COL], errors="coerce", format="mixed")
        chunk = chunk.dropna(subset=["date"])
        chunk["year"] = chunk["date"].dt.year.astype(int)

        year_parts.append(chunk.groupby([PRODUCT_COL, "year"]).size())
        bound_parts.append(chunk.groupby(PRODUCT_COL)["date"].agg(["min", "max"]))

    by_year = pd.concat(year_parts).groupby(level=[0, 1]).sum()
    year_table = by_year.unstack(fill_value=0).astype(int)

    bounds = pd.concat(bound_parts).groupby(level=0).agg(
        first_date=("min", "min"), last_date=("max", "max")
    )
    summary = pd.DataFrame(
        {
            "first_date": bounds["first_date"].dt.date,
            "last_date": bounds["last_date"].dt.date,
            "narratives": year_table.sum(axis=1),
        }
    ).sort_values("first_date")

    text = (
        "FIRST AND LAST DATE EACH PRODUCT LABEL APPEARS (rows with narratives)\n\n"
        + summary.to_string()
        + "\n\nNARRATIVES PER YEAR AND PRODUCT\n\n"
        + year_table.to_string()
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main() 