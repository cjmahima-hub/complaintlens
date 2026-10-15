# ComplaintLens project log

## Goal
Predict the Product category of a CFPB consumer complaint from its narrative.
Start with a TF-IDF + Logistic Regression baseline, evaluate with a time-based
split, then add extensions.

## Environment (verified 12 Oct 2026)
- Windows, VS Code, PowerShell, Python 3.14.7
- Project folder: C:\dev\complaintlens, virtual environment in .venv
- pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2
  (exact versions in requirements.txt)

## Done
- Folder structure, .gitignore, first commit
- Raw data in data/raw: complaints.csv.zip (official bulk file, 336.9 MB zipped,
  5,559.6 MB unzipped) and cfpb_export_no_narratives.csv (same size, same columns)
- scripts/inspect_raw.py reads large CSVs in chunks

## Blocker
- Both files have 15 columns and no "Consumer complaint narrative" column.
- The CFPB announced on 14 Aug 2026 that it stopped publishing narratives.
- We need an archived snapshot from before 14 Aug 2026 that includes narratives.

## Rule for accepting a data source
1. It has a narrative column.
2. Its origin and licence are documented.
3. It was captured before 14 Aug 2026.

## Next
- Check DataLumos (CFPB Consumer Complaint Database project) and Kaggle.
- Note: login needed?, file names and sizes, version dates, licence wording.


## Update 14 Oct 2026: data source and label decisions
- Source: DataLumos copy of the CFPB Consumer Complaint Database (complaints.csv.zip, 989.1 MB, file dated 02/11/2025). Licence: CC BY 4.0 per the DataLumos terms; cite the CFPB and DataLumos. The terms forbid re-identifying research subjects (people or organizations), so we do not use the Company column and never commit raw data.
- Why not the live CFPB file: since 14 Aug 2026 it has no narrative column.
- Snapshot: 7,765,380 rows, 2,482,364 with narratives, complaint dates 2011-12-01 to 2025-02-09.
- Product names changed on 2017-04-24 and 2023-08-25. We use 2017-04-24 to 2024-12-31 and merge renamed labels into 9 classes (src/complaintlens/labels.py).
- Credit reporting is about 47% of narratives in 2019, 69% in 2022, 77% in 2024, so class balance drifts over time.

## Step 13-14: building the clean dataset (15 Oct 2026)

**Counts (from the full raw file, reports/window_counts.txt):**
- 2,482,364 narratives in total; 2,309,053 inside the date window; 2,307,388 with a mapped label.
- Exact-duplicate removal (keep earliest) left 1,631,959 unique narratives, so 675,429 (about 29%) were exact repeats.
- 2,038 identical texts carried different labels. This is a lower bound on label noise, because it only catches word-for-word matches.

**Sample (seed 42, scripts/sample_ids.py):**
- Train: 2017-2022, at most 5,000 per class per year, 230,002 rows.
- Validation: 2023, random sample with the natural class mix, 50,000 rows.
- Test: 2024, random sample with the natural class mix, 100,000 rows.
- Training uses a more balanced mix than validation and test on purpose. Later experiment: capped vs natural-mix training, compared on macro-F1.
- Output: data/processed/complaints_clean.parquet (380,002 rows, about 240 MB, gitignored).

**Lesson learned: the hash check caught a pandas/pyarrow difference.**
- After installing pyarrow, pandas 3.0 switched its text engine from plain Python to Arrow.
- The Arrow engine did not treat a non-breaking space or an em space as whitespace in a regex, while Python does. 9 of 380,002 texts hashed differently from Stage 1.
- Fix: one shared function, normalise_text in src/complaintlens/text.py, using plain Python. Regression tests are in tests/test_text.py.
- Rule: never normalise text with pandas .str methods.

**Class profile (reports/class_profile.txt):**
- 78% to 88% of narratives contain XXXX in every class, so presence alone is a weak signal.
- The number of XXXX tokens differs: credit reporting has median 10 and mean 25.4; most other classes have median 4 to 6.
- Median length ranges from 119 words (debt collection) to 221 (mortgage).
- To test next: a model using only length and XXXX count, compared with the majority-class baseline.