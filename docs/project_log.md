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