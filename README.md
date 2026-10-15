# ComplaintLens

Predicts which of 9 product categories a US consumer financial complaint
belongs to, from the complaint narrative alone. Built on the public CFPB
Consumer Complaint Database. It is a portfolio project, not a production
system and not financial or legal advice.

The aim was less to maximise a score than to measure honestly how far the
score can be trusted: a time-based split, a near-duplicate check, a
random-split comparison, a redaction-token test, and a latency measurement.
Every number below comes from a script in `scripts/` and a report in
`reports/`.

## Results

**Setup.** TF-IDF (1-2 word phrases, `min_df=5`, up to 200,000 features,
sublinear tf) + Logistic Regression (`C=1.0`). Trained on narratives from
2017 to 2022 (230,002 rows) and tested on 100,000 narratives from 2024, so the
model is always scored on complaints written after the ones it learned from.

| Metric (2024 test set) | Value |
|---|---|
| Macro-F1 | **0.7446** |
| Accuracy | **0.8498** |

Macro-F1 is the headline because the classes are very unbalanced: credit
reporting is about two thirds of the 2024 test sample, so accuracy alone
flatters the model.

**Per-class F1 (time split, 2024 test):**

| Class | F1 |
|---|---|
| Credit reporting | 0.918 |
| Mortgage | 0.869 |
| Student loan | 0.830 |
| Checking or savings account | 0.801 |
| Credit card or prepaid card | 0.745 |
| Money transfer or virtual currency | 0.690 |
| Debt collection | 0.689 |
| Vehicle loan or lease | 0.611 |
| Payday, title, or personal loan | 0.548 |

The two weakest classes are small and easily confused with their neighbours.
Treat their predictions with the most caution.

## Is the score inflated? Checks we ran

Details and limits are in [`docs/leakage_checks.md`](docs/leakage_checks.md).

| Check | Result |
|---|---|
| Near-duplicates (20,000 test narratives vs all training narratives) | 6.6% have a training twin at cosine similarity 0.90 or more, almost all in credit reporting. Removing them changes accuracy from 0.8529 to 0.8462 and macro-F1 from 0.7405 to 0.7416. |
| Random split instead of time split (same rows, 3 seeds) | Macro-F1 0.7400, 0.7371, 0.7377, slightly *below* the time split's 0.7446. Random splitting did not inflate the score. |
| Redaction tokens (the `XXXX` placeholders) removed | Macro-F1 0.7398 and accuracy 0.8428, against 0.7446 and 0.8498 with the tokens kept. The tokens are not the main shortcut. |

Near-duplicates flatter accuracy a little (the model is right 94.8% of the
time on twins) but barely affect macro-F1. The 0.90 threshold is our choice,
not a standard, and the near-duplicate check covers a 20,000-row sample.

## Speed

Measured on one machine: a 12-thread Intel CPU, Windows 11, Python 3.14.7,
scikit-learn 1.9.1, narratives with a median length of 146 words. Other
hardware will differ.

| One complaint at a time | Median | 95th percentile | 99th percentile |
|---|---|---|---|
| Prediction only | 1.14 ms | 2.46 ms | 3.93 ms |
| Prediction + top 5 words | 1.61 ms | 3.34 ms | 5.79 ms |

Batch of 1,000 complaints: 0.48 s (about 2,100 per second). Loading the model
takes 1.42 s, paid once at startup.

## Demo

A Streamlit app (`app/streamlit_app.py`) takes a complaint, shows the predicted
category, the probability for all 9 classes, and the 5 words that pushed
hardest toward the prediction. Nothing you type is stored or logged. The three
built-in examples are invented, not real complaints.

```
streamlit run app\streamlit_app.py --browser.gatherUsageStats false
```

Tested with Streamlit 1.66.0.

## Run it yourself

The data and trained models are **not** in this repository.

1. Create a virtual environment, then install the requirements and the package:
   `pip install -r requirements.txt`, then `pip install -e .`
   (the scripts import the `complaintlens` package from `src/`)
2. Download the CFPB complaint data into `data/raw/` (not included; the
   narratives are public but large).
3. Build the dataset: `python scripts\build_dataset.py`
4. Train and save the model: `python scripts\top_words.py`. On the machine
   above this took about 2 minutes for TF-IDF and about 16 minutes for the
   classifier.
5. Start the demo (above).

The other scripts in `scripts/` reproduce each check; their outputs are in
`reports/`.

## Limitations

- **Nine categories only.** Anything else is forced into one of them.
- **Time shift.** Trained on 2017 to 2022 and tested on 2024. Products and
  wording change, so scores on later years may be lower.
- **Probabilities are not calibrated.** We did not check whether "90%" means
  right 90% of the time, so treat them as a ranking, not a confidence.
- **Company names carry much of the signal** (see `reports/top_words.txt`), so
  the model may do worse on companies that were rare or absent in training.
- **Small classes are noisy.** Payday, title or personal loan and vehicle loan
  or lease have the lowest F1 and the fewest examples.
- **Single dataset, single model family.** No comparison with embedding models
  or calibration is included yet.

## Repository layout

```
app/        Streamlit demo
docs/       leakage checks and project log
reports/    output of every measurement script
scripts/    one script per step (data build, training, checks, latency)
src/        small shared package (labels, text cleaning, evaluation)
tests/      unit tests for labels and text cleaning
```

## Data

CFPB Consumer Complaint Database (public). This project is not affiliated with
or endorsed by the CFPB. Raw data, processed data, and trained models are
excluded from the repository by `.gitignore`.
