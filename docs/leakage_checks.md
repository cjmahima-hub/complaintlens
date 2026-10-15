## Leakage checks

**Question.** Could the model score well because test complaints are near-copies
of training complaints, or because of how the data was split?

**Check 1: near-duplicates.** We took a random sample of 20,000 of the 2024 test
narratives (seed 42) and compared each with all 230,002 training narratives
(TF-IDF on single words, words in more than 5% of narratives dropped, cosine
similarity). 29 sampled narratives had no usable words after filtering and
could not be judged.

- 6.6% of the sample (1,319) has a training narrative with similarity of 0.90
  or more; 10.3% at 0.80 or more.
- These twins are concentrated in credit reporting (1,249 of 1,319; 9.5% of
  that class). Every other class is at or below 2.1%.
- The model is more accurate on twins (94.8% correct at 0.90 or more, against
  85.3% for the whole sample), so near-duplicates flatter accuracy.
- Removing twins at 0.90 or more lowers accuracy from 0.8529 to 0.8462 (about
  0.7 points) and leaves macro-F1 essentially unchanged (0.7405 to 0.7416).

**Check 2: random split vs time split.** We re-split the same 380,002
narratives at random, keeping the same number of rows per class in each part,
and trained the same model with three seeds.

| Split | Test macro-F1 | Test accuracy |
|---|---|---|
| Time split (train 2017-22, test 2024) | 0.7446 | 0.8498 |
| Random, seeds 42 / 1 / 2 | 0.7400 / 0.7371 / 0.7377 | 0.8433 / 0.8417 / 0.8416 |

The random split did not score higher; it scored slightly lower (about 0.007
macro-F1). The time split was not the harder test on this sample.

**Conclusion.** Near-duplicates inflate accuracy slightly but not macro-F1.
Random splitting did not inflate scores. We chose the time split because it
matches deployment (predicting future complaints from past ones), and it cost
nothing in score here.

**Limits.**
- The near-duplicate check covers a 20,000-row sample, not all 100,000 test rows.
- The 0.90 and 0.80 thresholds are our choices, not a standard.
- Removing twins removes mostly credit reporting rows, so the class mix of the
  remaining sample changes too.
- We did not test why the time split scored slightly higher.
- Check 2 is one sample of 380,002 rows, not all 1.6 million narratives.