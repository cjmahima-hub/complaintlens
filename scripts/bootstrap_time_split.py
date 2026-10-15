"""Step 20a: bootstrap confidence interval for the time-split test score.

Resamples the 2024 test rows (with replacement) 1,000 times, using the
predictions saved in Step 17, and reports the 95% range of macro-F1 and
accuracy.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TIME_PROBS = Path("data/processed/tfidf_lr_probs_test.parquet")
OUT = Path("reports/time_split_bootstrap.txt")
N_BOOT = 1000
SEED = 42


def main():
    tp = pd.read_parquet(TIME_PROBS)
    cls = [c for c in tp.columns if c not in ("complaint_id", "label")]
    k = len(cls)
    idx = {c: i for i, c in enumerate(cls)}
    y = tp["label"].map(idx).to_numpy()
    assert not np.isnan(y.astype(float)).any()
    p = tp[cls].to_numpy().argmax(axis=1)
    n = len(tp)

    def scores(yy, pp):
        cm = np.bincount(yy * k + pp, minlength=k * k).reshape(k, k)
        tp_ = np.diag(cm).astype(float)
        denom = cm.sum(axis=1) + cm.sum(axis=0)
        f1 = np.divide(2 * tp_, denom, out=np.zeros(k), where=denom > 0)
        return f1.mean(), tp_.sum() / len(yy)

    base_f1, base_acc = scores(y, p)
    print(f"Point estimate: macro-F1 {base_f1:.4f}, accuracy {base_acc:.4f}")

    rng = np.random.default_rng(SEED)
    f1s, accs = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        f, a = scores(y[s], p[s])
        f1s.append(f)
        accs.append(a)
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    ac_lo, ac_hi = np.percentile(accs, [2.5, 97.5])

    lines = [
        f"Bootstrap of the time-split test score ({n:,} rows, "
        f"{N_BOOT} resamples, seed {SEED})",
        f"macro-F1: {base_f1:.4f}  (95% range {f1_lo:.4f} to {f1_hi:.4f})",
        f"accuracy: {base_acc:.4f}  (95% range {ac_lo:.4f} to {ac_hi:.4f})",
        "Random split (seed 42) scored macro-F1 0.7400, accuracy 0.8433.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()