"""Evaluation helpers shared by every model."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def summarise(y_true, y_pred, labels):
    """Return (macro_f1, accuracy, per-class table) for one set of predictions."""
    macro = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    table = pd.DataFrame(report).T.loc[
        labels, ["precision", "recall", "f1-score", "support"]
    ]
    table["support"] = table["support"].astype(int)
    return macro, acc, table.round(3)