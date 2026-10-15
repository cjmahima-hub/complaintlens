"""Harmonise CFPB Product labels that were renamed over time.

The CFPB changed its list of Product names on 2017-04-24 and again on
2023-08-25. We only use complaints received between START_DATE and END_DATE,
so every raw label we keep maps cleanly to one clean label.
"""

START_DATE = "2017-04-24"  # first day of the 2017 Product list
END_DATE = "2024-12-31"  # 2025 is only January and under-filled

PRODUCT_MAP = {
    # Credit reporting: renamed in 2017 and again in 2023
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    # Labels that kept the same name throughout
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Checking or savings account": "Checking or savings account",
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Money transfer, virtual currency, or money service": "Money transfer or virtual currency",
    # Cards: merged in 2017, split again in 2023; we keep them merged
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    # Payday and personal loans: renamed in 2023
    "Payday loan, title loan, or personal loan": "Payday, title, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday, title, or personal loan",
}

_RETIRED = "retired by 2017-04-22, before START_DATE"

# Labels deliberately left out, with the reason (reused in the README).
EXCLUDED_LABELS = {
    "Debt or credit management": "new in 2023-08, so no older training data; only 1,702 narratives",
    "Consumer Loan": "retired 2017-04; split into two newer labels, so it cannot be mapped cleanly",
    "Bank account or service": _RETIRED,
    "Credit reporting": _RETIRED,
    "Payday loan": _RETIRED,
    "Money transfers": _RETIRED,
    "Other financial service": _RETIRED,
    "Virtual currency": _RETIRED,
}

CLEAN_LABELS = sorted(set(PRODUCT_MAP.values()))


def harmonise_product(raw_label):
    """Return the clean label for a raw Product value, or None if it is not used."""
    if not isinstance(raw_label, str):
        return None
    return PRODUCT_MAP.get(raw_label.strip())