import math

import pytest

from complaintlens.labels import CLEAN_LABELS, harmonise_product


def test_credit_reporting_renames_merge():
    old = "Credit reporting, credit repair services, or other personal consumer reports"
    new = "Credit reporting or other personal consumer reports"
    assert harmonise_product(old) == "Credit reporting"
    assert harmonise_product(new) == "Credit reporting"


def test_card_labels_merge():
    names = ["Credit card", "Prepaid card", "Credit card or prepaid card"]
    assert {harmonise_product(n) for n in names} == {"Credit card or prepaid card"}


def test_payday_labels_merge():
    names = [
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
    ]
    assert {harmonise_product(n) for n in names} == {"Payday, title, or personal loan"}


@pytest.mark.parametrize(
    "raw", ["Debt or credit management", "Consumer Loan", "", None, math.nan]
)
def test_unused_or_missing_labels_return_none(raw):
    assert harmonise_product(raw) is None


def test_whitespace_is_ignored():
    assert harmonise_product("  Mortgage ") == "Mortgage"


def test_number_of_clean_labels():
    assert len(CLEAN_LABELS) == 9