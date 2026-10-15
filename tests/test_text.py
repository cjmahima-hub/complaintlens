from complaintlens.text import collapse_whitespace, normalise_text


def test_normalise_lowercases_and_collapses():
    assert normalise_text("  Hello   WORLD\n") == "hello world"


def test_normalise_handles_unicode_spaces():
    assert normalise_text("Hello\u00a0World") == "hello world"
    assert normalise_text("TEST\u2003Text") == "test text"


def test_collapse_keeps_case_and_xxxx():
    assert collapse_whitespace("  XXXX  paid\tme \n") == "XXXX paid me"