"""Text normalisation shared by every pipeline step.

Uses plain Python string operations on purpose. In pandas 3.0 the
.str methods behave differently depending on whether pyarrow is
installed, which changes results on unusual Unicode characters.
"""
import re

_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace runs to one space, strip the ends."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def collapse_whitespace(text: str) -> str:
    """Collapse whitespace runs to one space and strip; keep the case."""
    return _WHITESPACE.sub(" ", text).strip()