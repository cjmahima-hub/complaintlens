"""Quick check: does pandas .str give the same result as plain Python?"""
import pandas as pd

from complaintlens.text import normalise_text

samples = [
    "plain  text\n",
    "Hello\u00a0World",
    "TEST\u2003Text",
    "tab\there",
]
s = pd.Series(samples)
print("pandas string backend:", type(s.array).__name__)

pandas_way = (
    s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)
for raw, p in zip(samples, pandas_way):
    py = normalise_text(raw)
    verdict = "same" if p == py else "DIFFERENT"
    print(verdict, "| input:", ascii(raw), "| pandas:", ascii(p), "| python:", ascii(py))