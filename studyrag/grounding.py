"""Does a quoted span actually come from the text it claims to?

Shared by ask and quiz: both let the model name its own evidence, and both have to
check that the evidence exists before a citation is written next to it. Extracted
when quiz became the second caller, not before.
"""

from __future__ import annotations

import re

# A quote must share at least this share of its words with the cited source.
# Exact substring matching was too strict: the model re-punctuates and re-wraps as it
# copies, so correct, grounded quotes were being dropped. Word overlap tolerates that
# drift while still failing a quote the model invented, which shares little with any
# source. Deliberately a blunt instrument — a cross-encoder would judge this properly.
QUOTE_OVERLAP = 0.75


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def quote_supported(quote: str, source: str) -> bool:
    """Is this quote actually drawn from this source text?"""
    quoted = words(quote)
    if not quoted:
        return False
    if " ".join(quoted) in " ".join(words(source)):
        return True  # exact after normalisation; the common case
    present = set(words(source))
    return sum(w in present for w in quoted) / len(quoted) >= QUOTE_OVERLAP
