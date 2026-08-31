"""Overview mode: explain a concept from the student's own material.

Plain RAG — retrieve, then generate. The orchestration is deliberately thin; all
the retrieval judgement lives in retrieve.py and all the wording in prompts.py.
"""

from __future__ import annotations

import logging
import re

import psycopg
from pydantic import BaseModel

from studyrag.llm import complete_json
from studyrag.prompts import NO_CONTEXT, OVERVIEW_SYSTEM, OVERVIEW_USER
from studyrag.retrieve import Passage, retrieve

log = logging.getLogger(__name__)


class Claim(BaseModel):
    """One grounded statement. Scored by ragas faithfulness."""

    text: str
    passage_index: int
    supporting_quote: str


class OverviewResponse(BaseModel):
    """Raw model output, before citations are resolved."""

    claims: list[Claim]
    analogy: str | None = None
    exam_angle: str | None = None
    common_trap: str | None = None
    check_yourself: list[str] = []
    not_covered: str | None = None


class GroundedClaim(BaseModel):
    """A claim whose citation was written by this code, not by the model."""

    text: str
    citation: str


class Overview(BaseModel):
    """What a caller gets. Grounded and generated stay separate all the way out."""

    question: str
    course: str

    # Grounded: every claim is quote-verified against its cited passage, and this is
    # the only field ragas faithfulness is scored over.
    claims: list[GroundedClaim]

    # Model-generated teaching scaffolding. Extra-corpus BY DESIGN, so it is kept in
    # separate fields and excluded from faithfulness — scoring it would penalise the
    # analogy for not being in the slides, which is the point of an analogy.
    analogy: str | None = None
    exam_angle: str | None = None
    common_trap: str | None = None
    check_yourself: list[str] = []

    not_covered: str | None = None
    passages: list[Passage] = []     # the exact context the model saw

    def explanation(self) -> str:
        """The claims as flowing prose.

        The prompt asks for claims that read as consecutive sentences, so joining
        them needs no second LLM call and no unverified connective text. Every
        sentence keeps its own citation, which for revision is a feature: the
        student can see which slide each sentence came from.
        """
        return " ".join(c.text for c in self.claims)


# A quote must share at least this share of its words with the cited passage.
# Exact substring matching was too strict: the model re-punctuates and re-wraps as it
# copies, so correct, grounded claims were being dropped. Word overlap tolerates that
# drift while still failing a quote the model invented, which shares little with any
# passage. Deliberately a blunt instrument — a cross-encoder would judge this properly.
QUOTE_OVERLAP = 0.75

# A claim that talks ABOUT the context rather than stating something FROM it.
# Word overlap cannot catch these — "the passage does not give the formula" shares most
# of its words with the passage — and they are not claims at all: they belong in
# not_covered. Left as an explicit rule rather than more prompt wording, because the
# prompt already forbids them and the model still produces them.
META_CLAIM = re.compile(
    r"\b(passages?|slides?|context|excerpts?|document)\b.{0,40}?"
    r"\b(do(es)? not|don't|doesn't|no|never|lack|omit|fail)\b"
    r"|\b(not|never)\b.{0,20}?\b(mentioned|provided|given|stated|shown|covered|"
    r"specified|described|included)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _quote_supported(quote: str, passage: str) -> bool:
    """Is this quote actually drawn from this passage?"""
    quoted = _words(quote)
    if not quoted:
        return False
    if " ".join(quoted) in " ".join(_words(passage)):
        return True  # exact after normalisation; the common case
    present = set(_words(passage))
    return sum(w in present for w in quoted) / len(quoted) >= QUOTE_OVERLAP


def _format_passages(passages: list[Passage]) -> str:
    """Number the passages so the model can reference them by index."""
    return "\n\n".join(
        f"[{i}] {p.citation()}\n{p.content}" for i, p in enumerate(passages)
    )


def explain(conn: psycopg.Connection, course: str, question: str) -> Overview:
    """Retrieve, then explain. Citations are resolved here, never by the model.

    An out-of-range passage_index means the model invented a source, so the claim
    is dropped rather than shown with a guessed citation. A bounds check catches
    that; a hallucinated filename would not be caught by anything.
    """
    passages = retrieve(conn, course, question)

    if not passages:
        return Overview(question=question, course=course, claims=[], not_covered=NO_CONTEXT)

    response = complete_json(
        system=OVERVIEW_SYSTEM,
        user=OVERVIEW_USER.format(question=question, passages=_format_passages(passages)),
        schema=OverviewResponse,
    )

    claims: list[GroundedClaim] = []
    for claim in response.claims:
        if not 0 <= claim.passage_index < len(passages):
            log.warning("dropping claim citing passage %d of %d", claim.passage_index, len(passages))
            continue

        # The quote must actually be in the passage. This is the difference between
        # asking the model to stay grounded and requiring it: a claim drawn from the
        # model's own knowledge has no real span to quote, and a claim ABOUT the
        # passages ("X is not mentioned") has none either. Both fail here.
        if META_CLAIM.search(claim.text):
            log.warning("dropping meta-claim: %s", claim.text[:80])
            continue

        passage = passages[claim.passage_index]
        if not _quote_supported(claim.supporting_quote, passage.content):
            log.warning("dropping unverified claim: %s", claim.text[:80])
            continue

        claims.append(GroundedClaim(text=claim.text, citation=passage.citation()))

    return Overview(
        question=question,
        course=course,
        claims=claims,
        analogy=response.analogy,
        exam_angle=response.exam_angle,
        common_trap=response.common_trap,
        check_yourself=response.check_yourself,
        not_covered=response.not_covered,
        passages=passages,
    )
