"""Shared data shapes for ingest."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DocType = Literal["notes", "slides", "tutorial_q", "tutorial_soln"]


class RawDoc(BaseModel):
    """One source file, extracted but not yet parsed.

    `pages` is one entry per PDF page (= per slide). Text files produce a
    single page; the chunker treats that as a document with no slide structure.
    """

    course: str
    doc_path: str
    doc_type: DocType
    content_hash: str   # SHA-256 of the raw file bytes; dedupe guard, not identity
    pages: list[str]


class Slide(BaseModel):
    """One page after parsing: title separated from body, footer removed."""

    page: int          # 1-based, as printed on the slide
    title: str | None
    body: str
    section: str | None = None


class Chunk(BaseModel):
    """One row of the `chunks` table, before embedding.

    `lecture` is carried from the parent document rather than stored on the chunk
    row — it is here only because `breadcrumb()` needs it at embedding time.
    """

    course: str
    lecture: str | None
    section: str | None
    ordinal: int
    page: int | None   # printed slide number; ordinal is post-filter position, not this
    content: str
    token_count: int
