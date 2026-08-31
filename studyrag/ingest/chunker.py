"""RawDoc -> list[Chunk]. One slide becomes one chunk (small-to-big).

Structure here is positional, not typographic: the corpus has no markdown
headings and its numbering is noise. See DECISIONS.md.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from studyrag.types import Chunk, RawDoc, Slide

# A page whose title is one of these announces the section that follows.
SECTION_MARKERS = {"outline", "agenda", "contents", "overview", "roadmap"}

# Trailing page number on a footer line: "Xavier Bresson 4" -> "Xavier Bresson".
FOOTER_PAGE_NO = re.compile(r"\s*\d{1,4}\s*$")

# The lecture identifier, taken from slide 1: "Lecture 5 : Convolutional..." -> "Lecture 5".
LECTURE_ID = re.compile(r"\bLecture\s+\d+\b")

# A footer counts as one if it repeats on at least this share of pages.
FOOTER_MIN_SHARE = 0.6

# Below this a slide is a section divider or a "Questions?" card, not teaching content.
# Such a chunk still costs an embedding and can still surface in top-k, where it
# displaces something useful. Tuned by inspection, not by theory.
MIN_TOKENS = 15


def _strip_page_no(line: str) -> str:
    """Normalise a line by removing a trailing page number."""
    return FOOTER_PAGE_NO.sub("", line).strip()


def detect_footer(pages: list[str]) -> str | None:
    """Return the boilerplate line repeated across most pages, or None.

    Detected by repetition rather than hardcoded, so a different lecturer's deck
    works without a code change. Counting is per DISTINCT page: a line repeated
    five times on one page must not outrank a real footer.
    """
    if not pages:
        return None

    pages_per_line: Counter[str] = Counter()
    for page in pages:
        seen = {_strip_page_no(line) for line in page.splitlines() if line.strip()}
        pages_per_line.update(seen - {""})

    if not pages_per_line:
        return None

    line, count = pages_per_line.most_common(1)[0]
    return line if count / len(pages) >= FOOTER_MIN_SHARE else None


def parse_slide(page_text: str, page_no: int, footer: str | None) -> Slide:
    """Split one page into title and body, dropping the footer.

    The footer is filtered, not sliced off the end: pdf extraction order is
    unreliable and on slide 1 the footer lands third from the top.

    The page number is stripped from the title only when it matches this page's
    own number. Stripping any leading digit would corrupt a real title such as
    "3D Convolution".
    """
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if footer is not None:
        lines = [line for line in lines if _strip_page_no(line) != footer]

    if not lines:
        return Slide(page=page_no, title=None, body="")

    title = re.sub(rf"^{page_no}\s*", "", lines[0]).strip()
    return Slide(page=page_no, title=title or None, body="\n".join(lines[1:]))


def assign_sections(slides: list[Slide]) -> list[Slide]:
    """Propagate the current section onto every slide.

    An "Outline" slide announces the section that follows. The deck's first one
    is the full table of contents and names no single section, so it is told
    apart by having several body lines where the per-section ones have exactly
    one. Marker slides are emptied, not deleted, so page numbers stay intact and
    chunk_document can drop them by the same "no body" rule as any blank slide.
    """
    current: str | None = None
    out: list[Slide] = []

    for slide in slides:
        is_marker = slide.title is not None and slide.title.lower() in SECTION_MARKERS
        if is_marker:
            names = [line for line in slide.body.splitlines() if line.strip()]
            if len(names) == 1:
                current = names[0].strip()
            out.append(Slide(page=slide.page, title=slide.title, body="", section=current))
        else:
            out.append(Slide(page=slide.page, title=slide.title, body=slide.body, section=current))

    return out


def detect_lecture(slides: list[Slide]) -> str | None:
    """Pull the lecture identifier off the title slide, e.g. "Lecture 5"."""
    if not slides:
        return None
    head = "\n".join(filter(None, [slides[0].title, slides[0].body]))
    match = LECTURE_ID.search(head)
    return match.group(0) if match else None


def chunk_slides(doc: RawDoc, count_tokens: Callable[[str], int]) -> list[Chunk]:
    """One page is one chunk. No text is ever cut — only dropped.

    PyPDF already split the file at page boundaries, and for a deck those
    boundaries are the right ones. So this function does structure recovery
    (title, section, footer) rather than division.

    `page` is the printed slide number; `ordinal` is the position after markers
    and duplicates are dropped, so the two deliberately diverge.
    """
    footer = detect_footer(doc.pages)
    slides = assign_sections(
        [parse_slide(page, i, footer) for i, page in enumerate(doc.pages, start=1)]
    )
    lecture = detect_lecture(slides)

    chunks: list[Chunk] = []
    seen: set[str] = set()

    for slide in slides:
        if not slide.body.strip():
            continue  # marker slide, or a page whose only text was the footer

        content = "\n".join(filter(None, [slide.title, slide.body]))
        if content in seen:
            continue  # slides 28-30 are byte-identical; keep the first
        seen.add(content)

        token_count = count_tokens(content)
        if token_count < MIN_TOKENS:
            continue  # "Questions?", title cards, stray figure labels

        chunks.append(
            Chunk(
                course=doc.course,
                lecture=lecture,
                section=slide.section,
                ordinal=len(chunks),  # dense, unlike page
                page=slide.page,
                content=content,
                token_count=token_count,
            )
        )

    return chunks


def chunk_prose(doc: RawDoc, count_tokens: Callable[[str], int]) -> list[Chunk]:
    """Notes and textbooks: join the pages, then re-split on meaning.

    The inversion vs slides: a page break in flowing prose is an arbitrary cut,
    so page boundaries are discarded and new ones are chosen.

    Guidance:
      - Concatenate pages, keeping a char-offset -> page map so each chunk can
        still record the page it starts on. Citations must survive the join.
      - Split into blocks: markdown headings if the document has them, else
        blank-line paragraphs.
      - Accumulate blocks to settings.prose_target_tokens, then carry
        settings.prose_overlap_tokens of tail into the next chunk. Overlap
        exists here and not in slides because a paragraph continues an argument
        that the previous one started; a slide is self-contained.
      - Never break mid-paragraph unless one paragraph alone exceeds the target,
        and then break on a sentence boundary.
      - `section` is the nearest preceding heading.
    """
    raise NotImplementedError


def chunk_questions(doc: RawDoc, count_tokens: Callable[[str], int]) -> list[Chunk]:
    """Tutorials and past papers: one question is one chunk, never split.

    Joining pages first is required for correctness here, not an optimisation:
    a question routinely spans a page break, and half a question retrieves as
    noise.

    UNVALIDATED: written without a real past-paper file, so the boundary
    patterns are informed guesses. Say so rather than implying otherwise.

    Guidance:
      - Join all pages, then find question starts, most specific pattern first:
        "Question <n>", then "Q<n>." or "Q<n>)", then a bare "<n>." at line start.
      - One chunk per question, including every sub-part (a), (b), (c).
      - No size splitting, ever. The "never split a question" rule outranks the
        token target: if a question exceeds the model window, keep it whole and
        let token_count record the overflow, so truncation is visible in the
        data instead of silent.
      - `section` is the question number; that is also how a tutorial_soln links
        back to its tutorial_q later.
    """
    raise NotImplementedError


# Which unit a document is chunked into. The key is the unit, not the file format:
# extraction is shared in loaders.py and only the boundary rule differs.
STRATEGIES: dict[str, Callable[[RawDoc, Callable[[str], int]], list[Chunk]]] = {
    "slides": chunk_slides,
    "notes": chunk_prose,
    "tutorial_q": chunk_questions,
    "tutorial_soln": chunk_questions,
}


def chunk_document(doc: RawDoc, count_tokens: Callable[[str], int]) -> list[Chunk]:
    """Route a document to its chunking strategy.

    `count_tokens` is injected so this module never imports the embedding model —
    chunking stays testable without loading torch.
    """
    strategy = STRATEGIES.get(doc.doc_type)
    if strategy is None:
        raise ValueError(f"{doc.doc_path}: no chunking strategy for doc_type={doc.doc_type!r}")
    return strategy(doc, count_tokens)


def breadcrumb(chunk: Chunk) -> str:
    """Text actually fed to the embedder: hierarchy prefix, then content.

    Given, not TODO — it defines the contract Phase 3 relies on.
    """
    trail = [chunk.course, chunk.lecture, chunk.section]
    prefix = " > ".join(t for t in trail if t)
    return f"{prefix}\n{chunk.content}" if prefix else chunk.content
