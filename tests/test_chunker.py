"""Tests for the pure chunking logic.

Fixtures are synthetic: data/raw/ is gitignored, so a test that opens the real deck
cannot run in CI. Only the pure functions are tested — anything touching the model,
the network or Postgres is not unit-testable and is covered by the eval instead.
"""

from studyrag.ingest.chunker import (
    MAX_TOKENS,
    assign_sections,
    chunk_slides,
    detect_footer,
    detect_lecture,
    parse_slide,
    split_oversized,
)
from studyrag.types import RawDoc, Slide


def words(text: str) -> int:
    """Stand-in for the real tokenizer, so tests never load torch."""
    return len(text.split())


# Distinct WORDS per page, not just distinct trailing numbers: normalisation strips the
# page number, so "body text 1" and "body text 2" are the same line and would tie with
# the footer for most-repeated.
NOUNS = "apples bridges canyons dolphins engines forests glaciers harbours islands jungles"


def deck(n: int, footer: str = "Xavier Bresson") -> list[str]:
    words_ = NOUNS.split()
    return [
        f"{i}Topic{words_[i - 1]}\n{words_[i - 1]} discussed at length here\n{footer} {i}"
        for i in range(1, n + 1)
    ]


# --- detect_footer ---------------------------------------------------------------

def test_footer_found_despite_varying_page_numbers():
    assert detect_footer(deck(10)) == "Xavier Bresson"


def test_no_footer_when_nothing_repeats():
    # Lines must differ in WORDS, not just in a trailing number: normalisation strips
    # the page number, so "body 1" and "body 2" are the same line by design.
    pages = ["1Alpha\napples", "2Beta\nbridges", "3Gamma\ncanyons", "4Delta\ndolphins"]
    assert detect_footer(pages) is None


def test_repetition_within_one_page_does_not_beat_a_real_footer():
    pages = deck(5)
    pages[0] += "\nnoise\nnoise\nnoise\nnoise\nnoise"
    assert detect_footer(pages) == "Xavier Bresson"


def test_empty_deck():
    assert detect_footer([]) is None


# --- parse_slide -----------------------------------------------------------------

def test_page_number_is_stripped_from_the_title():
    assert parse_slide("4Motivation\nbody", 4, None).title == "Motivation"


def test_a_title_starting_with_a_digit_survives():
    # Only THIS page's number is stripped, so "3D Convolution" on page 7 is untouched.
    assert parse_slide("73D Convolution\nbody", 7, None).title == "3D Convolution"


def test_footer_is_filtered_not_sliced():
    # Extraction order is unreliable: on slide 1 the footer lands third from the top.
    slide = parse_slide("1Title\nXavier Bresson 1\nreal body", 1, "Xavier Bresson")
    assert "Xavier Bresson" not in slide.body
    assert slide.body == "real body"


# --- assign_sections -------------------------------------------------------------

def test_section_propagates_forward_and_markers_are_emptied():
    slides = [
        Slide(page=1, title="Outline", body="A\nB"),      # table of contents: names nothing
        Slide(page=2, title="Outline", body="Section A"),  # marker: names the next section
        Slide(page=3, title="Content", body="teaching"),
    ]
    out = assign_sections(slides)
    assert out[0].section is None and out[0].body == ""
    assert out[2].section == "Section A"


# --- split_oversized -------------------------------------------------------------

def test_oversized_slide_splits_and_every_part_keeps_the_title():
    body = "\n".join(f"line {i} with several words in it" for i in range(200))
    parts = split_oversized("Backpropagation", body, words)
    assert len(parts) > 1
    assert all(p.startswith("Backpropagation\n") for p in parts)
    assert all(words(p) <= MAX_TOKENS for p in parts)


# --- chunk_slides ----------------------------------------------------------------

def _doc(pages: list[str]) -> RawDoc:
    return RawDoc(
        course="CS5242", doc_path="l5.pdf", doc_type="slides",
        content_hash="deadbeef", pages=pages,
    )


def test_duplicate_slides_are_dropped_and_ordinals_stay_dense():
    body = "\n".join(f"word{i}" for i in range(40))
    # Byte-identical slides, as pages 28-30 of the real deck are.
    pages = [f"{i}Same title\n{body}\nXavier Bresson {i}" for i in range(1, 4)]
    chunks = chunk_slides(_doc(pages), words, footer="Xavier Bresson")
    assert len(chunks) == 1
    assert [c.ordinal for c in chunks] == [0]


def test_page_is_the_printed_number_and_ordinal_is_position():
    body = "\n".join(f"word{i}" for i in range(40))
    pages = [
        "1Outline\nSection A\nXavier Bresson 1",     # marker: dropped
        f"2Real slide\n{body}\nXavier Bresson 2",
        f"3Another\n{body} extra\nXavier Bresson 3",
    ]
    chunks = chunk_slides(_doc(pages), words, footer="Xavier Bresson")
    assert [c.ordinal for c in chunks] == [0, 1]
    assert [c.page for c in chunks] == [2, 3]   # ordinal != page, by design


def test_lecture_is_read_from_the_title_slide():
    slides = [Slide(page=1, title=None, body="Lecture 5 : Convolutional Neural Networks")]
    assert detect_lecture(slides) == "Lecture 5"
