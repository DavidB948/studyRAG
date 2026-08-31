"""Ingest the corpus: discover -> load -> chunk -> embed -> write.

Usage:
    uv run python scripts/ingest.py            # everything under corpus_root
    uv run python scripts/ingest.py --course CS5242
    uv run python scripts/ingest.py --dry-run  # no DB, no model download
"""

from __future__ import annotations

import argparse
import logging
import sys

from studyrag.db.writer import connect, find_duplicate, upsert_chunks, upsert_document
from studyrag.embed import count_tokens as model_count_tokens
from studyrag.embed import embed_passages
from studyrag.ingest.chunker import chunk_document, detect_lecture, parse_slide
from studyrag.ingest.loaders import discover, identify, load
from studyrag.types import RawDoc

log = logging.getLogger("ingest")


def _lecture_of(doc: RawDoc) -> str | None:
    """Document-level lecture id, derived the same way the chunker does."""
    slides = [parse_slide(page, i, None) for i, page in enumerate(doc.pages, start=1)]
    return detect_lecture(slides)


def _word_count(text: str) -> int:
    """Stand-in for --dry-run, so inspecting chunk shape never loads the model."""
    return len(text.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", help="only ingest this course directory")
    parser.add_argument("--dry-run", action="store_true", help="chunk only; no embed, no write")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = [p for p in discover() if not args.course or f"/{args.course}/" in f"/{p}/"]
    if not paths:
        log.error("no documents found under the corpus root")
        return 1

    # studyrag.embed imports torch lazily, so this costs nothing until the model
    # is actually used — the laziness lives there, not in a conditional import here.
    count_tokens = _word_count if args.dry_run else model_count_tokens
    conn = None if args.dry_run else connect()
    ingested = skipped = total_chunks = 0

    for path in paths:
        # Hash first: identifying a file is cheap, extracting its text is not, so an
        # unchanged file costs one read and one SELECT rather than a full parse.
        course, doc_path, content_hash = identify(path)

        if conn is not None:
            existing = find_duplicate(conn, course, content_hash)
            if existing is not None:
                log.info("skip %s: same bytes already ingested as %s", doc_path, existing)
                skipped += 1
                continue

        doc = load(path)

        try:
            chunks = chunk_document(doc, count_tokens)
        except NotImplementedError as exc:
            log.warning("skip %s: %s", doc.doc_path, exc)
            skipped += 1
            continue

        total_chunks += len(chunks)
        log.info("%s [%s] -> %d chunks", doc.doc_path, doc.doc_type, len(chunks))

        if conn is None:
            continue

        # One transaction per document: a failure here must not roll back the run.
        with conn.transaction():
            document_id = upsert_document(conn, doc, _lecture_of(doc))
            upsert_chunks(conn, document_id, chunks, embed_passages([c.content for c in chunks]))
        ingested += 1

    if conn is not None:
        conn.close()

    log.info("done: %d ingested, %d skipped, %d chunks", ingested, skipped, total_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
