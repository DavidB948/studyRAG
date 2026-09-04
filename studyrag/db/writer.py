"""Ingest writes. One transaction per document.

Per-document rather than per-run so a single bad file cannot roll back an entire
ingest: a constraint violation poisons its transaction, and every later statement
in it fails until rollback.
"""

from __future__ import annotations

import logging

import psycopg
from pgvector.psycopg import register_vector

from studyrag.config import settings
from studyrag.types import Chunk, RawDoc

log = logging.getLogger(__name__)


def connect() -> psycopg.Connection:
    """Connect in autocommit mode so `conn.transaction()` means what it says.

    Without autocommit, psycopg opens an implicit transaction on the first
    statement — here the duplicate pre-check — and a later `conn.transaction()`
    block then degrades to a SAVEPOINT inside it. Exiting the block releases the
    savepoint but commits nothing, so `conn.close()` rolls the whole run back
    while every statement reported success.
    """
    conn = psycopg.connect(settings().database_url, autocommit=True)
    register_vector(conn)
    return conn


def find_duplicate(conn: psycopg.Connection, course: str, content_hash: str) -> str | None:
    """Return the doc_path already holding these bytes in this course, or None.

    The pre-check exists because the alternative — insert and catch the unique
    violation — only fails *after* parsing and embedding the whole file. One
    indexed SELECT is worth roughly a thousand times less than the work it saves.
    The constraint stays as the backstop, since this check is advisory and racy.
    """
    row = conn.execute(
        "SELECT doc_path FROM documents WHERE course = %s AND content_hash = %s",
        (course, content_hash),
    ).fetchone()
    return row[0] if row else None


def upsert_document(conn: psycopg.Connection, doc: RawDoc, lecture: str | None) -> int:
    """Insert or refresh the document row, returning its id."""
    row = conn.execute(
        """
        INSERT INTO documents (course, doc_path, content_hash, doc_type, lecture)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (course, doc_path) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            doc_type     = EXCLUDED.doc_type,
            lecture      = EXCLUDED.lecture,
            updated_at   = now()
        RETURNING id
        """,
        (doc.course, doc.doc_path, doc.content_hash, doc.doc_type, lecture),
    ).fetchone()
    assert row is not None  # RETURNING on an upsert always yields a row
    return row[0]


def upsert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Write chunks, then remove any the document no longer has.

    The delete matters: upsert only touches rows it writes, so a file edited from
    31 slides down to 25 would leave ordinals 25-30 behind as orphans that still
    match queries.
    """
    conn.cursor().executemany(
        """
        INSERT INTO chunks (document_id, course, section, ordinal, page,
                            content, token_count, embedding, embed_model)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id, ordinal) DO UPDATE SET
            course      = EXCLUDED.course,
            section     = EXCLUDED.section,
            page        = EXCLUDED.page,
            content     = EXCLUDED.content,
            token_count = EXCLUDED.token_count,
            embedding   = EXCLUDED.embedding,
            embed_model = EXCLUDED.embed_model,
            updated_at  = now()
        """,
        [
            (
                document_id,
                c.course,
                c.section,
                c.ordinal,
                c.page,
                c.content,
                c.token_count,
                v,
                settings().embed_model,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )
    conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND ordinal >= %s",
        (document_id, len(chunks)),
    )
