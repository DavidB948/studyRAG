"""Retrieval: embed the query, search, expand small hits into big context.

Small-to-big. Slides average ~60 tokens, so a single hit is precise but thin;
embedding whole sections instead would average several ideas into one vector and
match everything weakly. So: embed per slide, expand at read time.

Every query filters by course. That is an invariant, not an option — see CLAUDE.md.
"""

from __future__ import annotations

import psycopg
from pydantic import BaseModel

from studyrag.embed import embed_query
from studyrag.rerank import score as rerank_score

# Stage 1 casts wide: the bi-encoder only has to get the right chunk into the
# candidate set, and the cross-encoder decides the order. Recall matters here,
# precision does not.
DEFAULT_K = 20
DEFAULT_MAX_PASSAGES = 4

# Fallback context when a chunk has no section (title slides, decks with no Outline).
WINDOW_RADIUS = 1

# Loose cosine prefilter only. It is deliberately permissive: stage 1 exists to
# build a candidate set, and anything it discards the reranker never sees.
MIN_SCORE = 0.30

# Relevance floor on the RERANKER's score, which unlike cosine is a trained
# probability rather than a geometric byproduct.
#
# Swept against the golden set rather than picked by eye. Top-1 scores:
#   answerable    0.0428  0.2442  0.5322  0.5554  0.9604 ... 0.9969
#   unanswerable  0.0025  0.8778
# Any value in [0.005, 0.02] keeps 10/10 answerable questions; 0.05 starts dropping
# real answers. 0.02 is the top of that band, leaving ~2x margin under the lowest
# true positive.
#
# The 0.8778 outlier is not a calibration failure. That question asks for a formula
# the deck renders as an IMAGE, on a slide literally titled "Exact formula for the
# convolution layer" — so the passage is genuinely relevant and genuinely lacks the
# answer. No threshold can separate that; it is caught downstream by claim
# verification instead. Retrieval relevance and answer presence are different
# properties and need different mechanisms.
MIN_RERANK_SCORE = 0.02

# Relative gate: keep only hits scoring at least this share of the TOP hit, so the
# cutoff scales with how strong the best match was. Catches a different failure from
# the absolute floor — "one strong match plus three also-rans" rather than "nothing
# here matches at all".
#
# 0.10 where the cosine version used 0.85, because the multiplier is scale-dependent:
# cosine was compressed into ~0.4-0.8, while reranker scores span 0.0000-0.9969. At
# 0.85 a top hit of 0.96 would set a floor of 0.82 and discard everything else.
RELATIVE_FLOOR = 0.10


class Hit(BaseModel):
    """One chunk returned by vector search, before expansion."""

    chunk_id: int
    document_id: int
    doc_path: str
    lecture: str | None
    section: str | None
    ordinal: int
    page: int | None
    content: str
    score: float  # cosine similarity in [0, 1] until rerank replaces it


class Passage(BaseModel):
    """What generation actually sees: expanded context plus its citation."""

    course: str
    doc_path: str
    lecture: str | None
    section: str | None
    pages: list[int]
    content: str
    score: float  # inherited from the best hit inside this passage

    def citation(self) -> str:
        parts = [self.doc_path, self.lecture, self.section]
        where = " | ".join(p for p in parts if p)
        pages = f" p{self.pages[0]}" if len(self.pages) == 1 else f" pp{self.pages[0]}-{self.pages[-1]}"
        return f"[{where}{pages}]"


# --- SQL: given, not TODO ------------------------------------------------------

def vector_search(
    conn: psycopg.Connection, course: str, query_vector: list[float], k: int
) -> list[Hit]:
    """Top-k chunks by cosine similarity, within one course.

    `<=>` is pgvector's cosine DISTANCE operator (0 = identical), and it is the
    operator the HNSW index was built for. Similarity is 1 - distance.

    The course filter is inside the SQL, never applied afterwards in Python: a
    post-filter would let another course's rows consume the top-k slots first.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, d.doc_path, d.lecture, c.section,
               c.ordinal, c.page, c.content,
               1 - (c.embedding <=> %(q)s::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.course = %(course)s
        ORDER BY c.embedding <=> %(q)s::vector
        LIMIT %(k)s
        """,
        {"q": query_vector, "course": course, "k": k},
    ).fetchall()

    return [
        Hit(
            chunk_id=r[0], document_id=r[1], doc_path=r[2], lecture=r[3], section=r[4],
            ordinal=r[5], page=r[6], content=r[7], score=r[8],
        )
        for r in rows
    ]


def fetch_section(conn: psycopg.Connection, document_id: int, section: str) -> list[tuple]:
    """Every chunk of one section, in reading order. Returns (ordinal, page, content)."""
    return conn.execute(
        """
        SELECT ordinal, page, content FROM chunks
        WHERE document_id = %s AND section = %s
        ORDER BY ordinal
        """,
        (document_id, section),
    ).fetchall()


def fetch_window(
    conn: psycopg.Connection, document_id: int, ordinal: int, radius: int = WINDOW_RADIUS
) -> list[tuple]:
    """Neighbouring chunks, for hits with no section. Returns (ordinal, page, content)."""
    return conn.execute(
        """
        SELECT ordinal, page, content FROM chunks
        WHERE document_id = %s AND ordinal BETWEEN %s AND %s
        ORDER BY ordinal
        """,
        (document_id, ordinal - radius, ordinal + radius),
    ).fetchall()


# --- Core logic: yours --------------------------------------------------------

def expand(
    conn: psycopg.Connection,
    course: str,
    hits: list[Hit],
    max_passages: int = DEFAULT_MAX_PASSAGES,
    min_score: float = MIN_RERANK_SCORE,
    relative_floor: float = RELATIVE_FLOOR,
) -> list[Passage]:
    """Turn ranked hits into deduplicated, expanded passages.

    Hits arrive in score order and passages leave in that same order, so the
    best-matching material stays first in the LLM's context.

    Deduplication is on the assembled ordinals rather than on the section key:
    two hits in one section collapse to one passage, and a window fallback can
    also land inside a section already emitted.
    """
    if not hits:
        return []

    # Both gates are absolute values, so one comparison per hit suffices.
    cutoff = max(min_score, hits[0].score * relative_floor)

    passages: list[Passage] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    for hit in hits:
        if hit.score < cutoff:
            break  # hits are ranked, so everything after this is worse too

        if hit.section is not None:
            rows = fetch_section(conn, hit.document_id, hit.section)
        else:
            rows = fetch_window(conn, hit.document_id, hit.ordinal)

        if not rows:  # defensive: the hit itself should always come back
            continue

        key = (hit.document_id, tuple(r[0] for r in rows))
        if key in seen:
            continue  # same span already emitted, by a higher-scoring hit
        seen.add(key)

        passages.append(
            Passage(
                course=course,
                doc_path=hit.doc_path,
                lecture=hit.lecture,
                section=hit.section,
                pages=sorted({r[1] for r in rows if r[1] is not None}),
                content="\n\n".join(r[2] for r in rows),
                score=hit.score,  # the best hit inside this span earned its rank
            )
        )

        if len(passages) == max_passages:
            break

    return passages


def retrieve(
    conn: psycopg.Connection,
    course: str,
    query: str,
    k: int = DEFAULT_K,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    min_score: float = MIN_RERANK_SCORE,
) -> list[Passage]:
    """Embed the query, search wide, rerank, expand.

    Two stages because neither model can do both jobs: the bi-encoder is fast enough
    to search but compresses each passage before the query exists, and the
    cross-encoder reads both texts together but needs one forward pass per pair.
    Cheap-and-wide, then expensive-and-narrow.

    `course` is a required argument with no default: a retrieval with no course
    scope is a bug, not a broad search.

    An empty list is a real answer — "this corpus does not cover that" — and the
    caller must handle it rather than treating it as an error.
    """
    hits = vector_search(conn, course, embed_query(query), k)
    if not hits:
        return []

    # The reranker's score replaces cosine from here on: same field, better scale.
    for hit, relevance in zip(hits, rerank_score(query, [h.content for h in hits]), strict=True):
        hit.score = relevance
    hits.sort(key=lambda h: h.score, reverse=True)

    return expand(conn, course, hits, max_passages, min_score)
