"""Quiz mode: generate revision questions over a scope.

Retrieval here is a metadata filter, not a similarity search. There is no query to
embed: the student picks a lecture, and the material for that lecture is by
definition in scope. No embedder, no reranker, no relevance gate.

The work is deciding WHICH material becomes questions. A lecture is 20-70 chunks and
a quiz is 10 questions, so something has to choose, and the choice is visible to the
student as coverage.
"""

from __future__ import annotations

import logging

import psycopg
from pydantic import BaseModel

from studyrag.grounding import quote_supported
from studyrag.llm import complete_json
from studyrag.prompts import QUIZ_SYSTEM, QUIZ_USER

log = logging.getLogger(__name__)

# Used only when a deck has no recovered sections at all. Lecture 6 is exactly this
# case: 66 chunks, every section NULL. Reading order is the only structure left.
FALLBACK_BUCKET_CHUNKS = 8

MAX_QUESTIONS = 25

# A section can lose its question to the quote check, or the model can just write
# fewer than asked. One extra call re-asks only the sections still short. Capped at
# one: a section that failed twice has nothing to ask about, and retrying forever
# would spend calls to prove it.
MAX_TOPUP_ROUNDS = 1


class Bucket(BaseModel):
    """A unit of quizzable material.

    Either a recovered section or, when the deck has none, a contiguous run of
    slides. Both are ordinal ranges, which is why one type covers both: sections are
    assigned in reading order, so a section IS a range.
    """

    document_id: int
    doc_path: str
    lecture: str | None
    label: str            # section heading, or "slides 1-8" for a fallback range
    first_ordinal: int
    last_ordinal: int     # inclusive
    n_chunks: int

    @property
    def key(self) -> str:
        """Identity, as opposed to `label`, which is only for display.

        Two documents in one lecture can produce the same heading, and the fallback
        labels collide outright: every deck has a "slides 1-8". Counting coverage or
        looking a section up by label would merge them.
        """
        return f"{self.document_id}:{self.first_ordinal}"


class QuizQuestion(BaseModel):
    """Raw model output for one question, before its citation is resolved."""

    question: str
    answer: str
    supporting_quote: str
    section_index: int


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]


class GradedQuestion(BaseModel):
    """One question a caller gets. The citation was written here, not by the model."""

    question: str
    answer: str
    citation: str
    section: str          # the label, for display
    section_key: str      # identity, for coverage and for the eval's lookup
    supporting_quote: str


class Quiz(BaseModel):
    """A quiz plus the evidence that it covered the lecture.

    `coverage` is reported alongside the questions on purpose: a quiz drawn entirely
    from one section is a worse quiz, and the student should be able to see that
    without running an eval.
    """

    course: str
    lecture: str | None
    n_requested: int
    questions: list[GradedQuestion]
    sections_covered: int
    sections_available: int
    # Questions the model produced that were thrown away, and why. Reported rather
    # than hidden so a quiz that quietly shrank is distinguishable from a lecture
    # that only had five questions in it.
    n_generated: int
    n_dropped_bad_section: int
    n_dropped_unverified: int
    n_dropped_over_quota: int
    n_topup_rounds: int


def fetch_buckets(conn: psycopg.Connection, course: str, lecture: str | None) -> list[Bucket]:
    """Every section in scope, as ordinal ranges, in reading order.

    Grouping happens in SQL because the chunks table already has what a bucket is:
    min and max ordinal per (document, section). Sections with no heading come back
    as one NULL group per document, which `plan_buckets` then splits.
    """
    rows = conn.execute(
        """
        SELECT c.document_id, d.doc_path, d.lecture, c.section,
               min(c.ordinal), max(c.ordinal), count(*)
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.course = %(course)s
          AND (%(lecture)s::text IS NULL OR d.lecture = %(lecture)s)
        GROUP BY c.document_id, d.doc_path, d.lecture, c.section
        ORDER BY d.doc_path, min(c.ordinal)
        """,
        {"course": course, "lecture": lecture},
    ).fetchall()

    return [
        Bucket(
            document_id=r[0], doc_path=r[1], lecture=r[2],
            label=r[3] or f"slides {r[4] + 1}-{r[5] + 1}",
            first_ordinal=r[4], last_ordinal=r[5], n_chunks=r[6],
        )
        for r in rows
    ]


def fetch_bucket_text(conn: psycopg.Connection, bucket: Bucket) -> str:
    """The slides of one bucket, in reading order, as a single block of text."""
    rows = conn.execute(
        """
        SELECT content FROM chunks
        WHERE document_id = %s AND ordinal BETWEEN %s AND %s
        ORDER BY ordinal
        """,
        (bucket.document_id, bucket.first_ordinal, bucket.last_ordinal),
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


def split_unsectioned(bucket: Bucket) -> list[Bucket]:
    """One unsectioned run of slides, cut into fixed-size ranges.

    A last resort, and labelled as one: the student sees "slides 17-24", not a
    heading, so a quiz built this way never claims structure the deck did not have.
    """
    parts: list[Bucket] = []
    for start in range(bucket.first_ordinal, bucket.last_ordinal + 1, FALLBACK_BUCKET_CHUNKS):
        end = min(start + FALLBACK_BUCKET_CHUNKS - 1, bucket.last_ordinal)
        parts.append(
            bucket.model_copy(update={
                "label": f"slides {start + 1}-{end + 1}",
                "first_ordinal": start,
                "last_ordinal": end,
                "n_chunks": end - start + 1,
            })
        )
    return parts


def plan_buckets(buckets: list[Bucket]) -> list[Bucket]:
    """Raw sections -> the buckets a quiz is actually allocated over.

    One correction only: an unsectioned run is cut by reading order. Small sections
    are deliberately left alone. Merging a one-slide section into its neighbour would
    make the merged bucket cite the neighbour's heading for a fact that came from the
    absorbed slide, and a citation that names the wrong section is worse than a quiz
    that skips a thin one. `allocate` already drops the smallest buckets when there
    are more of them than questions.
    """
    planned: list[Bucket] = []
    for bucket in buckets:
        if bucket.label.startswith("slides ") and bucket.n_chunks > FALLBACK_BUCKET_CHUNKS:
            planned.extend(split_unsectioned(bucket))
        else:
            planned.append(bucket)
    return planned


def allocate(buckets: list[Bucket], n_questions: int) -> list[int]:
    """How many questions each bucket gets. Returns one count per bucket, in order.

    Largest-remainder apportionment with a floor of one:

    - Every bucket is guaranteed at least one question, so a short section cannot be
      skipped entirely. That is what makes coverage a real metric rather than a
      side effect of section length.
    - The questions left over after the floor are shared out in proportion to
      bucket size, so a 15-slide section ends up with more than a 4-slide one.
    - Proportional shares are fractional and question counts are not, so the
      remainders decide: sort by the fractional part and give the spare questions to
      the largest ones first. Ties break toward the bigger bucket.
    - When there are more buckets than questions the floor cannot hold. Cover the
      largest `n_questions` buckets, one each, and leave the rest out — a quiz that
      touches every section once is not possible, and pretending otherwise would
      silently drop questions later.

    The returned list sums to exactly `min(n_questions, len(buckets))` questions or
    `n_questions`, whichever the floor allows, and aligns with `buckets` index for
    index, because the caller maps counts back onto buckets by position.
    """
    if not buckets or n_questions <= 0:
        return [0] * len(buckets)

    by_size_desc = sorted(
        range(len(buckets)), key=lambda i: (-buckets[i].n_chunks, i)
    )

    # Fewer questions than buckets: the floor cannot hold, so cover the biggest.
    if n_questions < len(buckets):
        counts = [0] * len(buckets)
        for index in by_size_desc[:n_questions]:
            counts[index] = 1
        return counts

    spare = n_questions - len(buckets)
    if spare == 0:
        return [1] * len(buckets)

    total_chunks = sum(b.n_chunks for b in buckets)
    shares = [spare * b.n_chunks / total_chunks for b in buckets]
    extra = [int(share) for share in shares]

    # int() truncated every share, so `spare - sum(extra)` questions are still unspent.
    # They go to the buckets that lost the most to truncation, biggest bucket first
    # on a tie.
    by_remainder_desc = sorted(
        range(len(buckets)),
        key=lambda i: (-(shares[i] - extra[i]), -buckets[i].n_chunks, i),
    )
    for index in by_remainder_desc[: spare - sum(extra)]:
        extra[index] += 1

    return [1 + e for e in extra]


def _format_sections(buckets: list[Bucket], quotas: list[int], texts: list[str]) -> str:
    """Numbered sections with their quotas, as the model sees them."""
    blocks = []
    for index, (bucket, quota, text) in enumerate(zip(buckets, quotas, texts, strict=True)):
        blocks.append(f"[section {index}] {bucket.label} — write {quota} question(s)\n{text}")
    return "\n\n---\n\n".join(blocks)


def _already_asked(questions: list[GradedQuestion]) -> str:
    """The questions a top-up round must not repeat."""
    if not questions:
        return ""
    asked = "\n".join(f"- {q.question}" for q in questions)
    return (
        "\nThese questions have already been written for this lecture. "
        f"Ask about something else in the section:\n{asked}\n"
    )


def generate_quiz(
    conn: psycopg.Connection, course: str, lecture: str | None, n_questions: int = 10
) -> Quiz:
    """Pick buckets, allocate questions across them, generate, then resolve citations.

    One LLM call for the whole quiz, plus at most one top-up round for the sections
    that came back empty or lost a question to the quote check. Per-bucket calls would
    ground each question slightly harder but cost one call per section, and the
    section_index bounds check below already catches the failure that would prevent.
    """
    if not 1 <= n_questions <= MAX_QUESTIONS:
        raise ValueError(f"n_questions must be 1..{MAX_QUESTIONS}, got {n_questions}")

    buckets = plan_buckets(fetch_buckets(conn, course, lecture))
    if not buckets:
        raise LookupError(f"no material ingested for course={course!r} lecture={lecture!r}")

    quotas = allocate(buckets, n_questions)
    chosen = [(b, q) for b, q in zip(buckets, quotas, strict=True) if q > 0]
    texts = [fetch_bucket_text(conn, b) for b, _ in chosen]

    questions: list[GradedQuestion] = []
    generated = 0
    dropped_bad_section = 0
    dropped_unverified = 0
    dropped_over_quota = 0
    written: dict[int, int] = {}
    rounds = 0

    # Round 0 asks every section for its full quota; each later round re-asks only the
    # sections still short, so a section that already delivered is never asked twice.
    while True:
        outstanding = {i: q - written.get(i, 0) for i, (_, q) in enumerate(chosen)}
        outstanding = {i: n for i, n in outstanding.items() if n > 0}
        if not outstanding:
            break

        indices = sorted(outstanding)
        response = complete_json(
            system=QUIZ_SYSTEM,
            user=QUIZ_USER.format(
                scope=lecture or f"{course}, all lectures",
                sections=_format_sections(
                    [chosen[i][0] for i in indices],
                    [outstanding[i] for i in indices],
                    [texts[i] for i in indices],
                ),
            )
            + _already_asked(questions),
            schema=QuizResponse,
            # Pinned, unlike ask mode. Ask wants some sampling variety for the analogy
            # and the exam angle; quiz is structured extraction, and at 0.2 the same
            # lecture returned 8 questions on one run and 1 on the next.
            temperature=0.0,
        )
        rounds += 1
        generated += len(response.questions)

        for item in response.questions:
            # The model numbers sections within the round it was given, so its index
            # is translated back to the quiz-wide one before anything is cited.
            if not 0 <= item.section_index < len(indices):
                log.warning("dropping question citing section %d of %d",
                            item.section_index, len(indices))
                dropped_bad_section += 1
                continue

            index = indices[item.section_index]
            bucket, quota = chosen[index]

            # The quota is the allocation's contract, and the prompt is not a
            # guarantee: one lecture came back with 16 questions for 8 requested.
            if written.get(index, 0) >= quota:
                dropped_over_quota += 1
                continue

            if not quote_supported(item.supporting_quote, texts[index]):
                # The model named evidence that is not in the section it pointed at.
                # The answer may still be right, but nothing here can show that, and a
                # quiz answer without working evidence is what this mode exists to
                # avoid. The slot goes back into the pool for the next round.
                log.warning("dropping question with unverifiable quote in %r", bucket.label)
                dropped_unverified += 1
                continue

            written[index] = written.get(index, 0) + 1
            questions.append(
                GradedQuestion(
                    question=item.question,
                    answer=item.answer,
                    citation=(
                        f"[{bucket.doc_path} | {bucket.lecture or 'no lecture'} | {bucket.label}]"
                    ),
                    section=bucket.label,
                    section_key=bucket.key,
                    supporting_quote=item.supporting_quote,
                )
            )

        if rounds > MAX_TOPUP_ROUNDS:
            break

    if dropped_over_quota:
        log.info("dropped %d question(s) over quota", dropped_over_quota)
    if len(questions) < min(n_questions, len(chosen)):
        # The prompt lets the model write fewer than asked for a thin section, so this
        # is expected, not an error. It is logged and reported rather than padded.
        log.info("quiz returned %d of %d requested questions", len(questions), n_questions)

    return Quiz(
        course=course,
        lecture=lecture,
        n_requested=n_questions,
        questions=questions,
        sections_covered=len({q.section_key for q in questions}),
        sections_available=len(buckets),
        n_generated=generated,
        n_dropped_bad_section=dropped_bad_section,
        n_dropped_unverified=dropped_unverified,
        n_dropped_over_quota=dropped_over_quota,
        n_topup_rounds=rounds - 1,
    )
