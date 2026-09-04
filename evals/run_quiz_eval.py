"""Score quiz mode. Separate harness, because quiz fails differently from ask.

The golden set does not apply here. It exists to score a question -> answer task:
every item pairs a question with a reference answer, and three of the four ragas
metrics need one or both. Quiz mode has neither — there is no user question, and
nothing to compare a generated question against. Reusing the golden set would
measure nothing.

What can be measured, and what each catches that the others miss:

  1. faithfulness — is the model answer supported by the section it was drawn from?
     The only ragas metric that survives the change of task, because it needs a
     response and a context and nothing else.
  2. coverage — how much of the lecture the quiz actually touched. A quiz that is
     faithful ten times over to one section is a bad quiz, and faithfulness rates it
     perfect. No LLM call: it is arithmetic over section labels.
  3. quote validity — the share of generated questions whose evidence was found in
     the section they cited. Measured, not assumed: generate_quiz drops the failures,
     so without this number a quiz that silently halved would look like a short one.

Usage:  uv run python evals/run_quiz_eval.py
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.judge import THRESHOLD, judge
from studyrag.config import settings
from studyrag.db.writer import connect
from studyrag.modes.quiz import fetch_bucket_text, fetch_buckets, generate_quiz, plan_buckets

log = logging.getLogger("quiz-eval")

RESULTS = Path(__file__).parent / "latest_quiz_results.json"
N_QUESTIONS = 8


def scopes(conn) -> list[tuple[str, str]]:
    """Every (course, lecture) with material ingested. The quiz scope IS the golden set."""
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT DISTINCT course, lecture FROM documents "
            "WHERE lecture IS NOT NULL ORDER BY course, lecture"
        ).fetchall()
    ]


def section_text(conn, course: str, lecture: str) -> dict[str, str]:
    """Bucket key -> full text, for every bucket in scope.

    Rebuilt here rather than returned by generate_quiz: the harness must look the
    material up independently, or it would be grading the model against a context
    the model itself chose to report.
    """
    return {
        bucket.key: fetch_bucket_text(conn, bucket)
        for bucket in plan_buckets(fetch_buckets(conn, course, lecture))
    }


def run() -> dict[str, Any]:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness

    conn = connect()
    by_scope: dict[str, Any] = {}
    samples: list[dict] = []
    sample_scope: list[str] = []
    # Per question, because a scope average hides which question failed, and the
    # question text is the only thing that explains a low score.
    per_question: list[dict] = []

    for course, lecture in scopes(conn):
        key = f"{course} / {lecture}"
        sections = section_text(conn, course, lecture)

        try:
            quiz = generate_quiz(conn, course, lecture, N_QUESTIONS)
        except Exception as exc:  # noqa: BLE001 - one bad lecture must not lose the run
            log.error("%s failed: %s", key, exc)
            by_scope[key] = {"error": str(exc)[:200]}
            continue

        for question in quiz.questions:
            samples.append(
                {
                    "user_input": question.question,
                    "response": question.answer,
                    "retrieved_contexts": [sections.get(question.section_key, "")],
                }
            )
            sample_scope.append(key)
            per_question.append(
                {"scope": key, "section": question.section, "question": question.question}
            )

        # n_generated counts what the model produced; the rest is what survived the
        # bounds check and the quote check.
        # Over-quota questions were valid, just surplus, so they do not count against
        # quote validity — only invented sources and unfindable evidence do.
        verified = quiz.n_generated - quiz.n_dropped_unverified - quiz.n_dropped_bad_section
        by_scope[key] = {
            "n_requested": quiz.n_requested,
            "n_generated": quiz.n_generated,
            "n_kept": len(quiz.questions),
            "sections_covered": quiz.sections_covered,
            "sections_available": quiz.sections_available,
            "coverage": round(quiz.sections_covered / quiz.sections_available, 4),
            "quote_validity": round(verified / quiz.n_generated, 4) if quiz.n_generated else None,
            "n_dropped_over_quota": quiz.n_dropped_over_quota,
            "n_topup_rounds": quiz.n_topup_rounds,
        }

    conn.close()

    if not samples:
        raise SystemExit("nothing to score: no lecture produced a usable quiz")

    scores = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(llm=judge())],
    ).to_pandas()

    for key, scope_results in by_scope.items():
        rows = [
            scores["faithfulness"][i]
            for i, scope in enumerate(sample_scope)
            if scope == key and not math.isnan(scores["faithfulness"][i])
        ]
        scope_results["faithfulness"] = round(sum(rows) / len(rows), 4) if rows else None

    for entry, value in zip(per_question, scores["faithfulness"], strict=True):
        entry["faithfulness"] = None if math.isnan(value) else round(float(value), 4)

    return {
        "run_at": datetime.now(UTC).isoformat(),
        "judge_model": settings().judge_model or settings().llm_model,
        "n_questions_requested": N_QUESTIONS,
        "threshold": THRESHOLD,
        "by_scope": by_scope,
        "per_question": sorted(
            per_question, key=lambda q: (q["faithfulness"] is not None, q["faithfulness"])
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run()
    RESULTS.write_text(json.dumps(results, indent=2))

    failed = False
    print(f"\n{'scope':22s} {'kept':>9s} {'faith':>6s} {'cover':>6s} {'quote':>6s}")
    for key, d in results["by_scope"].items():
        if "error" in d:
            print(f"  {key:20s} ERROR {d['error'][:40]}")
            failed = True
            continue
        cells = []
        for metric in ("faithfulness", "coverage", "quote_validity"):
            value = d[metric]
            cells.append(f"{value:6.3f}" if value is not None else f"{'n/a':>6s}")
        print(f"  {key:20s} {d['n_kept']:3d}/{d['n_requested']:<5d} {' '.join(cells)}")

        # Only faithfulness gates. Coverage is bounded by how many sections a lecture
        # has against how many questions were asked, so a low number can be correct;
        # it is reported for reading, not for passing.
        if d["faithfulness"] is not None and d["faithfulness"] < THRESHOLD:
            failed = True

    print(f"\nwrote {RESULTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
