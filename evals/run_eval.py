"""Score the pipeline against the golden set. Per course, never in aggregate.

Two things are measured, because they fail differently:

  1. Answerable questions -> ragas, four metrics. Faithfulness alone would pass a
     system that grounds a confident answer in the wrong section, so precision and
     recall (which need a reference) are scored too.
  2. NOT_IN_CORPUS questions -> abstention rate AND faithfulness. Nothing in ragas
     tests whether a retriever knows its limits: a system that always returns k
     passages scores the same as one that says "I don't have this". Faithfulness is
     scored here too, because these are exactly the questions where a model reaches
     for outside knowledge — routing them past the metric would hide the hallucination
     the abstention check exists to detect. They get no context_precision or
     context_recall: those need a reference answer, and there is none.

Usage:  uv run python evals/run_eval.py
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.judge import judge as _judge
from studyrag.config import settings
from studyrag.db.writer import connect
from studyrag.modes.ask import answer_question

log = logging.getLogger("eval")

HERE = Path(__file__).parent
GOLD = HERE / "gold_questions.json"
RESULTS = HERE / "latest_results.json"

NOT_IN_CORPUS = "NOT_IN_CORPUS"
THRESHOLD = 0.80


class LocalEmbeddings:
    """Adapter so ragas scores answer_relevancy with the same model retrieval uses.

    Using a different embedding model for the metric than for the system would
    measure agreement between two models rather than the system's own behaviour.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from studyrag.embed import embed_passages

        return embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        from studyrag.embed import embed_query

        return embed_query(text)


def run() -> dict[str, Any]:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    gold = json.loads(GOLD.read_text())
    conn = connect()

    samples: list[dict[str, Any]] = []
    abstain_samples: list[dict[str, Any]] = []
    abstention: dict[str, list[bool]] = defaultdict(list)
    per_question: list[dict[str, Any]] = []

    for item in gold:
        course, question = item["course"], item["question"]
        try:
            result = answer_question(conn, course, question)
        except Exception as exc:  # noqa: BLE001 - one bad response must not lose the run
            # A malformed model response used to abort the entire eval, losing every
            # score computed so far. Record it as a failure and continue: a generation
            # failure is a result, not a reason to have no results.
            log.error("%s failed: %s", item["id"], exc)
            per_question.append(
                {"id": item["id"], "course": course, "type": "error", "error": str(exc)[:200]}
            )
            continue
        answer = " ".join(c.text for c in result.claims)
        contexts = [p.content for p in result.passages]

        # Trust `kind` as well as the sentinel: a drafted question is routed by what it
        # IS, not by whether the drafting model happened to phrase the reference the
        # exact way this harness matches on. Getting this wrong sends an unanswerable
        # question to ragas, where the metrics assume an answer exists.
        if item["reference"] == NOT_IN_CORPUS or item.get("kind") == "unanswerable":
            # Correct behaviour is naming the gap, not staying silent. Every surviving
            # claim is already quote-verified against its passage, so a related grounded
            # fact alongside an admitted gap is a better answer than nothing — demanding
            # zero claims would score the more useful behaviour as a failure. What must
            # never happen is asserting an answer that is not there, and that is what
            # abstention_faithfulness measures.
            abstained = bool(result.not_covered)
            abstention[course].append(abstained)
            entry = {
                "id": item["id"],
                "course": course,
                "type": "abstention",
                "passed": abstained,
                "n_claims": len(result.claims),
            }

            if result.claims:
                # Claims were made on a question the corpus cannot answer. Whether they
                # are grounded is the whole question, so send them to the judge.
                abstain_samples.append(
                    {
                        "user_input": question,
                        "response": answer,
                        "retrieved_contexts": contexts or [""],
                        "reference": "",
                    }
                )
            else:
                # No claims means nothing ungrounded was asserted. Vacuously faithful,
                # and recorded without spending a judge call.
                entry["faithfulness"] = 1.0

            per_question.append(entry)
            continue

        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["reference"],
            }
        )
        per_question.append(
            {
                "id": item["id"],
                "course": course,
                "type": "ragas",
                "kind": item.get("kind", "conceptual"),
            }
        )

    if not samples:
        # ragas evaluate() raises on an empty dataset, which would bury the reason.
        failures = [q for q in per_question if q["type"] == "error"]
        first = failures[0]["error"] if failures else "no answerable questions in the set"
        raise SystemExit(f"nothing to score: {len(failures)} question(s) errored. First: {first}")

    judge = _judge()
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())  # type: ignore[arg-type]

    scores = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=embeddings),
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge),
        ],
    )

    if abstain_samples:
        # Faithfulness only: precision and recall need a reference answer, and for a
        # question the corpus cannot answer there is none to compare against.
        abstain_scores = evaluate(
            dataset=EvaluationDataset.from_list(abstain_samples),
            metrics=[Faithfulness(llm=judge)],
        ).to_pandas()
        scored_abstentions = [
            p for p in per_question if p["type"] == "abstention" and p["n_claims"]
        ]
        for row, item in zip(
            abstain_scores.to_dict("records"), scored_abstentions, strict=True
        ):
            item["faithfulness"] = row["faithfulness"]

    df = scores.to_pandas()
    ragas_items = [p for p in per_question if p["type"] == "ragas"]
    for row, item in zip(df.to_dict("records"), ragas_items, strict=True):
        item.update({k: v for k, v in row.items() if isinstance(v, float)})

    by_course: dict[str, Any] = {}
    for course in {item["course"] for item in gold}:
        rows = [i for i in ragas_items if i["course"] == course]
        # NaN means the judge call failed (rate limit, timeout), not that the system
        # scored zero. Averaging them in would report an infrastructure problem as a
        # quality failure, so they are excluded and counted separately.
        metrics: dict[str, float] = {}
        scored: dict[str, int] = {}
        for m in ("faithfulness", "answer_relevancy",
                  "llm_context_precision_with_reference", "context_recall"):
            values = [r[m] for r in rows if m in r and not math.isnan(r[m])]
            scored[m] = len(values)
            if values:
                metrics[m] = round(sum(values) / len(values), 4)
        flags = abstention.get(course, [])
        abstain_rows = [
            i for i in per_question
            if i["type"] == "abstention" and i["course"] == course
            and not math.isnan(i.get("faithfulness", float("nan")))
        ]
        by_course[course] = {
            "n_answerable": len(rows),
            "n_abstention": len(flags),
            "abstention_rate": round(sum(flags) / len(flags), 4) if flags else None,
            "abstention_faithfulness": (
                round(sum(i["faithfulness"] for i in abstain_rows) / len(abstain_rows), 4)
                if abstain_rows else None
            ),
            "metrics": metrics,
            "scored": scored,
            "unscored": {m: len(rows) - n for m, n in scored.items() if n < len(rows)},
            "below_threshold": [m for m, v in metrics.items() if v < THRESHOLD],
        }

    # Per-kind, because an average over mixed query types hides both directions: a
    # lexical weakness that dense retrieval is known for disappears into a mean
    # dominated by conceptual questions it handles well.
    by_kind: dict[str, Any] = {}
    for kind in {i.get("kind") for i in ragas_items}:
        rows = [i for i in ragas_items if i.get("kind") == kind]
        by_kind[str(kind)] = {
            "n": len(rows),
            # None, not 0.0, when every question in the kind failed to score: a
            # missing measurement and a measured zero must not read the same.
            **{
                m: _mean_or_none([r.get(m, float("nan")) for r in rows])
                for m in ("faithfulness", "answer_relevancy",
                          "llm_context_precision_with_reference", "context_recall")
            },
        }

    return {
        "run_at": datetime.now(UTC).isoformat(),
        "by_kind": by_kind,
        "embed_model": settings.embed_model,
        "judge_model": settings.judge_model or settings.llm_model,
        "threshold": THRESHOLD,
        "by_course": by_course,
        "per_question": per_question,
    }


def _mean_or_none(values: list[float]) -> float | None:
    """Mean over the values that scored, or None when none of them did."""
    scored = [v for v in values if not math.isnan(v)]
    return round(sum(scored) / len(scored), 4) if scored else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run()
    RESULTS.write_text(json.dumps(results, indent=2))

    failed = False
    for course, data in results["by_course"].items():
        print(f"\n{course}  ({data['n_answerable']} answerable, {data['n_abstention']} abstention)")
        for metric in data["scored"]:
            if metric not in data["metrics"]:
                print(f"  {metric:42s}    --   NOT SCORED (all {len(data['scored'])} judge calls failed)")
                continue
            value = data["metrics"][metric]
            mark = "PASS" if value >= THRESHOLD else "FAIL"
            missing = data["unscored"].get(metric, 0)
            note = f"  ({missing} unscored)" if missing else ""
            print(f"  {metric:42s} {value:.3f}  {mark}{note}")
        if data["abstention_rate"] is not None:
            print(f"  {'abstention_rate':42s} {data['abstention_rate']:.3f}")
        if data["abstention_faithfulness"] is not None:
            value = data["abstention_faithfulness"]
            mark = "PASS" if value >= THRESHOLD else "FAIL"
            print(f"  {'abstention_faithfulness':42s} {value:.3f}  {mark}")
        failed |= bool(data["below_threshold"])

    print("\nby question kind:")
    print(f"  {'kind':14s} {'n':>3s}  {'faith':>6s} {'relev':>6s} {'prec':>6s} {'recall':>6s}")
    for kind, d in sorted(results["by_kind"].items()):
        cells = " ".join(
            f"{d[m]:6.3f}" if d[m] is not None else f"{'  n/a':>6s}"
            for m in ("faithfulness", "answer_relevancy",
                      "llm_context_precision_with_reference", "context_recall")
        )
        print(f"  {kind:14s} {d['n']:3d}  {cells}")

    print(f"\nwrote {RESULTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
