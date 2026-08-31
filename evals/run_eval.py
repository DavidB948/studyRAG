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

from studyrag.config import settings
from studyrag.db.writer import connect
from studyrag.modes.overview import explain

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


def _judge():
    """The judge model, wrapped for ragas.

    Deliberately not the smallest available model: ragas decomposes an answer into
    claims and rules on each, so a weak judge produces scores too noisy to act on.
    The judge is part of the eval's own failure surface.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.judge_model or settings.llm_model or "",
            api_key=settings.judge_api_key or settings.llm_api_key,
            base_url=settings.judge_base_url or settings.llm_base_url,
            temperature=0.0,  # a judge must be reproducible
        )
    )


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
        result = explain(conn, course, question)
        answer = " ".join(c.text for c in result.claims)
        contexts = [p.content for p in result.passages]

        if item["reference"] == NOT_IN_CORPUS:
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
        per_question.append({"id": item["id"], "course": course, "type": "ragas"})

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

    return {
        "run_at": datetime.now(UTC).isoformat(),
        "embed_model": settings.embed_model,
        "judge_model": settings.judge_model or settings.llm_model,
        "threshold": THRESHOLD,
        "by_course": by_course,
        "per_question": per_question,
    }


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

    print(f"\nwrote {RESULTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
