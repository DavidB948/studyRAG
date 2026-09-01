"""The judge model, shared by the ask and quiz eval harnesses.

Deliberately not the smallest available model: ragas decomposes an answer into
claims and rules on each, so a weak judge produces scores too noisy to act on. The
judge is part of the eval's own failure surface.

`JUDGE_*` settings fall back to `LLM_*`. Using one model to both generate and grade
risks self-preference bias, so they are separate knobs even when they hold the same
value — the gap is then a config change, not a code change.
"""

from __future__ import annotations

from studyrag.config import settings

THRESHOLD = 0.80


def judge():
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
