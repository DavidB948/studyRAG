"""One LLM client, behind one interface.

The provider is OpenAI-compatible on purpose: swapping Gemini for the NUS API or
a local Ollama is a change to two .env values, not a change to any code.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from studyrag.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not settings().llm_api_key or not settings().llm_base_url:
        raise RuntimeError("LLM_API_KEY and LLM_BASE_URL must be set in .env")
    return OpenAI(api_key=settings().llm_api_key, base_url=settings().llm_base_url)


def complete_json[T: BaseModel](
    system: str,
    user: str,
    schema: type[T],
    temperature: float = 0.2,
    retries: int = 1,
) -> T:
    """One LLM call, parsed into `schema`.

    Structured output rather than free text: the grounded/generated split in the
    response contract IS a schema, and parsing it is what makes faithfulness
    scorable over claims alone.

    A single retry on a validation error, with the error fed back. More than one
    is papering over a bad prompt, and each attempt costs a full call.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in range(retries + 1):
        response = _client().chat.completions.create(
            model=settings().llm_model or "",
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""

        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            if attempt == retries:
                raise
            log.warning("malformed response, retrying: %s", exc)
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That did not match the schema: {exc}. Return only valid JSON.",
                },
            ]

    raise AssertionError("unreachable")
