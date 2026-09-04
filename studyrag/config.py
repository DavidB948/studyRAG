"""Central config. Everything tunable lives here, not scattered as literals.

The split with `.env` is deliberate: `.env` holds secrets and values that differ
per machine (connection strings, API keys, endpoints). Everything else is a design
decision that must move together with the code and the schema, so it lives here
under version control. A model name in `.env` looks harmless and is not: it can
drift out of step with `embed_dim` and the `vector(N)` column with nothing to
catch it.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- from .env: secrets and per-machine values ---
    database_url: str
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    # The judge is configured separately from the generator, provider included: it runs
    # ~20x more calls (ragas decomposes each answer into statements and verifies each),
    # so it hits different quota limits and is often worth pointing at another vendor.
    # Keeping them separate also means a judge upgrade never silently changes what the
    # system under test produces. Falls back to the llm_* values when unset.
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_model: str | None = None

    # --- design decisions: change these here, not in the environment ---

    # Ingest root. `course` is the first path segment under this, `doc_path` the rest.
    corpus_root: Path = Path("data/raw")

    # 512-token window and 768 dims. Chosen over all-MiniLM-L6-v2 because prose chunks
    # target 400-800 tokens and MiniLM silently truncates at 256.
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768

    # BGE is trained with an asymmetric instru gction: the QUERY gets this prefix, the
    # stored passage does not. Prefixing both, or neither, measurably costs recall.
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Cross-encoder for reranking and for claim entailment. Same size class as the
    # embedder (~110M params, ~420 MB), runs locally, no API cost.
    rerank_model: str = "BAAI/bge-reranker-base"

    # Target size for prose chunks, under the model's 512-token window.
    prose_target_tokens: int = 500
    prose_overlap_tokens: int = 50


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Config, built on first use rather than at import.

    Instantiating at module scope made importing ANY module that transitively
    reaches config require a populated .env — so `allocate()`, a pure function over
    integers, could not be imported without a database URL. CI caught it: the pure
    unit tests failed on a missing `database_url` they never touch.

    Cached, so the .env is still read once per process and a bad value still fails
    loudly, just at the point of use instead of the point of import.
    """
    return Settings()  # type: ignore[call-arg]
