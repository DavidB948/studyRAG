"""The embedding model, loaded once.

Isolated in its own module so the chunker never imports torch: `count_tokens` is
passed into chunking as a plain callable, which keeps chunking pure and fast to
run in isolation.
"""

from __future__ import annotations

from functools import lru_cache

from studyrag.config import settings


@lru_cache(maxsize=1)
def _model():
    """Load the model on first use. ~440 MB, so never at import time.

    The dimension is checked here rather than left to Postgres. A mismatch would
    otherwise surface as an opaque insert error deep in ingest, or — worse — not
    at all, if the wrong model happened to share the column's dimension.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    # Renamed in sentence-transformers v5; the old name still works but warns.
    dimension_of = getattr(model, "get_embedding_dimension", None) or (
        model.get_embedding_dimension
    )
    actual = dimension_of()
    if actual != settings.embed_dim:
        raise RuntimeError(
            f"{settings.embed_model} produces {actual}-dim vectors but embed_dim is "
            f"{settings.embed_dim}. The schema's vector(N) must match. Check .env: it "
            f"overrides the default in config.py."
        )
    return model


def count_tokens(text: str) -> int:
    """Tokens as the embedding model counts them, not words.

    This is the number that matters: anything past the model's window is
    silently dropped from the vector, so `token_count` has to mean what the
    model actually read.
    """
    return len(_model().tokenizer.encode(text, add_special_tokens=False))


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed stored text. No instruction prefix — see embed_query."""
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a search query, with the instruction prefix BGE expects.

    BGE is trained asymmetrically: the query carries an instruction, the stored
    passage does not. Prefixing both, or neither, measurably costs recall.
    """
    vector = _model().encode(
        settings.query_prefix + text, normalize_embeddings=True
    )
    return vector.tolist()
