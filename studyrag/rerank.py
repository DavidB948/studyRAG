"""Cross-encoder reranking.

The bi-encoder in embed.py compresses each passage into one vector at ingest,
before any query exists — so it cannot know which part of a passage matters for a
given question. A cross-encoder reads query and passage together and outputs a
relevance score directly, which is more accurate and far too slow to search with.

So they compose rather than compete: the bi-encoder retrieves wide and
approximately, this narrows precisely. Nothing here is precomputable — the input
is the pair, and the query does not exist until query time.
"""

from __future__ import annotations

from functools import lru_cache

from studyrag.config import settings


@lru_cache(maxsize=1)
def _model():
    """Load on first use. ~420 MB, same size class as the embedder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def score(query: str, texts: list[str]) -> list[float]:
    """Relevance of each text to the query, as a calibrated 0-1 probability.

    No instruction prefix here, unlike embed_query: the asymmetry BGE's bi-encoder
    needs comes from the prefix, but a cross-encoder sees both texts at once and
    was trained on raw query/passage pairs.
    """
    if not texts:
        return []
    # `predict` already applies sigmoid for this model's single-label head. Applying
    # it a second time squashes every score to ~0.5 and destroys the ranking, which
    # is silent: the numbers still look like probabilities.
    return [float(s) for s in _model().predict([(query, t) for t in texts])]
