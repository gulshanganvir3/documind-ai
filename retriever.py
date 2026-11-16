"""Step 5: semantic search over the FAISS index with relevance filtering."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config


def retrieve(
    store: FAISS,
    query: str,
    sources: list[str] | None = None,
    k: int = config.TOP_K,
    max_distance: float = config.MAX_DISTANCE,
) -> list[tuple[Document, float]]:
    """Return up a k (chunk, distance) pairs, best first, dropping irrelevant ones."""
    metadata_filter = None
    if sources:
        allowed = set(sources)
        metadata_filter = lambda meta: meta.get("source") in allowed
    results = store.similarity_search_with_score(
        query,
        k=k,
        filter = metadata_filter,
        fetch_k=max(config.FETCH_K, k),
    )

    return [(doc, score) for doc, score in results if score <= max_distance]
