"""
retriever.py — Retrieval module for QueryBridge.

Given a natural-language query, embeds it via ``embedder.embed_texts``,
searches the ChromaDB vector store, and returns the top-k most relevant
chunks ranked by cosine similarity.

Key features:
    - Cosine similarity scoring (converted from ChromaDB distances).
    - Duplicate-chunk filtering: chunks with identical text content are
      collapsed so the results never contain the same passage twice, even
      if it was ingested from overlapping sources.
    - Configurable ``top_k`` (default 5).

Returned format::

    [
        {"text": str, "score": float, "source": str, "metadata": dict},
        ...
    ]
"""

import logging
from typing import Any

from querybridge import embedder, vectorstore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deduplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove results whose text content has already been seen.

    When the same passage is ingested from multiple files, or when
    overlapping chunks produce near-identical text, the vector store can
    return duplicates.  This function keeps only the first (highest-
    scoring) occurrence of each unique text.

    Args:
        results: List of result dicts, assumed to be sorted by score
            descending.

    Returns:
        Deduplicated list preserving the original order.
    """
    seen_texts: set[str] = set()
    unique: list[dict[str, Any]] = []

    for result in results:
        normalised = result["text"].strip()
        if normalised in seen_texts:
            logger.debug(
                "Dropped duplicate chunk from source '%s'.",
                result.get("source", "unknown"),
            )
            continue
        seen_texts.add(normalised)
        unique.append(result)

    dropped = len(results) - len(unique)
    if dropped:
        logger.info("Deduplication removed %d duplicate result(s).", dropped)

    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = vectorstore.DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a natural-language query.

    Embeds *query* using the sentence-transformer model, searches the
    ChromaDB collection, deduplicates results, and returns up to
    *top_k* chunks sorted by cosine similarity (highest first).

    Args:
        query: The user's natural-language question or search string.
        top_k: Maximum number of results to return.  Defaults to 5.
            Extra candidates are fetched internally to compensate for
            duplicates that will be removed.
        collection_name: ChromaDB collection to search.  Defaults to
            ``"querybridge"``.

    Returns:
        List of result dicts (up to *top_k*), each containing:
            - ``text`` (str): The chunk text.
            - ``score`` (float): Cosine similarity in [0, 1].
            - ``source`` (str): Origin file path.
            - ``metadata`` (dict): All stored metadata fields.

        Returns an empty list when the collection is empty or the query
        cannot be embedded.

    Raises:
        TypeError: If *query* is not a string.
        ValueError: If *top_k* is less than 1.
        RuntimeError: If embedding or vector search fails.
    """
    if not isinstance(query, str):
        raise TypeError(f"Expected a string query, got {type(query).__name__}.")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    logger.info("Retrieving top-%d chunks for query: '%s'", top_k, query[:80])

    # Embed the query
    try:
        query_vectors = embedder.embed_texts([query])
        query_embedding = query_vectors[0]
    except RuntimeError as exc:
        logger.error("Failed to embed query: %s", exc)
        raise

    # Fetch extra candidates so that after deduplication we still have
    # at least top_k results (when possible).
    fetch_k = min(top_k * 2, top_k + 10)

    raw_results = vectorstore.query(
        text_embedding=query_embedding,
        top_k=fetch_k,
        collection_name=collection_name,
    )

    if not raw_results:
        logger.warning("No results found for query: '%s'", query[:80])
        return []

    # Remove duplicate chunks (same text from overlapping sources)
    unique_results = _deduplicate_results(raw_results)

    # Trim to requested top_k
    final_results = unique_results[:top_k]

    logger.info(
        "Retrieval complete: %d result(s) returned (fetched %d, %d unique).",
        len(final_results),
        len(raw_results),
        len(unique_results),
    )
    return final_results
