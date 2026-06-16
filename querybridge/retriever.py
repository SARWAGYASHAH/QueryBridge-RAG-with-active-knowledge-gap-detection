"""
retriever.py — Retrieval module for QueryBridge.

Provides a ``Retriever`` class that, given a natural-language query,
embeds it via the embedding module, searches the vector store, and
returns the top-k most relevant chunks ranked by cosine similarity.

Key features:
    - Cosine similarity scoring (converted from ChromaDB distances).
    - Duplicate-chunk filtering: chunks with identical text content are
      collapsed so the results never contain the same passage twice, even
      if it was ingested from overlapping sources.
    - Configurable ``top_k`` (default 5).
    - Accepts a ``VectorStore`` instance for dependency injection,
      making the retriever fully testable with mock stores.

A module-level ``retrieve()`` convenience function is provided for
backward compatibility and simple scripts.

Returned format::

    [
        {"text": str, "score": float, "source": str, "metadata": dict},
        ...
    ]
"""

import logging
from typing import Any

from querybridge import embedder
from querybridge.vectorstore import VectorStore, DEFAULT_COLLECTION

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
# Retriever class
# ---------------------------------------------------------------------------


class Retriever:
    """Retrieves the most relevant chunks for a natural-language query.

    Wraps the embedding step and vector-store query behind a single
    ``retrieve()`` call.  Accepts a ``VectorStore`` instance at init
    time for dependency injection, making it straightforward to swap
    stores in tests.

    Args:
        store: A ``VectorStore`` instance to query against.  If *None*,
            a default store using the ``"querybridge"`` collection is
            created on first use.
        top_k: Default number of results to return.  Can be overridden
            per query.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._store = store
        self._default_top_k = top_k

    # -- lazy store --------------------------------------------------------

    @property
    def store(self) -> VectorStore:
        """Return the vector store, creating a default one if needed."""
        if self._store is None:
            self._store = VectorStore()
        return self._store

    # -- public methods ----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for *query*.

        Embeds *query* using the sentence-transformer model, searches the
        vector store, deduplicates results, and returns up to *top_k*
        chunks sorted by cosine similarity (highest first).

        Args:
            query: The user's natural-language question or search string.
            top_k: Maximum number of results to return.  Defaults to the
                value passed at init (or 5).

        Returns:
            List of result dicts (up to *top_k*), each containing:
                - ``text`` (str): The chunk text.
                - ``score`` (float): Cosine similarity in [0, 1].
                - ``source`` (str): Origin file path.
                - ``metadata`` (dict): All stored metadata fields.

            Returns an empty list when the collection is empty or the
            query cannot be embedded.

        Raises:
            TypeError: If *query* is not a string.
            ValueError: If *top_k* is less than 1.
            RuntimeError: If embedding or vector search fails.
        """
        k = top_k if top_k is not None else self._default_top_k

        if not isinstance(query, str):
            raise TypeError(f"Expected a string query, got {type(query).__name__}.")
        if k < 1:
            raise ValueError(f"top_k must be at least 1, got {k}.")

        logger.info("Retrieving top-%d chunks for query: '%s'", k, query[:80])

        # Embed the query
        try:
            query_vectors = embedder.embed_texts([query])
            query_embedding = query_vectors[0]
        except RuntimeError as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

        # Fetch extra candidates so that after deduplication we still
        # have at least top_k results (when possible).
        fetch_k = min(k * 2, k + 10)

        raw_results = self.store.query(
            text_embedding=query_embedding,
            top_k=fetch_k,
        )

        if not raw_results:
            logger.warning("No results found for query: '%s'", query[:80])
            return []

        # Remove duplicate chunks (same text from overlapping sources)
        unique_results = _deduplicate_results(raw_results)

        # Trim to requested top_k
        final_results = unique_results[:k]

        logger.info(
            "Retrieval complete: %d result(s) returned (fetched %d, %d unique).",
            len(final_results),
            len(raw_results),
            len(unique_results),
        )
        return final_results


# ---------------------------------------------------------------------------
# Module-level convenience function (backward compatibility)
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a natural-language query.

    This is a module-level convenience wrapper that creates a
    ``Retriever`` backed by a ``VectorStore`` for the given collection.

    See :meth:`Retriever.retrieve` for full documentation.
    """
    store = VectorStore(collection_name=collection_name)
    retriever = Retriever(store=store, top_k=top_k)
    return retriever.retrieve(query, top_k=top_k)
