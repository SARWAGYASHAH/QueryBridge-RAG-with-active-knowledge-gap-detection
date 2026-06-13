"""
embedder.py — Embedding module for QueryBridge.

Uses the ``sentence-transformers/all-MiniLM-L6-v2`` model to convert text
into dense vector representations.  Designed to work directly on chunk dicts
produced by ``chunker.py``.

Key features:
    - Batch embedding support to minimise GPU/CPU round-trips.
    - Local disk cache (``data/embeddings_cache/``) so that identical texts
      are never re-encoded across runs.
    - Returns plain Python lists (not NumPy arrays) so outputs are directly
      JSON-serialisable and ChromaDB-compatible.

Cache format:
    A single ``cache.json`` file maps SHA-256(text) → embedding list.
    The cache is loaded once at module import and flushed to disk after
    every ``embed_chunks`` call that produced new embeddings.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
_CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings_cache"
_CACHE_FILE = _CACHE_DIR / "cache.json"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache() -> dict[str, list[float]]:
    """Load the embedding cache from disk.

    Returns:
        Dict mapping text hash → embedding list.  Returns an empty dict if
        the cache file does not yet exist.
    """
    if not _CACHE_FILE.exists():
        return {}
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read embedding cache (%s). Starting fresh.", exc)
        return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    """Persist the embedding cache to disk.

    Args:
        cache: The full cache dict to write.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _CACHE_FILE.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        logger.debug("Embedding cache saved (%d entries).", len(cache))
    except OSError as exc:
        logger.error("Failed to save embedding cache: %s", exc)


def _hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text* for use as a cache key.

    Args:
        text: The input string to hash.

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model loader (lazy singleton)
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """Return the singleton SentenceTransformer model, loading it on first call.

    Returns:
        A loaded ``SentenceTransformer`` instance.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading embedding model: %s", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{MODEL_NAME}': {exc}"
            ) from exc
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of raw strings and return their vector representations.

    Texts that are already in the local cache are returned immediately;
    only cache-misses are forwarded to the model.  The cache is updated
    and flushed after encoding new texts.

    Args:
        texts: List of strings to embed.  Empty strings are embedded as-is
            (the model handles them gracefully).
        batch_size: Number of texts to send to the model per forward pass.
            Reduce this if you run into memory issues.  Defaults to
            ``DEFAULT_BATCH_SIZE`` (64).

    Returns:
        List of embedding vectors (one per input text), each a
        ``list[float]`` of length 384 (all-MiniLM-L6-v2 output dim).

    Raises:
        TypeError: If *texts* is not a list of strings.
        RuntimeError: If the underlying model call fails.
    """
    if not isinstance(texts, list):
        raise TypeError(f"Expected a list of strings, got {type(texts).__name__}.")

    cache = _load_cache()
    embeddings: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    # Populate from cache where possible
    for i, text in enumerate(texts):
        key = _hash_text(text)
        if key in cache:
            embeddings[i] = cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    logger.info(
        "Embedding request: %d total, %d cached, %d to encode.",
        len(texts),
        len(texts) - len(uncached_texts),
        len(uncached_texts),
    )

    if uncached_texts:
        model = _get_model()

        # Process in explicit batches to cap peak memory usage.
        # model.encode() builds the full output array in memory; encoding
        # thousands of texts at once can cause OOM on machines with limited
        # RAM.  By slicing uncached_texts into chunks of *batch_size*,
        # converting each batch to Python lists, and deleting the NumPy
        # array immediately, we keep only one batch worth of vectors in
        # memory at a time.
        encoded_count = 0
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]

            try:
                batch_vecs = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Model encoding failed on batch "
                    f"[{batch_start}:{batch_end}]: {exc}"
                ) from exc

            for j, vec in enumerate(batch_vecs):
                global_j = batch_start + j
                idx = uncached_indices[global_j]
                embedding = vec.tolist()
                embeddings[idx] = embedding
                cache[_hash_text(texts[idx])] = embedding

            encoded_count += len(batch_texts)

            # Release the NumPy array so memory is freed before the next
            # batch is allocated.
            del batch_vecs

            logger.debug(
                "Encoded batch %d–%d (%d/%d uncached texts).",
                batch_start,
                batch_end - 1,
                encoded_count,
                len(uncached_texts),
            )

        _save_cache(cache)

    return embeddings  # type: ignore[return-value]


def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Embed a list of chunk dicts and attach the embedding to each.

    Each chunk dict is updated **in-place** with a new ``"embedding"`` key
    containing the vector as a ``list[float]``.  The original dict is also
    returned for convenience.

    Args:
        chunks: List of chunk dicts as produced by ``chunker.chunk_documents``.
            Each dict must have a ``"text"`` key.
        batch_size: Forwarded to :func:`embed_texts`.

    Returns:
        The same list of chunk dicts, each now containing an ``"embedding"``
        key.

    Raises:
        TypeError: If *chunks* is not a list or any element lacks a ``"text"``
            key.
        RuntimeError: If the underlying model call fails.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"Expected a list of chunk dicts, got {type(chunks).__name__}.")

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "text" not in chunk:
            raise TypeError(
                f"Chunk at index {i} must be a dict with a 'text' key."
            )

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts, batch_size=batch_size)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec

    logger.info("Attached embeddings to %d chunk(s).", len(chunks))
    return chunks
