"""
vectorstore.py — ChromaDB vector store integration for QueryBridge.

Wraps the ChromaDB client behind a clean interface that operates on
chunk dicts produced by ``embedder.embed_chunks``.

Responsibilities:
    - Manage a persistent ChromaDB collection (create / reuse).
    - Upsert chunks with their pre-computed embeddings and metadata.
    - Query by embedding vector and return raw results (text, score,
      source, metadata) for use by the retriever.
    - Expose a delete_collection helper for testing and resets.

Persistence:
    The ChromaDB database is stored at ``<project_root>/chroma_db/``.
    This path is listed in ``.gitignore`` so it is never committed.

Collection naming:
    The default collection name is ``"querybridge"``.  Pass a custom name
    if you need multiple isolated collections (e.g., during tests).
"""

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "querybridge"
_CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_client():
    """Return a persistent ChromaDB client rooted at *_CHROMA_DIR*.

    Returns:
        A ``chromadb.PersistentClient`` instance.

    Raises:
        RuntimeError: If chromadb cannot be imported or the client fails
            to initialise.
    """
    try:
        import chromadb  # type: ignore

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(_CHROMA_DIR))
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Run: pip install chromadb"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise ChromaDB client: {exc}") from exc


def _get_collection(client, collection_name: str):
    """Retrieve or create a ChromaDB collection.

    Args:
        client: A chromadb client instance.
        collection_name: Name of the collection to get or create.

    Returns:
        A chromadb ``Collection`` object.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(
        "Using ChromaDB collection '%s' (%d existing doc(s)).",
        collection_name,
        collection.count(),
    )
    return collection


def _chunk_to_chroma_record(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert a QueryBridge chunk dict into ChromaDB record components.

    Args:
        chunk: A chunk dict with at least ``text``, ``source``, and
            ``embedding`` keys.  Optional keys: ``metadata``, ``chunk_index``.

    Returns:
        Dict with keys ``id``, ``document``, ``embedding``, ``metadata``.

    Raises:
        KeyError: If required keys are missing from *chunk*.
    """
    for required in ("text", "source", "embedding"):
        if required not in chunk:
            raise KeyError(f"Chunk is missing required key: '{required}'")

    # Build a stable ID: source path + chunk index (or a UUID fallback)
    chunk_index = chunk.get("chunk_index", None)
    if chunk_index is not None:
        doc_id = f"{chunk['source']}::chunk::{chunk_index}"
    else:
        doc_id = str(uuid.uuid4())

    # Flatten metadata so ChromaDB can store it (all values must be scalar)
    raw_meta = chunk.get("metadata", {}) or {}
    flat_meta: dict[str, Any] = {
        "source": chunk["source"],
    }
    for key, value in raw_meta.items():
        if isinstance(value, (str, int, float, bool)):
            flat_meta[key] = value
        else:
            flat_meta[key] = str(value)

    if chunk_index is not None:
        flat_meta["chunk_index"] = chunk_index

    return {
        "id": doc_id,
        "document": chunk["text"],
        "embedding": chunk["embedding"],
        "metadata": flat_meta,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upsert(
    chunks: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Insert or update a list of embedded chunks in ChromaDB.

    Chunks that already exist (matched by ID) are updated in place;
    new chunks are inserted.  Embeddings must have been attached to each
    chunk before calling this function (see ``embedder.embed_chunks``).

    Args:
        chunks: List of chunk dicts, each containing ``text``, ``source``,
            ``embedding``, and optionally ``metadata`` / ``chunk_index``.
        collection_name: Target ChromaDB collection.  Defaults to
            ``"querybridge"``.

    Returns:
        Number of chunks successfully upserted.

    Raises:
        TypeError: If *chunks* is not a list or any element is invalid.
        KeyError: If a chunk is missing a required field.
        RuntimeError: If the ChromaDB operation fails.
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"Expected a list of chunk dicts, got {type(chunks).__name__}."
        )
    if not chunks:
        logger.warning("upsert() called with an empty chunk list — nothing to do.")
        return 0

    client = _get_client()
    collection = _get_collection(client, collection_name)

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        try:
            record = _chunk_to_chroma_record(chunk)
        except KeyError as exc:
            raise KeyError(f"Chunk at index {i} is invalid: {exc}") from exc

        ids.append(record["id"])
        documents.append(record["document"])
        embeddings.append(record["embedding"])
        metadatas.append(record["metadata"])

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB upsert failed: {exc}") from exc

    logger.info(
        "Upserted %d chunk(s) into collection '%s'.", len(chunks), collection_name
    )
    return len(chunks)


def query(
    text_embedding: list[float],
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Query the vector store by a pre-computed embedding vector.

    Args:
        text_embedding: The query embedding as a ``list[float]``.
        top_k: Number of nearest neighbours to return.  Defaults to 5.
        collection_name: ChromaDB collection to search.

    Returns:
        List of result dicts (up to *top_k*), each containing:
            - ``text`` (str): The chunk text.
            - ``score`` (float): Cosine similarity in [0, 1].
            - ``source`` (str): Origin file path.
            - ``metadata`` (dict): All stored metadata fields.

    Raises:
        ValueError: If *top_k* is less than 1.
        RuntimeError: If the ChromaDB query fails.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    client = _get_client()
    collection = _get_collection(client, collection_name)

    if collection.count() == 0:
        logger.warning(
            "Collection '%s' is empty — returning no results.", collection_name
        )
        return []

    try:
        raw = collection.query(
            query_embeddings=[text_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

    results: list[dict[str, Any]] = []

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        similarity = max(0.0, 1.0 - (dist / 2.0))
        results.append(
            {
                "text": doc,
                "score": round(similarity, 6),
                "source": meta.get("source", ""),
                "metadata": meta,
            }
        )

    logger.info(
        "Query returned %d result(s) from collection '%s'.",
        len(results),
        collection_name,
    )
    return results


def delete_collection(collection_name: str = DEFAULT_COLLECTION) -> None:
    """Delete a ChromaDB collection and all its data.

    Intended for use in tests and full resets.  This operation is
    **irreversible** — all stored chunks and embeddings will be lost.

    Args:
        collection_name: Name of the collection to delete.

    Raises:
        RuntimeError: If the deletion fails (e.g., collection does not exist).
    """
    client = _get_client()
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted ChromaDB collection '%s'.", collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to delete collection '{collection_name}': {exc}"
        ) from exc


def collection_count(collection_name: str = DEFAULT_COLLECTION) -> int:
    """Return the number of documents stored in a collection.

    Args:
        collection_name: Name of the collection to inspect.

    Returns:
        Integer count of stored documents.
    """
    client = _get_client()
    collection = _get_collection(client, collection_name)
    count = collection.count()
    logger.debug("Collection '%s' contains %d document(s).", collection_name, count)
    return count
