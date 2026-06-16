"""
vectorstore.py — ChromaDB vector store integration for QueryBridge.

Provides a ``VectorStore`` class that wraps ChromaDB behind a clean,
testable interface.  The class handles collection lifecycle, upserting
embedded chunks, querying by embedding vector, and deleting collections.

Module-level convenience functions (``upsert``, ``query``,
``delete_collection``, ``collection_count``) delegate to a shared
default ``VectorStore`` instance so that existing call-sites continue
to work unchanged.

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


def _build_chroma_client(persist_dir: Path):
    """Return a persistent ChromaDB client rooted at *persist_dir*.

    Args:
        persist_dir: Filesystem path for ChromaDB persistence.

    Returns:
        A ``chromadb.PersistentClient`` instance.

    Raises:
        RuntimeError: If chromadb cannot be imported or the client fails
            to initialise.
    """
    try:
        import chromadb  # type: ignore

        persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_dir))
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Run: pip install chromadb"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise ChromaDB client: {exc}") from exc


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
# VectorStore class
# ---------------------------------------------------------------------------


class VectorStore:
    """Manages a ChromaDB collection for storing and querying embeddings.

    Encapsulates client creation, collection lifecycle, and all CRUD
    operations behind a single object.  Accepts ``persist_dir`` and
    ``collection_name`` at init time so that tests can create fully
    isolated instances.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Filesystem directory for ChromaDB persistence.
            Defaults to ``<project_root>/chroma_db/``.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: Path | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir or _CHROMA_DIR
        self._client = _build_chroma_client(self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "VectorStore initialised: collection='%s', docs=%d.",
            self._collection_name,
            self._collection.count(),
        )

    # -- properties --------------------------------------------------------

    @property
    def collection_name(self) -> str:
        """Return the name of the underlying ChromaDB collection."""
        return self._collection_name

    # -- public methods ----------------------------------------------------

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or update a list of embedded chunks.

        Chunks that already exist (matched by ID) are updated in place;
        new chunks are inserted.  Embeddings must have been attached to
        each chunk before calling this method.

        Args:
            chunks: List of chunk dicts, each containing ``text``,
                ``source``, ``embedding``, and optionally ``metadata`` /
                ``chunk_index``.

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
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB upsert failed: {exc}") from exc

        logger.info(
            "Upserted %d chunk(s) into collection '%s'.",
            len(chunks),
            self._collection_name,
        )
        return len(chunks)

    def query(
        self,
        text_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the vector store by a pre-computed embedding vector.

        Args:
            text_embedding: The query embedding as a ``list[float]``.
            top_k: Number of nearest neighbours to return.  Defaults to 5.

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

        if self._collection.count() == 0:
            logger.warning(
                "Collection '%s' is empty — returning no results.",
                self._collection_name,
            )
            return []

        try:
            raw = self._collection.query(
                query_embeddings=[text_embedding],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

        results: list[dict[str, Any]] = []

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance is in [0, 2]; convert to similarity
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
            self._collection_name,
        )
        return results

    def delete_collection(self) -> None:
        """Delete the ChromaDB collection and all its data.

        This operation is **irreversible** — all stored chunks and
        embeddings will be lost.

        Raises:
            RuntimeError: If the deletion fails.
        """
        try:
            self._client.delete_collection(name=self._collection_name)
            logger.info("Deleted ChromaDB collection '%s'.", self._collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete collection '{self._collection_name}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of documents stored in the collection.

        Returns:
            Integer count of stored documents.
        """
        n = self._collection.count()
        logger.debug(
            "Collection '%s' contains %d document(s).",
            self._collection_name,
            n,
        )
        return n


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward compatibility)
# ---------------------------------------------------------------------------

_stores: dict[str, VectorStore] = {}


def _get_store(collection_name: str) -> VectorStore:
    """Return a cached ``VectorStore`` for *collection_name*.

    Reuses existing instances to avoid re-creating the ChromaDB client
    on every function call.

    Args:
        collection_name: Name of the ChromaDB collection.

    Returns:
        A ``VectorStore`` instance for the given collection.
    """
    if collection_name not in _stores:
        _stores[collection_name] = VectorStore(collection_name=collection_name)
    return _stores[collection_name]


def upsert(
    chunks: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Insert or update chunks via the module-level convenience API.

    See :meth:`VectorStore.upsert` for full documentation.
    """
    return _get_store(collection_name).upsert(chunks)


def query(
    text_embedding: list[float],
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Query the vector store via the module-level convenience API.

    See :meth:`VectorStore.query` for full documentation.
    """
    return _get_store(collection_name).query(text_embedding, top_k=top_k)


def delete_collection(collection_name: str = DEFAULT_COLLECTION) -> None:
    """Delete a collection via the module-level convenience API.

    See :meth:`VectorStore.delete_collection` for full documentation.
    """
    _get_store(collection_name).delete_collection()
    _stores.pop(collection_name, None)


def collection_count(collection_name: str = DEFAULT_COLLECTION) -> int:
    """Return the document count via the module-level convenience API.

    See :meth:`VectorStore.count` for full documentation.
    """
    return _get_store(collection_name).count()
