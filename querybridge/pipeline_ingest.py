"""
pipeline_ingest.py — Document upsert pipeline for QueryBridge.

Orchestrates the full ingestion flow from raw files to a searchable
vector store in a single call:

    load_document / load_directory
        → chunk_documents
            → embed_chunks
                → vectorstore.upsert

This module is intentionally thin — each step delegates to its
specialist module.  It exists so that callers (API, CLI, tests) have
one stable entry-point for ingestion without re-wiring the chain.

Typical usage::

    from querybridge.pipeline_ingest import ingest_file, ingest_directory

    result = ingest_file("data/raw/attention.pdf")
    print(result)
    # {'source': '...', 'chunks_ingested': 42, 'collection': 'querybridge'}

    result = ingest_directory("data/raw/")
    print(result)
    # {'sources_processed': 3, 'chunks_ingested': 127, 'collection': 'querybridge'}
"""

import logging
from typing import Any

from querybridge import chunker, embedder, loader, vectorstore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_ingest(
    documents: list[dict[str, Any]],
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_batch_size: int,
) -> int:
    """Run chunk → embed → upsert for a list of document dicts.

    Args:
        documents: Loaded document dicts (output of loader).
        collection_name: Target ChromaDB collection.
        chunk_size: Character limit per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        embedding_batch_size: Number of texts per embedding forward pass.

    Returns:
        Total number of chunks upserted.

    Raises:
        RuntimeError: If any stage fails.
    """
    if not documents:
        logger.warning("No documents to ingest — pipeline_ingest received empty list.")
        return 0

    logger.info("Ingestion pipeline started: %d raw document(s).", len(documents))

    # Stage 1 — chunk
    chunks = chunker.chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Stage 1 complete: %d chunk(s) produced.", len(chunks))

    # Stage 2 — embed
    chunks = embedder.embed_chunks(chunks, batch_size=embedding_batch_size)
    logger.info("Stage 2 complete: embeddings attached to %d chunk(s).", len(chunks))

    # Stage 3 — upsert
    n_upserted = vectorstore.upsert(chunks, collection_name=collection_name)
    logger.info("Stage 3 complete: %d chunk(s) upserted to '%s'.", n_upserted, collection_name)

    return n_upserted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_file(
    file_path: str,
    collection_name: str = vectorstore.DEFAULT_COLLECTION,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = chunker.DEFAULT_CHUNK_OVERLAP,
    embedding_batch_size: int = embedder.DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Ingest a single document file into the vector store.

    Loads, chunks, embeds, and upserts the document in one call.
    Supported formats: ``.pdf``, ``.txt``, ``.md``.

    Args:
        file_path: Path to the document file.
        collection_name: Target ChromaDB collection.  Defaults to
            ``"querybridge"``.
        chunk_size: Max characters per chunk.  Defaults to 512.
        chunk_overlap: Overlap characters between chunks.  Defaults to 50.
        embedding_batch_size: Texts per embedding batch.  Defaults to 64.

    Returns:
        Dict with keys:
            - ``source`` (str): The file path that was ingested.
            - ``chunks_ingested`` (int): Number of chunks stored.
            - ``collection`` (str): Target collection name.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file extension is unsupported.
        RuntimeError: If any stage of the pipeline fails.
    """
    logger.info("Ingesting file: %s", file_path)
    documents = loader.load_document(file_path)

    n_chunks = _run_ingest(
        documents=documents,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=embedding_batch_size,
    )

    result: dict[str, Any] = {
        "source": file_path,
        "chunks_ingested": n_chunks,
        "collection": collection_name,
    }
    logger.info("File ingestion complete: %s", result)
    return result


def ingest_directory(
    dir_path: str,
    collection_name: str = vectorstore.DEFAULT_COLLECTION,
    chunk_size: int = chunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = chunker.DEFAULT_CHUNK_OVERLAP,
    embedding_batch_size: int = embedder.DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Ingest all supported documents from a directory into the vector store.

    Recursively discovers ``.pdf``, ``.txt``, and ``.md`` files, then runs
    the full load → chunk → embed → upsert pipeline on the combined set.

    Args:
        dir_path: Path to the directory containing documents.
        collection_name: Target ChromaDB collection.  Defaults to
            ``"querybridge"``.
        chunk_size: Max characters per chunk.  Defaults to 512.
        chunk_overlap: Overlap characters between chunks.  Defaults to 50.
        embedding_batch_size: Texts per embedding batch.  Defaults to 64.

    Returns:
        Dict with keys:
            - ``directory`` (str): The directory that was processed.
            - ``sources_processed`` (int): Number of distinct files loaded.
            - ``chunks_ingested`` (int): Total chunks stored.
            - ``collection`` (str): Target collection name.

    Raises:
        FileNotFoundError: If *dir_path* does not exist.
        NotADirectoryError: If *dir_path* is not a directory.
        RuntimeError: If any stage of the pipeline fails.
    """
    logger.info("Ingesting directory: %s", dir_path)
    documents = loader.load_directory(dir_path)

    # Track unique source files for reporting
    sources = {doc["source"] for doc in documents}

    n_chunks = _run_ingest(
        documents=documents,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=embedding_batch_size,
    )

    result: dict[str, Any] = {
        "directory": dir_path,
        "sources_processed": len(sources),
        "chunks_ingested": n_chunks,
        "collection": collection_name,
    }
    logger.info("Directory ingestion complete: %s", result)
    return result
