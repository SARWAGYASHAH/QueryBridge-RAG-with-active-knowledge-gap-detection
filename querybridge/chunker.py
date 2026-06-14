"""
chunker.py — Text chunking module for QueryBridge.

Wraps LangChain's RecursiveCharacterTextSplitter to split document dicts
(produced by loader.py) into smaller, overlapping chunks while preserving
sentence boundaries.

Each output chunk is a dict:

    {
        "text":       str,   # chunk text content
        "source":     str,   # original file path
        "chunk_index": int,  # position of this chunk within the source doc
        "metadata":   dict,  # merged original metadata + chunk-level fields
    }

Default split parameters:
    - chunk_size:    512 tokens  (approximated as characters for splitter)
    - chunk_overlap:  50 tokens

Duplicate-content guard:
    After splitting, a deduplication pass removes any chunk whose full text
    is contained within the text of an immediately adjacent chunk.  This
    prevents the overlap window from producing near-identical chunks when the
    overlap-to-size ratio is high.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Separators tried in order — respects paragraphs → sentences → words
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_splitter(chunk_size: int, chunk_overlap: int):
    """Instantiate a RecursiveCharacterTextSplitter with QueryBridge defaults.

    Args:
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def _deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove chunks whose text is fully contained in an adjacent chunk.

    When the overlap window is proportionally large, RecursiveCharacterText-
    Splitter can emit consecutive chunks that are near-identical because the
    overlap region spans almost the entire chunk body.  This pass drops any
    chunk whose stripped text is a substring of either its predecessor or its
    successor, keeping only the longer, more informative version.

    Chunk indices are reassigned after deduplication so they remain
    contiguous.

    Args:
        chunks: Ordered list of chunk dicts (all from the same source doc).

    Returns:
        Deduplicated list with refreshed ``chunk_index`` values in both the
        top-level key and the nested ``metadata`` dict.
    """
    if len(chunks) <= 1:
        return chunks

    kept: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip()

        # Check against immediate predecessor
        if i > 0:
            prev_text = chunks[i - 1]["text"].strip()
            if text in prev_text:
                logger.debug(
                    "Dropped duplicate chunk at index %d (substring of predecessor)",
                    chunk["chunk_index"],
                )
                continue

        # Check against immediate successor
        if i < len(chunks) - 1:
            next_text = chunks[i + 1]["text"].strip()
            if text in next_text:
                logger.debug(
                    "Dropped duplicate chunk at index %d (substring of successor)",
                    chunk["chunk_index"],
                )
                continue

        kept.append(chunk)

    # Reassign chunk indices to keep them contiguous
    for new_idx, chunk in enumerate(kept):
        chunk["chunk_index"] = new_idx
        chunk["metadata"]["chunk_index"] = new_idx

    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info("Deduplication removed %d duplicate chunk(s).", dropped)

    return kept


def _chunk_single_doc(
    doc: dict[str, Any],
    splitter,
    base_index: int,
) -> list[dict[str, Any]]:
    """Split one document dict into chunk dicts.

    Args:
        doc: A document dict with keys ``text``, ``source``, ``metadata``.
        splitter: A configured RecursiveCharacterTextSplitter instance.
        base_index: Starting chunk_index value for this document's chunks.

    Returns:
        List of chunk dicts with injected ``chunk_index`` and ``metadata``.
    """
    raw_chunks = splitter.split_text(doc["text"])
    chunks: list[dict[str, Any]] = []

    for i, text in enumerate(raw_chunks):
        chunk_meta = {
            **doc.get("metadata", {}),
            "chunk_index": base_index + i,
            "source": doc["source"],
        }
        chunks.append(
            {
                "text": text,
                "source": doc["source"],
                "chunk_index": base_index + i,
                "metadata": chunk_meta,
            }
        )

    # Remove any chunks whose content is duplicated by an adjacent chunk
    chunks = _deduplicate_chunks(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split a list of document dicts into overlapping text chunks.

    Processes each document sequentially and assigns a monotonically
    increasing ``chunk_index`` across the entire collection so that every
    chunk has a globally unique position identifier.

    Args:
        documents: List of dicts as returned by ``loader.load_document`` or
            ``loader.load_directory``.  Each dict must contain the keys
            ``text``, ``source``, and ``metadata``.
        chunk_size: Maximum number of characters per chunk.  Defaults to
            ``DEFAULT_CHUNK_SIZE`` (512).
        chunk_overlap: Character overlap between consecutive chunks.
            Defaults to ``DEFAULT_CHUNK_OVERLAP`` (50).

    Returns:
        Flat list of chunk dicts ordered by source document then position.

    Raises:
        ValueError: If *chunk_overlap* is greater than or equal to
            *chunk_size*, which would produce degenerate chunks.
        TypeError: If *documents* is not a list, or any element is not a dict
            with the required keys.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    if not isinstance(documents, list):
        raise TypeError(f"Expected a list of documents, got {type(documents).__name__}.")

    splitter = _build_splitter(chunk_size, chunk_overlap)

    all_chunks: list[dict[str, Any]] = []
    global_index = 0

    for doc_num, doc in enumerate(documents):
        if not isinstance(doc, dict) or "text" not in doc:
            raise TypeError(
                f"Document at index {doc_num} must be a dict with a 'text' key."
            )

        doc_chunks = _chunk_single_doc(doc, splitter, global_index)
        logger.debug(
            "Document '%s' → %d chunk(s)", doc.get("source", "unknown"), len(doc_chunks)
        )
        all_chunks.extend(doc_chunks)
        global_index += len(doc_chunks)

    logger.info(
        "Chunking complete: %d document(s) → %d chunk(s) "
        "(size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
