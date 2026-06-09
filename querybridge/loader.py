"""
loader.py — Document loading module for QueryBridge.

Wraps LangChain document loaders (PDF, plain text, Markdown) behind a
uniform interface. All loaders return a list of document dicts:

    [{"text": str, "source": str, "metadata": dict}, ...]

Supported formats: .pdf, .txt, .md
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _langchain_docs_to_dicts(docs: list, source: str) -> list[dict[str, Any]]:
    """Convert a list of LangChain Document objects to QueryBridge dicts."""
    result = []
    for doc in docs:
        result.append(
            {
                "text": doc.page_content,
                "source": source,
                "metadata": doc.metadata,
            }
        )
    return result


def _load_pdf(file_path: Path) -> list[dict[str, Any]]:
    """Load a PDF file using LangChain PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore

    logger.info("Loading PDF: %s", file_path)
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    return _langchain_docs_to_dicts(docs, str(file_path))


def _load_text(file_path: Path) -> list[dict[str, Any]]:
    """Load a plain-text file using LangChain TextLoader."""
    from langchain_community.document_loaders import TextLoader  # type: ignore

    logger.info("Loading text file: %s", file_path)
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    return _langchain_docs_to_dicts(docs, str(file_path))


def _load_markdown(file_path: Path) -> list[dict[str, Any]]:
    """Load a Markdown file using LangChain UnstructuredMarkdownLoader."""
    from langchain_community.document_loaders import UnstructuredMarkdownLoader  # type: ignore

    logger.info("Loading Markdown: %s", file_path)
    loader = UnstructuredMarkdownLoader(str(file_path))
    docs = loader.load()
    return _langchain_docs_to_dicts(docs, str(file_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_markdown,
}


def load_document(file_path: str) -> list[dict[str, Any]]:
    """Load a single document from *file_path* and return normalised dicts.

    Args:
        file_path: Absolute or relative path to a .pdf, .txt, or .md file.

    Returns:
        List of dicts with keys ``text``, ``source``, and ``metadata``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file extension is not supported.
        UnicodeDecodeError: If a text file cannot be decoded as UTF-8.
        RuntimeError: If the underlying LangChain loader fails.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()
    loader_fn = _LOADER_MAP.get(suffix)

    if loader_fn is None:
        supported = ", ".join(_LOADER_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {supported}"
        )

    try:
        documents = loader_fn(path)
    except UnicodeDecodeError as exc:
        logger.error("Encoding error reading %s: %s", file_path, exc)
        raise
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        raise RuntimeError(f"Loader failed for '{file_path}': {exc}") from exc

    logger.info("Loaded %d page(s) from %s", len(documents), file_path)
    return documents


def load_directory(dir_path: str) -> list[dict[str, Any]]:
    """Recursively load all supported documents from *dir_path*.

    Skips files with unsupported extensions and logs a warning for each.

    Args:
        dir_path: Path to a directory containing .pdf, .txt, or .md files.

    Returns:
        Combined list of document dicts from all loaded files.

    Raises:
        FileNotFoundError: If *dir_path* does not exist.
        NotADirectoryError: If *dir_path* is not a directory.
    """
    path = Path(dir_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_documents: list[dict[str, Any]] = []
    skipped = 0

    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        if file.suffix.lower() not in _LOADER_MAP:
            logger.debug("Skipping unsupported file: %s", file)
            skipped += 1
            continue
        docs = load_document(str(file))
        all_documents.extend(docs)

    logger.info(
        "Directory load complete: %d documents, %d file(s) skipped",
        len(all_documents),
        skipped,
    )
    return all_documents
