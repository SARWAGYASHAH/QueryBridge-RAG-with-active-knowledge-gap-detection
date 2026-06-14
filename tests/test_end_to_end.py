"""
test_end_to_end.py — Basic end-to-end query test for QueryBridge.

Exercises the full ingestion-to-retrieval pipeline:

    load_document → chunk_documents → embed_chunks → vectorstore.upsert
        → retriever.retrieve

Uses a temporary text file with known content and an isolated ChromaDB
collection to keep tests deterministic and side-effect-free.
"""

import logging
import os
import tempfile

import pytest

from querybridge import loader
from querybridge.chunker import chunk_documents
from querybridge.embedder import embed_chunks
from querybridge import vectorstore
from querybridge.retriever import retrieve

logger = logging.getLogger(__name__)

# Isolated collection name so tests never touch production data.
_TEST_COLLECTION = "querybridge_test_e2e"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Known document content with distinct topics for targeted retrieval.
_DOC_CONTENT = """\
The Transformer architecture was introduced in the paper "Attention Is All
You Need" by Vaswani et al. in 2017. It replaced recurrence and convolutions
entirely with self-attention mechanisms for sequence-to-sequence modelling.

The key innovation is the multi-head attention mechanism, which allows the
model to jointly attend to information from different representation
subspaces at different positions. This enables better parallelisation
during training compared to recurrent neural networks.

Retrieval-Augmented Generation, or RAG, combines a retriever module with
a generative language model. The retriever fetches relevant passages from
an external knowledge base, and the generator conditions its output on
both the query and the retrieved context. This approach reduces
hallucination and allows the model to access up-to-date information
without retraining.

ChromaDB is an open-source vector database designed for AI applications.
It stores embeddings alongside metadata, supports cosine similarity
search, and provides persistent storage. ChromaDB is commonly used as
the vector store component in RAG pipelines.
"""


@pytest.fixture(scope="module")
def sample_text_file():
    """Create a temporary text file with known content.

    Yields:
        Absolute path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(_DOC_CONTENT)
        fh.flush()
        yield fh.name

    # Cleanup after all tests in this module
    os.unlink(fh.name)


@pytest.fixture(scope="module")
def ingested_collection(sample_text_file: str):
    """Ingest the sample file into an isolated ChromaDB collection.

    Yields:
        The collection name used for the test.
    """
    # Clean up any leftover collection from a previous run
    try:
        vectorstore.delete_collection(_TEST_COLLECTION)
    except RuntimeError:
        pass  # Collection didn't exist — that's fine

    # Stage 1 — Load
    documents = loader.load_document(sample_text_file)
    assert len(documents) > 0, "Loader returned no documents."

    # Stage 2 — Chunk
    chunks = chunk_documents(documents, chunk_size=256, chunk_overlap=30)
    assert len(chunks) > 0, "Chunker produced no chunks."

    # Stage 3 — Embed
    embedded = embed_chunks(chunks)
    for chunk in embedded:
        assert "embedding" in chunk, "Chunk missing embedding."
        assert len(chunk["embedding"]) > 0, "Embedding vector is empty."

    # Stage 4 — Upsert
    n_upserted = vectorstore.upsert(embedded, collection_name=_TEST_COLLECTION)
    assert n_upserted == len(embedded), (
        f"Upserted {n_upserted} but expected {len(embedded)}."
    )

    yield _TEST_COLLECTION

    # Teardown — remove the test collection
    try:
        vectorstore.delete_collection(_TEST_COLLECTION)
    except RuntimeError:
        logger.warning("Could not clean up test collection '%s'.", _TEST_COLLECTION)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndQuery:
    """Verify that the full ingest → retrieve pipeline returns correct results."""

    def test_retrieve_returns_results(self, ingested_collection: str) -> None:
        """A basic query should return at least one result."""
        results = retrieve(
            "What is the Transformer architecture?",
            top_k=3,
            collection_name=ingested_collection,
        )
        assert len(results) > 0, "Retriever returned no results."

    def test_result_structure(self, ingested_collection: str) -> None:
        """Each result must contain text, score, source, and metadata."""
        results = retrieve(
            "How does multi-head attention work?",
            top_k=3,
            collection_name=ingested_collection,
        )
        for result in results:
            assert "text" in result, "Result missing 'text' key."
            assert "score" in result, "Result missing 'score' key."
            assert "source" in result, "Result missing 'source' key."
            assert "metadata" in result, "Result missing 'metadata' key."

    def test_scores_are_valid(self, ingested_collection: str) -> None:
        """Cosine similarity scores must be floats in [0, 1]."""
        results = retrieve(
            "What is RAG?",
            top_k=5,
            collection_name=ingested_collection,
        )
        for result in results:
            score = result["score"]
            assert isinstance(score, float), f"Score is not a float: {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_scores_sorted_descending(self, ingested_collection: str) -> None:
        """Results should be returned in descending order of score."""
        results = retrieve(
            "retrieval augmented generation",
            top_k=5,
            collection_name=ingested_collection,
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not in descending order: {scores}"
        )

    def test_relevant_content_ranked_high(self, ingested_collection: str) -> None:
        """A query about ChromaDB should surface the ChromaDB paragraph."""
        results = retrieve(
            "What is ChromaDB and how is it used in RAG?",
            top_k=3,
            collection_name=ingested_collection,
        )
        top_text = results[0]["text"].lower()
        assert "chromadb" in top_text or "vector" in top_text, (
            f"Top result does not mention ChromaDB or vectors: {top_text[:120]}"
        )

    def test_top_k_respected(self, ingested_collection: str) -> None:
        """Number of results must not exceed top_k."""
        for k in (1, 2, 3):
            results = retrieve(
                "attention mechanism",
                top_k=k,
                collection_name=ingested_collection,
            )
            assert len(results) <= k, (
                f"Requested top_k={k} but got {len(results)} results."
            )

    def test_empty_collection_returns_empty(self) -> None:
        """Querying a non-existent or empty collection returns no results."""
        empty_collection = "querybridge_test_empty"
        try:
            vectorstore.delete_collection(empty_collection)
        except RuntimeError:
            pass

        results = retrieve(
            "anything",
            top_k=3,
            collection_name=empty_collection,
        )
        assert results == [], f"Expected empty results, got {len(results)}."
