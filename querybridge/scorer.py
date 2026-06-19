"""
scorer.py — Multi-factor confidence scoring for QueryBridge.

Computes a calibrated confidence score for a retrieval result by
combining four independent signals:

    confidence = (similarity × 0.4)
               + (coverage  × 0.4)
               + (agreement × 0.2)
               − contradiction_penalty

Signals:
    - **Retrieval similarity**: Mean cosine similarity of the top-k
      retrieved chunks.  Higher means the vector store returned
      passages that are semantically close to the query.
    - **Context coverage**: Measures whether the retrieved chunks
      collectively address the query.  Computed as the cosine
      similarity between the query embedding and the mean of the
      chunk embeddings (a simple "centroid" approach).
    - **Source agreement**: Ratio of chunks that originate from
      distinct sources.  When multiple independent documents agree
      on the same topic, confidence rises.
    - **Contradiction penalty**: Passed in from
      ``contradiction_detector.detect_contradictions()``.  Range
      is [0.0, 0.3].

Output format::

    {
        "confidence": float,
        "label": "high" | "medium" | "low",
        "reason": str,
        "signals": {
            "retrieval_similarity": float,
            "context_coverage": float,
            "source_agreement": float,
            "contradiction_penalty": float
        }
    }
"""

import logging
import math
from typing import Any

from querybridge import embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights — must sum to 1.0 (before penalty subtraction)
# ---------------------------------------------------------------------------

WEIGHT_SIMILARITY = 0.4
WEIGHT_COVERAGE = 0.4
WEIGHT_AGREEMENT = 0.2

# Confidence label thresholds
HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------


def _compute_retrieval_similarity(
    chunks: list[dict[str, Any]],
) -> float:
    """Compute mean retrieval similarity from chunk scores.

    Each chunk dict is expected to have a ``score`` key containing the
    cosine similarity (0–1) assigned by the retriever.

    Args:
        chunks: Retrieved chunk dicts with ``score`` fields.

    Returns:
        Mean similarity as a float in [0.0, 1.0].  Returns 0.0 if no
        chunks carry a valid score.
    """
    scores = [
        c["score"] for c in chunks
        if isinstance(c.get("score"), (int, float))
    ]
    if not scores:
        logger.warning("No valid scores found in chunks.")
        return 0.0

    mean_score = sum(scores) / len(scores)
    logger.debug("Retrieval similarity: %.4f (from %d scores).", mean_score, len(scores))
    return mean_score


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _compute_context_coverage(
    query: str,
    chunks: list[dict[str, Any]],
) -> float:
    """Measure how well retrieved chunks collectively cover the query.

    Embeds the query and computes the cosine similarity between the
    query vector and the centroid (mean) of all chunk embeddings.
    A high value means the retrieved passages, taken together, are
    semantically aligned with what the user asked.

    Args:
        query: The original user query string.
        chunks: Retrieved chunk dicts.  Each must have a ``text`` key.

    Returns:
        Coverage score in [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    try:
        query_embedding = embedder.embed_texts([query])[0]
    except RuntimeError as exc:
        logger.warning("Failed to embed query for coverage: %s", exc)
        return 0.0

    # Build chunk embeddings
    chunk_texts = [c["text"] for c in chunks if c.get("text")]
    if not chunk_texts:
        return 0.0

    try:
        chunk_embeddings = embedder.embed_texts(chunk_texts)
    except RuntimeError as exc:
        logger.warning("Failed to embed chunks for coverage: %s", exc)
        return 0.0

    # Compute centroid of chunk embeddings
    dim = len(chunk_embeddings[0])
    centroid = [0.0] * dim
    for emb in chunk_embeddings:
        for i, val in enumerate(emb):
            centroid[i] += val
    centroid = [v / len(chunk_embeddings) for v in centroid]

    similarity = _cosine_similarity(query_embedding, centroid)

    # Cosine similarity can be negative; treat negative as zero coverage
    coverage = max(0.0, similarity)

    logger.debug("Context coverage: %.4f.", coverage)
    return coverage


def _compute_source_agreement(
    chunks: list[dict[str, Any]],
) -> float:
    """Compute source agreement from chunk metadata.

    Agreement is higher when multiple independent sources contribute
    chunks.  A single-source retrieval gets a lower score because there
    is no corroboration.

    Formula::

        agreement = 1 - (1 / num_unique_sources)

    This yields 0.0 for one source, 0.5 for two, 0.67 for three, etc.

    Args:
        chunks: Retrieved chunk dicts with ``source`` or ``metadata``
            fields.

    Returns:
        Agreement score in [0.0, 1.0).
    """
    sources: set[str] = set()

    for chunk in chunks:
        source = chunk.get("source", "")
        if not source:
            # Try metadata fallback
            meta = chunk.get("metadata", {})
            source = meta.get("source", "")
        if source:
            sources.add(source)

    num_sources = max(len(sources), 1)

    agreement = 1.0 - (1.0 / num_sources)

    logger.debug(
        "Source agreement: %.4f (%d unique source(s)).",
        agreement,
        num_sources,
    )
    return agreement


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def _assign_label(confidence: float) -> tuple[str, str]:
    """Assign a human-readable label and reason based on the score.

    Args:
        confidence: The computed confidence score.

    Returns:
        Tuple of ``(label, reason)`` where label is one of
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    if confidence >= HIGH_THRESHOLD:
        return (
            "high",
            "Strong retrieval similarity, good coverage, and "
            "multiple sources agree.",
        )
    if confidence >= LOW_THRESHOLD:
        return (
            "medium",
            "Moderate confidence — some signals are weak. "
            "Consider verifying with additional sources.",
        )
    return (
        "low",
        "Low confidence — retrieval quality is poor or "
        "contradictions were detected. Gap detection recommended.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_confidence(
    query: str,
    chunks: list[dict[str, Any]],
    contradiction_penalty: float = 0.0,
) -> dict[str, Any]:
    """Compute a multi-factor confidence score for a retrieval result.

    Combines retrieval similarity, context coverage, source agreement,
    and a contradiction penalty into a single calibrated score with an
    interpretive label.

    Args:
        query: The original user query.
        chunks: Retrieved chunk dicts as returned by
            ``retriever.retrieve()``.  Each must have ``text`` and
            ``score`` keys.
        contradiction_penalty: Penalty from the contradiction detector,
            expected in [0.0, 0.3].  Defaults to 0.0 (no contradictions).

    Returns:
        Dict with ``confidence``, ``label``, ``reason``, and ``signals``
        keys.  See module docstring for the full schema.

    Raises:
        TypeError: If *chunks* is not a list.
        ValueError: If *contradiction_penalty* is negative.
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"Expected a list of chunk dicts, got {type(chunks).__name__}."
        )
    if contradiction_penalty < 0:
        raise ValueError(
            f"contradiction_penalty must be non-negative, "
            f"got {contradiction_penalty}."
        )

    # Handle empty retrieval
    if not chunks:
        logger.info("No chunks provided — returning zero confidence.")
        return {
            "confidence": 0.0,
            "label": "low",
            "reason": "No chunks were retrieved.",
            "signals": {
                "retrieval_similarity": 0.0,
                "context_coverage": 0.0,
                "source_agreement": 0.0,
                "contradiction_penalty": contradiction_penalty,
            },
        }

    # Compute individual signals
    similarity = _compute_retrieval_similarity(chunks)
    coverage = _compute_context_coverage(query, chunks)
    agreement = _compute_source_agreement(chunks)

    # Weighted combination minus penalty
    raw_confidence = (
        (similarity * WEIGHT_SIMILARITY)
        + (coverage * WEIGHT_COVERAGE)
        + (agreement * WEIGHT_AGREEMENT)
        - contradiction_penalty
    )

    # Clamp to valid range — the raw formula can go negative when the
    # contradiction penalty is large, or exceed 1.0 in edge cases.
    confidence = round(max(0.0, min(1.0, raw_confidence)), 4)

    label, reason = _assign_label(confidence)

    result = {
        "confidence": confidence,
        "label": label,
        "reason": reason,
        "signals": {
            "retrieval_similarity": round(similarity, 4),
            "context_coverage": round(coverage, 4),
            "source_agreement": round(agreement, 4),
            "contradiction_penalty": round(contradiction_penalty, 4),
        },
    }

    logger.info(
        "Confidence scored: %.4f (%s) — sim=%.2f, cov=%.2f, "
        "agr=%.2f, penalty=%.2f.",
        confidence,
        label,
        similarity,
        coverage,
        agreement,
        contradiction_penalty,
    )

    return result
