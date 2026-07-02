"""
router.py — Decision routing for QueryBridge.

Takes the output of the confidence scorer and gap detector, then routes
the query to the correct handler:

    - **answer**:   The knowledge base has sufficient context.  Generate
                    an LLM answer with source attribution.
    - **search**:   A partial or complete knowledge gap was detected.
                    Generate search queries → rank → execute → re‑score
                    the augmented context.
    - **escalate**: Contradictions are severe, confidence is critically
                    low, or the search path has been exhausted.  Produce
                    a structured escalation report.

The router does **not** call upstream modules itself — it receives their
outputs as arguments and decides what to do next.  The ``pipeline.py``
module is responsible for the full orchestration loop.

Output format::

    {
        "action": "answer" | "search" | "escalate",
        "answer": str | None,
        "sources": [str, ...],
        "escalation_report": dict | None,
        "search_queries": [str, ...] | None,
        "reasoning": str
    }
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence thresholds — aligned with scorer.py and gap_detector.py.
HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.4
CRITICAL_CONFIDENCE = 0.2

# Contradiction penalty above which escalation is forced (even at medium
# confidence).  This is the "medium confidence + contradiction" edge case.
SEVERE_CONTRADICTION_PENALTY = 0.2

# LLM settings for answer generation.
_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_answer_prompt(
    query: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Build the LLM prompt for answer generation with source attribution.

    Args:
        query: The user's original question.
        chunks: Retrieved context chunks, each with ``content`` and
            ``metadata`` keys.

    Returns:
        The assembled prompt string.
    """
    context_parts: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        text = chunk.get("content", "")
        context_parts.append(
            f"[Source {idx}: {source}]\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    return (
        "You are a helpful assistant. Answer the question using ONLY "
        "the provided context. Cite sources by their number "
        "(e.g. [Source 1]). If the context does not contain enough "
        "information to fully answer, say so explicitly.\n\n"
        f"### Context\n{context_block}\n\n"
        f"### Question\n{query}\n\n"
        "### Answer"
    )


def _call_llm(prompt: str) -> str:
    """Call the Groq LLM and return the text response.

    Args:
        prompt: The prompt to send.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If the API key is missing or the request fails.
    """
    import requests  # Local import to avoid circular issues at module load.

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            _GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _extract_sources(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract unique source file names from chunk metadata.

    Args:
        chunks: Retrieved context chunks.

    Returns:
        Deduplicated list of source identifiers.
    """
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _build_escalation_report(
    query: str,
    confidence: float,
    gap_result: dict[str, Any],
    contradiction_result: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a structured escalation report for human review.

    Args:
        query: The user's original question.
        confidence: The confidence score from the scorer.
        gap_result: Output from the gap detector.
        contradiction_result: Output from the contradiction detector.
        chunks: Retrieved context chunks.

    Returns:
        A dict containing the escalation details.
    """
    return {
        "query": query,
        "confidence": round(confidence, 4),
        "gap_type": gap_result.get("gap_type", "unknown"),
        "missing_info": gap_result.get("missing_info", ""),
        "contradictions_found": contradiction_result.get(
            "contradictions_found", False
        ),
        "contradiction_penalty": contradiction_result.get("penalty", 0.0),
        "conflicting_pairs": contradiction_result.get(
            "conflicting_pairs", []
        ),
        "sources_checked": _extract_sources(chunks),
        "reason": _determine_escalation_reason(
            confidence, gap_result, contradiction_result
        ),
    }


def _determine_escalation_reason(
    confidence: float,
    gap_result: dict[str, Any],
    contradiction_result: dict[str, Any],
) -> str:
    """Produce a human-readable reason for escalation.

    Args:
        confidence: The confidence score.
        gap_result: Output from the gap detector.
        contradiction_result: Output from the contradiction detector.

    Returns:
        A concise explanation string.
    """
    reasons: list[str] = []

    if confidence < CRITICAL_CONFIDENCE:
        reasons.append(
            f"Critically low confidence ({confidence:.2f})"
        )

    penalty = contradiction_result.get("penalty", 0.0)
    if penalty >= SEVERE_CONTRADICTION_PENALTY:
        n_pairs = len(contradiction_result.get("conflicting_pairs", []))
        reasons.append(
            f"Severe contradictions detected "
            f"(penalty={penalty:.2f}, {n_pairs} pair(s))"
        )

    gap_type = gap_result.get("gap_type", "none")
    if gap_type == "missing":
        reasons.append("Required information completely missing")
    elif gap_type == "contradictory":
        reasons.append("Retrieved chunks contain contradictory claims")

    return "; ".join(reasons) if reasons else "Escalation triggered by policy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(
    query: str,
    chunks: list[dict[str, Any]],
    scorer_result: dict[str, Any],
    gap_result: dict[str, Any],
    contradiction_result: dict[str, Any],
) -> dict[str, Any]:
    """Decide the appropriate action and execute it.

    Routing logic:
        1. **Escalate** if confidence < 0.2 (critically low).
        2. **Escalate** if confidence < 0.7 **and** contradiction
           penalty ≥ 0.2 (medium confidence with severe contradiction).
        3. **Search** if the gap detector says ``search``.
        4. **Answer** otherwise (gap_type is ``none`` or action is
           ``answer``).

    Args:
        query: The user's original question.
        chunks: Retrieved context chunks (list of dicts with
            ``content`` and ``metadata``).
        scorer_result: Output from ``scorer.score()``.
        gap_result: Output from ``gap_detector.detect_gaps()``.
        contradiction_result: Output from
            ``contradiction_detector.detect_contradictions()``.

    Returns:
        A dict with keys ``action``, ``answer``, ``sources``,
        ``escalation_report``, ``search_queries``, and ``reasoning``.

    Raises:
        TypeError: If any required argument has the wrong type.
    """
    confidence = scorer_result.get("confidence", 0.0)
    penalty = contradiction_result.get("penalty", 0.0)
    gap_action = gap_result.get("action", "answer")
    gap_type = gap_result.get("gap_type", "none")

    logger.info(
        "Routing query (confidence=%.3f, penalty=%.3f, "
        "gap_type=%s, gap_action=%s).",
        confidence, penalty, gap_type, gap_action,
    )

    # ----- Rule 1: critically low confidence → escalate -----
    if confidence < CRITICAL_CONFIDENCE:
        logger.info(
            "Escalating: confidence %.3f below critical threshold %.2f.",
            confidence, CRITICAL_CONFIDENCE,
        )
        report = _build_escalation_report(
            query, confidence, gap_result,
            contradiction_result, chunks,
        )
        return {
            "action": "escalate",
            "answer": None,
            "sources": _extract_sources(chunks),
            "escalation_report": report,
            "search_queries": None,
            "reasoning": (
                f"Confidence {confidence:.3f} is below the critical "
                f"threshold ({CRITICAL_CONFIDENCE}). Escalating to "
                "human review."
            ),
        }

    # ----- Rule 2: medium confidence + severe contradiction → escalate -----
    if confidence < HIGH_CONFIDENCE and penalty >= SEVERE_CONTRADICTION_PENALTY:
        logger.info(
            "Escalating: medium confidence %.3f with severe "
            "contradiction penalty %.3f.",
            confidence, penalty,
        )
        report = _build_escalation_report(
            query, confidence, gap_result,
            contradiction_result, chunks,
        )
        return {
            "action": "escalate",
            "answer": None,
            "sources": _extract_sources(chunks),
            "escalation_report": report,
            "search_queries": None,
            "reasoning": (
                f"Confidence {confidence:.3f} is below high threshold "
                f"and contradiction penalty ({penalty:.3f}) is severe. "
                "Cannot produce a reliable answer — escalating."
            ),
        }

    # ----- Rule 3: gap detector says search → search -----
    if gap_action == "search":
        logger.info(
            "Routing to search: gap_type=%s, missing_info='%s'.",
            gap_type, gap_result.get("missing_info", "")[:80],
        )
        return {
            "action": "search",
            "answer": None,
            "sources": _extract_sources(chunks),
            "escalation_report": None,
            "search_queries": None,  # Pipeline will generate them.
            "reasoning": (
                f"Gap detected (type={gap_type}). Knowledge base has "
                f"insufficient coverage. Triggering web search to fill "
                f"the gap."
            ),
        }

    # ----- Rule 4: answer directly -----
    logger.info("Routing to answer: generating LLM response.")

    try:
        prompt = _build_answer_prompt(query, chunks)
        answer_text = _call_llm(prompt)
    except RuntimeError as exc:
        logger.error("Answer generation failed: %s. Escalating.", exc)
        report = _build_escalation_report(
            query, confidence, gap_result,
            contradiction_result, chunks,
        )
        return {
            "action": "escalate",
            "answer": None,
            "sources": _extract_sources(chunks),
            "escalation_report": report,
            "search_queries": None,
            "reasoning": (
                f"Answer generation failed ({exc}). Escalating to "
                "human review as a safety fallback."
            ),
        }

    return {
        "action": "answer",
        "answer": answer_text,
        "sources": _extract_sources(chunks),
        "escalation_report": None,
        "search_queries": None,
        "reasoning": (
            f"Confidence {confidence:.3f} is sufficient and no critical "
            f"gaps detected (gap_type={gap_type}). Answering directly "
            "with source attribution."
        ),
    }
