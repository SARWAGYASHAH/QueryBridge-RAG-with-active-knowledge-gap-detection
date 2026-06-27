"""
gap_detector.py — Knowledge gap detection for QueryBridge.

Analyses the confidence score, retrieved chunks, and contradiction
results to classify what kind of knowledge gap exists (if any) and
decide the appropriate system action.

Gap types:
    - ``none``:          All query aspects covered with consistent
                         information.  Confidence is high.
    - ``partial``:       Some aspects of the query are addressed by the
                         retrieved context, but others are not.
    - ``missing``:       Required information is absent from every
                         retrieved chunk.
    - ``contradictory``: Retrieved chunks contain conflicting claims
                         that prevent a reliable answer.

Each gap type maps to an action:
    - ``answer``:   Generate an answer directly from retrieved context.
    - ``search``:   Trigger search query generation to fill the gap.
    - ``escalate``: Escalate to a human (used when contradictions are
                    severe or confidence is critically low).

Output format::

    {
        "gap_detected": bool,
        "gap_type": "missing" | "contradictory" | "partial" | "none",
        "missing_info": str,
        "action": "answer" | "search" | "escalate"
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
# Thresholds
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.4
SEVERE_PENALTY = 0.2
CRITICAL_CONFIDENCE = 0.2

# LLM settings
_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_llm_client():
    """Return a Groq client for LLM calls.

    Returns:
        A ``groq.Groq`` client instance.

    Raises:
        RuntimeError: If the Groq SDK is not installed or the API key
            is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    try:
        from groq import Groq  # type: ignore

        return Groq(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError(
            "groq SDK is not installed. Run: pip install groq"
        ) from exc


def _build_gap_analysis_prompt(
    query: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Build the LLM prompt for gap analysis.

    The prompt asks the LLM to assess whether the retrieved chunks
    collectively answer the user's query, and if not, what specific
    information is missing.

    Args:
        query: The original user query.
        chunks: Retrieved chunk dicts with ``text`` keys.

    Returns:
        A formatted prompt string.
    """
    chunk_texts = "\n---\n".join(
        f"[Chunk {i + 1}]: {c.get('text', '')[:500]}"
        for i, c in enumerate(chunks)
    )

    return f"""You are a precise information gap analyst. Given a user query and retrieved text passages, determine whether the passages fully answer the query.

USER QUERY:
\"\"\"{query}\"\"\"

RETRIEVED PASSAGES:
{chunk_texts}

Analyse the passages against the query and respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{
    "coverage": "full" or "partial" or "none",
    "missing_aspects": "specific description of what information is missing or not covered, or 'Nothing — all aspects are addressed' if full coverage",
    "reasoning": "brief explanation of your assessment"
}}

RULES:
- "full" means ALL aspects of the query are clearly addressed by the passages.
- "partial" means SOME aspects are addressed but specific parts are missing.
- "none" means the passages do NOT address the query at all.
- Be specific about what is missing — do not say "more details needed" without saying WHAT details.
- If the query has multiple parts or sub-questions, check each one individually."""


def _call_llm(prompt: str, model: str = _DEFAULT_MODEL) -> dict[str, Any]:
    """Send a prompt to the LLM and parse the JSON response.

    Args:
        prompt: The full prompt string.
        model: Groq model identifier.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        RuntimeError: If the LLM call or JSON parsing fails.
    """
    client = _get_llm_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM returned invalid JSON for gap analysis: %s. "
            "Falling back to rule-based classification.",
            raw_text[:200],
        )
        raise RuntimeError(
            "Failed to parse LLM gap analysis response."
        ) from exc


def _classify_from_llm_response(
    llm_result: dict[str, Any],
    contradiction_penalty: float,
) -> tuple[str, str]:
    """Derive gap type and missing info from the LLM response.

    Combines the LLM's coverage assessment with contradiction data
    to produce the final gap classification.

    Args:
        llm_result: Parsed JSON from the LLM gap analysis.
        contradiction_penalty: Penalty from the contradiction detector.

    Returns:
        Tuple of ``(gap_type, missing_info)``.
    """
    coverage = llm_result.get("coverage", "none").lower().strip()
    missing_aspects = llm_result.get(
        "missing_aspects",
        "Unable to determine missing information.",
    )

    # Contradictions override coverage assessment when severe
    if contradiction_penalty >= SEVERE_PENALTY:
        return (
            "contradictory",
            f"Contradictory sources detected (penalty={contradiction_penalty:.2f}). "
            f"LLM assessment: {missing_aspects}",
        )

    if coverage == "full":
        return "none", "Nothing — all aspects are addressed."

    if coverage == "partial":
        return "partial", missing_aspects

    # coverage == "none" or unrecognised value
    return "missing", missing_aspects


def _classify_rule_based(
    confidence: float,
    contradiction_penalty: float,
    chunks: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """Rule-based fallback when the LLM call fails or is unavailable.

    Uses confidence score, contradiction penalty, and chunk count
    to determine gap type, missing info description, and action.

    Args:
        confidence: Confidence score from the scorer (0.0–1.0).
        contradiction_penalty: Penalty from the contradiction detector.
        chunks: Retrieved chunk dicts.

    Returns:
        Tuple of ``(gap_type, missing_info, action)``.
    """
    if contradiction_penalty >= SEVERE_PENALTY:
        return (
            "contradictory",
            "Retrieved sources contain conflicting claims that "
            "prevent a reliable answer.",
            "search",
        )

    if not chunks:
        return (
            "missing",
            "No relevant passages were retrieved from the knowledge base.",
            "search",
        )

    if confidence >= HIGH_CONFIDENCE:
        return (
            "none",
            "Nothing — all aspects are addressed.",
            "answer",
        )

    if confidence >= LOW_CONFIDENCE:
        return (
            "partial",
            "Some aspects of the query are addressed, but the "
            "confidence level suggests incomplete coverage.",
            "search",
        )

    return (
        "missing",
        "Retrieved passages do not adequately address the query. "
        "Confidence is below the acceptable threshold.",
        "search",
    )


def _determine_action(
    gap_type: str,
    confidence: float,
    contradiction_penalty: float,
) -> str:
    """Determine the system action based on gap type and signals.

    Args:
        gap_type: One of ``"none"``, ``"partial"``, ``"missing"``,
            or ``"contradictory"``.
        confidence: Confidence score (0.0–1.0).
        contradiction_penalty: Penalty from contradictions.

    Returns:
        Action string: ``"answer"``, ``"search"``, or ``"escalate"``.
    """
    if gap_type == "none":
        return "answer"

    # Severe contradictions with very low confidence → escalate
    if (
        gap_type == "contradictory"
        and contradiction_penalty >= SEVERE_PENALTY
        and confidence < LOW_CONFIDENCE
    ):
        return "escalate"

    # Critically low confidence regardless of gap type → escalate
    if confidence < CRITICAL_CONFIDENCE:
        return "escalate"

    # All other gaps → try searching for more information
    return "search"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_gap(
    query: str,
    chunks: list[dict[str, Any]],
    score_result: dict[str, Any],
    contradiction_result: dict[str, Any],
    model: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """Detect and classify knowledge gaps in retrieved context.

    Combines LLM-based coverage analysis with rule-based signals
    (confidence score, contradiction penalty) to produce a gap
    classification and recommended action.

    If the LLM call fails, falls back to a purely rule-based
    classification using the confidence score and contradiction data.

    Args:
        query: The original user query.
        chunks: Retrieved chunk dicts as returned by
            ``retriever.retrieve()``.
        score_result: Output from ``scorer.score_confidence()``.
            Must have ``confidence`` and ``signals`` keys.
        contradiction_result: Output from
            ``contradiction_detector.detect_contradictions()``.
            Must have ``contradictions_found`` and ``penalty`` keys.
        model: Groq model identifier.  Defaults to the value of
            ``GROQ_MODEL`` in the environment (or ``llama3-8b-8192``).

    Returns:
        Dict with keys:
            - ``gap_detected`` (bool): Whether a gap was found.
            - ``gap_type`` (str): One of ``"missing"``,
              ``"contradictory"``, ``"partial"``, or ``"none"``.
            - ``missing_info`` (str): Description of what is missing.
            - ``action`` (str): One of ``"answer"``, ``"search"``,
              or ``"escalate"``.

    Raises:
        TypeError: If *chunks* is not a list.
        TypeError: If *score_result* or *contradiction_result* are
            not dicts.
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"Expected a list of chunk dicts, got {type(chunks).__name__}."
        )
    if not isinstance(score_result, dict):
        raise TypeError(
            f"Expected score_result dict, got {type(score_result).__name__}."
        )
    if not isinstance(contradiction_result, dict):
        raise TypeError(
            f"Expected contradiction_result dict, "
            f"got {type(contradiction_result).__name__}."
        )

    confidence = score_result.get("confidence", 0.0)
    contradiction_penalty = contradiction_result.get("penalty", 0.0)

    logger.info(
        "Starting gap detection — confidence=%.4f, "
        "contradiction_penalty=%.4f, chunks=%d.",
        confidence,
        contradiction_penalty,
        len(chunks),
    )

    # Fast path: high confidence, no contradictions → no gap
    if confidence >= HIGH_CONFIDENCE and contradiction_penalty == 0.0:
        logger.info("High confidence with no contradictions — no gap.")
        return {
            "gap_detected": False,
            "gap_type": "none",
            "missing_info": "Nothing — all aspects are addressed.",
            "action": "answer",
        }

    # Fast path: no chunks at all → missing
    if not chunks:
        logger.info("No chunks retrieved — gap type is 'missing'.")
        action = _determine_action("missing", confidence, contradiction_penalty)
        return {
            "gap_detected": True,
            "gap_type": "missing",
            "missing_info": (
                "No relevant passages were retrieved from the "
                "knowledge base."
            ),
            "action": action,
        }

    # LLM-based gap analysis for nuanced cases
    try:
        prompt = _build_gap_analysis_prompt(query, chunks)
        llm_result = _call_llm(prompt, model=model)
        gap_type, missing_info = _classify_from_llm_response(
            llm_result, contradiction_penalty
        )
    except RuntimeError:
        logger.warning(
            "LLM gap analysis failed — falling back to rule-based."
        )
        gap_type, missing_info, _ = _classify_rule_based(
            confidence, contradiction_penalty, chunks
        )

    action = _determine_action(gap_type, confidence, contradiction_penalty)
    gap_detected = gap_type != "none"

    result = {
        "gap_detected": gap_detected,
        "gap_type": gap_type,
        "missing_info": missing_info,
        "action": action,
    }

    logger.info(
        "Gap detection complete: detected=%s, type=%s, action=%s.",
        gap_detected,
        gap_type,
        action,
    )
    return result
