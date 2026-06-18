"""
contradiction_detector.py — Contradiction detection for QueryBridge.

Compares retrieved chunks pairwise using an LLM prompt to identify
conflicting claims.  The output includes a boolean flag, a list of
conflicting pairs with explanations, and a penalty score (0.0–0.3)
that is passed to the confidence scorer.

Key design decisions:
    - Only chunks with sufficiently different content are compared
      (a cosine-similarity pre-filter skips near-duplicates to avoid
      false positives from paraphrased passages).
    - The LLM is asked to respond in strict JSON so that parsing is
      deterministic.
    - A minimum textual overlap threshold prevents flagging chunks
      that simply discuss the same topic from different angles.

Output format::

    {
        "contradictions_found": bool,
        "conflicting_pairs": [
            {
                "chunk_a": str,
                "chunk_b": str,
                "explanation": str
            },
            ...
        ],
        "penalty": float   # 0.0–0.3
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

MAX_PENALTY = 0.3
MIN_CHUNK_LENGTH = 30
SIMILARITY_SKIP_THRESHOLD = 0.95

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


def _normalise_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for comparison.

    Args:
        text: Raw chunk text.

    Returns:
        Normalised string.
    """
    return " ".join(text.lower().split())


def _texts_are_near_duplicates(text_a: str, text_b: str) -> bool:
    """Check whether two chunk texts are near-duplicates.

    Uses a simple token-overlap ratio to detect paraphrased or
    heavily overlapping passages that should NOT be compared for
    contradictions (they would produce false positives).

    Args:
        text_a: First chunk text.
        text_b: Second chunk text.

    Returns:
        True if the texts overlap above ``SIMILARITY_SKIP_THRESHOLD``.
    """
    tokens_a = set(_normalise_text(text_a).split())
    tokens_b = set(_normalise_text(text_b).split())

    if not tokens_a or not tokens_b:
        return True

    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))

    overlap_ratio = len(intersection) / smaller if smaller > 0 else 1.0
    return overlap_ratio >= SIMILARITY_SKIP_THRESHOLD


def _build_comparison_prompt(chunk_a: str, chunk_b: str) -> str:
    """Build the LLM prompt for comparing two chunks.

    Args:
        chunk_a: Text of the first chunk.
        chunk_b: Text of the second chunk.

    Returns:
        A formatted prompt string.
    """
    return f"""You are a factual consistency checker. Compare the two text passages below and determine if they contain any contradictory or conflicting claims.

IMPORTANT RULES:
- Only flag DIRECT contradictions where one passage states something that conflicts with the other.
- Do NOT flag differences in scope, detail level, or focus as contradictions.
- Do NOT flag complementary information as contradictions.
- Do NOT flag passages that discuss different aspects of the same topic as contradictions.
- Two passages can both be true even if they emphasize different things.

Passage A:
\"\"\"{chunk_a}\"\"\"

Passage B:
\"\"\"{chunk_b}\"\"\"

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{
    "has_contradiction": true or false,
    "explanation": "brief explanation of the contradiction if found, or 'No contradiction' if none"
}}"""


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
            max_tokens=256,
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
            "LLM returned invalid JSON: %s. Treating as no contradiction.",
            raw_text[:200],
        )
        return {"has_contradiction": False, "explanation": "Parse error — skipped"}


def _compute_penalty(num_contradictions: int, num_pairs: int) -> float:
    """Compute a contradiction penalty in [0.0, MAX_PENALTY].

    The penalty scales linearly with the proportion of conflicting
    pairs, capped at ``MAX_PENALTY`` (0.3).

    Args:
        num_contradictions: Number of conflicting pairs found.
        num_pairs: Total number of pairs compared.

    Returns:
        Penalty float in [0.0, 0.3].
    """
    if num_pairs == 0:
        return 0.0
    ratio = num_contradictions / num_pairs
    return round(min(ratio * MAX_PENALTY, MAX_PENALTY), 4)


def _generate_pairs(
    chunks: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Generate index pairs for pairwise comparison.

    Skips near-duplicate pairs and chunks that are too short to
    contain meaningful claims.

    Args:
        chunks: List of chunk dicts with ``text`` keys.

    Returns:
        List of ``(i, j)`` index tuples to compare.
    """
    pairs: list[tuple[int, int]] = []

    for i in range(len(chunks)):
        if len(chunks[i].get("text", "").strip()) < MIN_CHUNK_LENGTH:
            continue
        for j in range(i + 1, len(chunks)):
            if len(chunks[j].get("text", "").strip()) < MIN_CHUNK_LENGTH:
                continue
            if _texts_are_near_duplicates(chunks[i]["text"], chunks[j]["text"]):
                logger.debug(
                    "Skipping near-duplicate pair (%d, %d).", i, j
                )
                continue
            pairs.append((i, j))

    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_contradictions(
    chunks: list[dict[str, Any]],
    model: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """Compare retrieved chunks pairwise for contradictory claims.

    Generates all valid pairs from *chunks*, sends each pair to the
    LLM for factual consistency checking, and aggregates the results
    into a summary with a penalty score for the confidence scorer.

    Args:
        chunks: List of retrieved chunk dicts.  Each must have a
            ``text`` key.  Typically the output of
            ``retriever.retrieve()``.
        model: Groq model identifier.  Defaults to the value of
            ``GROQ_MODEL`` in the environment (or ``llama3-8b-8192``).

    Returns:
        Dict with keys:
            - ``contradictions_found`` (bool): Whether any contradictions
              were detected.
            - ``conflicting_pairs`` (list[dict]): Each dict has
              ``chunk_a``, ``chunk_b``, and ``explanation``.
            - ``penalty`` (float): Value in [0.0, 0.3] for the scorer.

    Raises:
        TypeError: If *chunks* is not a list.
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"Expected a list of chunk dicts, got {type(chunks).__name__}."
        )

    result: dict[str, Any] = {
        "contradictions_found": False,
        "conflicting_pairs": [],
        "penalty": 0.0,
    }

    if len(chunks) < 2:
        logger.info("Fewer than 2 chunks — skipping contradiction detection.")
        return result

    pairs = _generate_pairs(chunks)
    if not pairs:
        logger.info("No valid pairs to compare after filtering.")
        return result

    logger.info(
        "Comparing %d chunk pair(s) for contradictions.", len(pairs)
    )

    conflicting: list[dict[str, Any]] = []

    for i, j in pairs:
        text_a = chunks[i]["text"]
        text_b = chunks[j]["text"]

        prompt = _build_comparison_prompt(text_a, text_b)

        try:
            llm_result = _call_llm(prompt, model=model)
        except RuntimeError as exc:
            logger.warning(
                "LLM call failed for pair (%d, %d): %s. Skipping.", i, j, exc
            )
            continue

        has_contradiction = llm_result.get("has_contradiction", False)

        # Guard against LLM returning string "true"/"false"
        if isinstance(has_contradiction, str):
            has_contradiction = has_contradiction.lower() == "true"

        if has_contradiction:
            explanation = llm_result.get("explanation", "No details provided")
            conflicting.append(
                {
                    "chunk_a": text_a[:200],
                    "chunk_b": text_b[:200],
                    "explanation": explanation,
                }
            )
            logger.info(
                "Contradiction found between chunks %d and %d: %s",
                i,
                j,
                explanation[:100],
            )

    penalty = _compute_penalty(len(conflicting), len(pairs))

    result = {
        "contradictions_found": len(conflicting) > 0,
        "conflicting_pairs": conflicting,
        "penalty": penalty,
    }

    logger.info(
        "Contradiction detection complete: %d conflict(s), penalty=%.4f.",
        len(conflicting),
        penalty,
    )
    return result
