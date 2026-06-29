"""
query_generator.py — Search query generation for QueryBridge.

When the gap detector identifies missing or incomplete information,
this module generates 3–5 diverse search queries to fill the gap.
Each query targets a **different information angle** — they are not
restatements of the original query.

Design decisions:
    - The LLM is prompted with the original query, the gap type, and
      a description of what information is missing.  This context
      ensures the generated queries are targeted, not generic.
    - A minimum of 3 and maximum of 5 queries are enforced.
    - A validation step filters out queries that are too short, too
      similar to the original query, or clearly restated duplicates.

Output format::

    ["query1", "query2", "query3", ...]
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

MIN_QUERIES = 3
MAX_QUERIES = 5
MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 200

# Token-overlap ratio above which two queries are considered duplicates.
_DUPLICATE_THRESHOLD = 0.75

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


def _build_query_generation_prompt(
    original_query: str,
    gap_type: str,
    missing_info: str,
) -> str:
    """Build the LLM prompt for generating search queries.

    Args:
        original_query: The user's original question.
        gap_type: The classified gap type (missing/partial/contradictory).
        missing_info: Description of what information is missing.

    Returns:
        A formatted prompt string.
    """
    return f"""You are a search query strategist. Given a user query that could not be fully answered from a local knowledge base, generate 3 to 5 diverse search queries to find the missing information online.

ORIGINAL USER QUERY:
\"\"\"{original_query}\"\"\"

GAP TYPE: {gap_type}
MISSING INFORMATION: {missing_info}

RULES:
- Generate exactly 3 to 5 search queries.
- Each query MUST target a DIFFERENT information angle or perspective.
- Do NOT simply restate or rephrase the original query.
- Make queries specific and search-engine-friendly (concise, keyword-rich).
- If the gap is "contradictory", focus queries on finding authoritative or official sources.
- If the gap is "partial", focus queries on the specific missing aspects.
- If the gap is "missing", cast a wider net with varied approaches.

Respond with ONLY a valid JSON array of strings (no markdown, no explanation):
["query 1", "query 2", "query 3"]"""


def _call_llm(prompt: str, model: str = _DEFAULT_MODEL) -> list[str]:
    """Send a prompt to the LLM and parse the JSON array response.

    Args:
        prompt: The full prompt string.
        model: Groq model identifier.

    Returns:
        List of query strings from the LLM.

    Raises:
        RuntimeError: If the LLM call or JSON parsing fails.
    """
    client = _get_llm_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
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
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM returned invalid JSON for query generation: %s.",
            raw_text[:200],
        )
        raise RuntimeError(
            "Failed to parse LLM query generation response."
        ) from exc

    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Expected a JSON array, got {type(parsed).__name__}."
        )

    return [str(q) for q in parsed]


def _normalise_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace.

    Args:
        text: Raw text string.

    Returns:
        Normalised string.
    """
    return " ".join(text.lower().split())


def _are_near_duplicates(query_a: str, query_b: str) -> bool:
    """Check if two queries are near-duplicates via token overlap.

    Args:
        query_a: First query string.
        query_b: Second query string.

    Returns:
        True if the token overlap ratio exceeds the threshold.
    """
    tokens_a = set(_normalise_text(query_a).split())
    tokens_b = set(_normalise_text(query_b).split())

    if not tokens_a or not tokens_b:
        return True

    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))

    overlap_ratio = len(intersection) / smaller if smaller > 0 else 1.0
    return overlap_ratio >= _DUPLICATE_THRESHOLD


def _validate_queries(
    queries: list[str],
    original_query: str,
) -> list[str]:
    """Filter and validate generated search queries.

    Removes queries that are:
        - Too short (fewer than ``MIN_QUERY_LENGTH`` characters).
        - Too long (more than ``MAX_QUERY_LENGTH`` characters).
        - Near-duplicates of the original query.
        - Near-duplicates of an already-accepted query.

    Args:
        queries: Raw list of generated query strings.
        original_query: The user's original query for comparison.

    Returns:
        Validated list with duplicates and invalid entries removed.
    """
    validated: list[str] = []

    for query in queries:
        query = query.strip()

        if len(query) < MIN_QUERY_LENGTH:
            logger.debug(
                "Dropping query (too short, %d chars): '%s'.",
                len(query), query[:50],
            )
            continue

        if len(query) > MAX_QUERY_LENGTH:
            logger.debug(
                "Dropping query (too long, %d chars): '%s'.",
                len(query), query[:50],
            )
            continue

        # Skip if too similar to the original query
        if _are_near_duplicates(query, original_query):
            logger.debug(
                "Dropping query (too similar to original): '%s'.",
                query[:80],
            )
            continue

        # Skip if too similar to an already-accepted query
        is_dup = False
        for accepted in validated:
            if _are_near_duplicates(query, accepted):
                logger.debug(
                    "Dropping near-duplicate query: '%s'.",
                    query[:80],
                )
                is_dup = True
                break
        if is_dup:
            continue

        validated.append(query)

    return validated


def _generate_fallback_queries(
    original_query: str,
    gap_type: str,
    missing_info: str,
) -> list[str]:
    """Generate rule-based fallback queries when the LLM fails.

    Produces simple query variants based on the original query and
    gap context, ensuring the pipeline is never left with zero
    candidates.

    Args:
        original_query: The user's original query.
        gap_type: The classified gap type.
        missing_info: Description of what is missing.

    Returns:
        List of 3 fallback query strings.
    """
    base = original_query.strip()
    queries = []

    # Variant 1: add "official" or "authoritative" framing
    if gap_type == "contradictory":
        queries.append(f"{base} official source")
    else:
        queries.append(f"{base} explained")

    # Variant 2: use missing_info if available
    if missing_info and missing_info.lower() not in ("none", "n/a", "nothing"):
        # Take the first sentence of missing_info as a query
        first_sentence = missing_info.split(".")[0].strip()
        if len(first_sentence) >= MIN_QUERY_LENGTH:
            queries.append(first_sentence)
        else:
            queries.append(f"{base} details")
    else:
        queries.append(f"{base} details")

    # Variant 3: research-style query
    queries.append(f"{base} research paper findings")

    logger.info("Using %d fallback queries.", len(queries))
    return queries[:MAX_QUERIES]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_queries(
    original_query: str,
    gap_result: dict[str, Any],
    model: str = _DEFAULT_MODEL,
) -> list[str]:
    """Generate 3–5 diverse search queries to fill a knowledge gap.

    Uses the LLM to produce targeted queries based on the original
    query, gap type, and missing information description.  Falls back
    to rule-based generation if the LLM call fails.

    Args:
        original_query: The user's original question.
        gap_result: Output from ``gap_detector.detect_gap()``.  Must
            have ``gap_type`` and ``missing_info`` keys.
        model: Groq model identifier.  Defaults to the value of
            ``GROQ_MODEL`` in the environment (or ``llama3-8b-8192``).

    Returns:
        List of 3–5 unique, validated search query strings.

    Raises:
        TypeError: If *original_query* is not a string.
        TypeError: If *gap_result* is not a dict.
        ValueError: If *original_query* is empty.
    """
    if not isinstance(original_query, str):
        raise TypeError(
            f"Expected a string query, got {type(original_query).__name__}."
        )
    if not isinstance(gap_result, dict):
        raise TypeError(
            f"Expected gap_result dict, got {type(gap_result).__name__}."
        )
    if not original_query.strip():
        raise ValueError("original_query must not be empty.")

    gap_type = gap_result.get("gap_type", "missing")
    missing_info = gap_result.get("missing_info", "")

    logger.info(
        "Generating search queries — gap_type=%s, query='%s'.",
        gap_type,
        original_query[:80],
    )

    # Attempt LLM-based generation
    try:
        prompt = _build_query_generation_prompt(
            original_query, gap_type, missing_info,
        )
        raw_queries = _call_llm(prompt, model=model)
        validated = _validate_queries(raw_queries, original_query)
    except RuntimeError:
        logger.warning(
            "LLM query generation failed — using fallback."
        )
        validated = _generate_fallback_queries(
            original_query, gap_type, missing_info,
        )

    # Enforce min/max bounds
    if len(validated) < MIN_QUERIES:
        logger.info(
            "Only %d valid queries — supplementing with fallbacks.",
            len(validated),
        )
        fallbacks = _generate_fallback_queries(
            original_query, gap_type, missing_info,
        )
        for fb in fallbacks:
            if len(validated) >= MIN_QUERIES:
                break
            # Only add if not a duplicate of existing
            is_dup = any(
                _are_near_duplicates(fb, v) for v in validated
            )
            if not is_dup:
                validated.append(fb)

    validated = validated[:MAX_QUERIES]

    logger.info(
        "Query generation complete: %d queries produced.",
        len(validated),
    )
    return validated
