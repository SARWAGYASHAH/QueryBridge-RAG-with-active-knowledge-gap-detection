"""
search_ranker.py — Search query ranking and selection for QueryBridge.

After the query generator produces 3–5 candidate search queries, this
module scores each query for **specificity** and **informativeness**,
removes near-duplicates, and selects the single best query to execute
against a web search API.

Scoring signals:
    - **Specificity**: Longer queries with concrete nouns, numbers, or
      quoted phrases score higher.  Very short or overly generic queries
      (e.g. "tell me more") are penalised.
    - **Informativeness**: Queries that contain question words, comparison
      terms, or domain-specific tokens score higher.
    - **Diversity bonus**: When a query covers different keywords from the
      other candidates, it gets a small boost — this helps avoid sending
      a redundant query to the search engine.

Deduplication uses the same token-overlap approach as the query
generator to catch near-identical candidates that slipped through.

Output format::

    {
        "ranked_queries": [
            {"query": str, "score": float, "rank": int},
            ...
        ],
        "selected_query": str
    }
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DUPLICATE_THRESHOLD = 0.75

# Words that indicate a specific, well-formed search query.
_SPECIFICITY_KEYWORDS = {
    "how", "what", "why", "when", "where", "which", "who",
    "compare", "difference", "between", "versus", "vs",
    "best", "latest", "official", "research", "study",
    "definition", "example", "guide", "tutorial",
}

# Minimum meaningful query length (characters).
_MIN_QUERY_LENGTH = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _score_specificity(query: str) -> float:
    """Score a query for specificity on a 0–1 scale.

    Rewards queries that are:
        - Longer (more tokens = more specific).
        - Contain numbers or quoted phrases.
        - Not just a handful of vague words.

    Args:
        query: The candidate search query.

    Returns:
        Specificity score in [0.0, 1.0].
    """
    normalised = _normalise_text(query)
    tokens = normalised.split()

    if not tokens:
        return 0.0

    # Length signal — diminishing returns beyond 12 tokens
    length_score = min(len(tokens) / 12.0, 1.0)

    # Numbers boost — queries with dates, counts, or versions are specific
    has_numbers = 1.0 if re.search(r"\d", query) else 0.0

    # Quoted phrases boost — indicates exact-match intent
    has_quotes = 1.0 if '"' in query or "'" in query else 0.0

    score = (length_score * 0.6) + (has_numbers * 0.2) + (has_quotes * 0.2)
    return round(min(score, 1.0), 4)


def _score_informativeness(query: str) -> float:
    """Score a query for informativeness on a 0–1 scale.

    Rewards queries that contain question words, comparison terms,
    or domain-specific vocabulary that tends to yield better search
    results.

    Args:
        query: The candidate search query.

    Returns:
        Informativeness score in [0.0, 1.0].
    """
    normalised = _normalise_text(query)
    tokens = set(normalised.split())

    if not tokens:
        return 0.0

    # Count matches against specificity keywords
    keyword_matches = tokens & _SPECIFICITY_KEYWORDS
    keyword_ratio = len(keyword_matches) / max(len(tokens), 1)

    # Unique-word density — more diverse vocabulary = more informative
    unique_ratio = len(tokens) / max(len(normalised.split()), 1)

    score = (keyword_ratio * 0.5) + (unique_ratio * 0.5)
    return round(min(score, 1.0), 4)


def _score_diversity(
    query: str,
    other_queries: list[str],
) -> float:
    """Score how different a query is from the other candidates.

    A query that brings unique keywords to the set scores higher,
    discouraging redundant searches.

    Args:
        query: The candidate being scored.
        other_queries: The remaining candidates for comparison.

    Returns:
        Diversity score in [0.0, 1.0].
    """
    if not other_queries:
        return 1.0

    query_tokens = set(_normalise_text(query).split())
    if not query_tokens:
        return 0.0

    # Collect all tokens from the other queries
    other_tokens: set[str] = set()
    for other in other_queries:
        other_tokens.update(_normalise_text(other).split())

    if not other_tokens:
        return 1.0

    # Unique tokens not found in any other query
    unique_tokens = query_tokens - other_tokens
    diversity = len(unique_tokens) / max(len(query_tokens), 1)

    return round(min(diversity, 1.0), 4)


def _compute_final_score(
    specificity: float,
    informativeness: float,
    diversity: float,
) -> float:
    """Combine the three sub-scores into a single ranking score.

    Weights:
        - Specificity:      40%
        - Informativeness:   35%
        - Diversity:         25%

    Args:
        specificity: Specificity sub-score.
        informativeness: Informativeness sub-score.
        diversity: Diversity sub-score.

    Returns:
        Final score in [0.0, 1.0].
    """
    score = (
        (specificity * 0.40)
        + (informativeness * 0.35)
        + (diversity * 0.25)
    )
    return round(min(max(score, 0.0), 1.0), 4)


def _deduplicate_queries(queries: list[str]) -> list[str]:
    """Remove near-duplicate queries, keeping the first occurrence.

    Args:
        queries: List of candidate query strings.

    Returns:
        Deduplicated list preserving original order.
    """
    unique: list[str] = []

    for query in queries:
        is_dup = any(_are_near_duplicates(query, u) for u in unique)
        if not is_dup:
            unique.append(query)
        else:
            logger.debug(
                "Removed near-duplicate query: '%s'.", query[:80]
            )

    dropped = len(queries) - len(unique)
    if dropped:
        logger.info("Deduplication removed %d query/queries.", dropped)

    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_queries(
    queries: list[str],
    original_query: str = "",
) -> dict[str, Any]:
    """Rank candidate search queries and select the best one.

    Scores each query on specificity, informativeness, and diversity,
    then returns them sorted by descending score along with the
    top-ranked query as the ``selected_query``.

    Args:
        queries: List of candidate search query strings (typically
            3–5 from ``query_generator.generate_queries()``).
        original_query: The user's original query.  Used only for
            logging context.

    Returns:
        Dict with keys:
            - ``ranked_queries`` (list[dict]): Each dict has ``query``,
              ``score``, and ``rank`` keys, sorted best-first.
            - ``selected_query`` (str): The highest-scoring query.

    Raises:
        TypeError: If *queries* is not a list.
        ValueError: If *queries* is empty after deduplication.
    """
    if not isinstance(queries, list):
        raise TypeError(
            f"Expected a list of query strings, "
            f"got {type(queries).__name__}."
        )

    logger.info(
        "Ranking %d candidate queries (original='%s').",
        len(queries),
        original_query[:80],
    )

    # Filter out empty / too-short entries
    valid = [q.strip() for q in queries if len(q.strip()) >= _MIN_QUERY_LENGTH]
    if not valid:
        raise ValueError(
            "No valid queries to rank after filtering "
            f"(received {len(queries)} candidates)."
        )

    # Deduplicate
    unique = _deduplicate_queries(valid)
    if not unique:
        raise ValueError("All queries were duplicates — nothing to rank.")

    # Score each query
    scored: list[dict[str, Any]] = []

    for i, query in enumerate(unique):
        others = [q for j, q in enumerate(unique) if j != i]

        specificity = _score_specificity(query)
        informativeness = _score_informativeness(query)
        diversity = _score_diversity(query, others)
        final = _compute_final_score(specificity, informativeness, diversity)

        scored.append({
            "query": query,
            "score": final,
            "signals": {
                "specificity": specificity,
                "informativeness": informativeness,
                "diversity": diversity,
            },
        })

        logger.debug(
            "Query %d scored %.4f (spec=%.2f, info=%.2f, div=%.2f): '%s'.",
            i + 1, final, specificity, informativeness, diversity,
            query[:60],
        )

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(scored, start=1):
        ranked.append({
            "query": item["query"],
            "score": item["score"],
            "rank": rank,
        })

    selected = ranked[0]["query"]

    logger.info(
        "Query ranking complete: selected '%s' (score=%.4f) "
        "from %d candidates.",
        selected[:60],
        ranked[0]["score"],
        len(ranked),
    )

    return {
        "ranked_queries": ranked,
        "selected_query": selected,
    }
