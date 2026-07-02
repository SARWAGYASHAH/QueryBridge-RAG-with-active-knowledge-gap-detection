"""
search.py — Web search integration for QueryBridge.

Executes search queries against external search APIs to retrieve
supplementary information when the local knowledge base has gaps.

Provider strategy:
    - **Primary**: Serper.dev Google Search API — fast, structured results.
    - **Fallback**: Tavily Search API — used when Serper is unavailable
      or returns an error (rate limit, network failure, etc.).

Both providers return results normalised to a common format::

    [
        {"title": str, "snippet": str, "url": str},
        ...
    ]

Rate-limit handling:
    - HTTP 429 from Serper triggers an automatic switch to Tavily.
    - If both providers fail, the module raises ``RuntimeError`` so
      the router can decide whether to escalate.
"""

import json
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERPER_URL = "https://google.serper.dev/search"
_TAVILY_URL = "https://api.tavily.com/search"

_DEFAULT_NUM_RESULTS = 5
_REQUEST_TIMEOUT = 15  # seconds

# Retry configuration for transient failures.
_MAX_RETRIES = 2
_INITIAL_BACKOFF = 1.0  # seconds — doubles on each retry
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503}


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _retry_with_backoff(
    request_fn,
    provider_name: str,
) -> "requests.Response":
    """Execute *request_fn* with retries on transient HTTP errors.

    Uses exponential backoff (1 s → 2 s) and honours the ``Retry-After``
    header when the server provides one.

    Args:
        request_fn: A zero-argument callable that returns a
            ``requests.Response``.
        provider_name: Human-readable name used in log messages
            (e.g. ``"Serper"``, ``"Tavily"``).

    Returns:
        The successful ``requests.Response``.

    Raises:
        RuntimeError: If all retries are exhausted or a non-transient
            error is encountered.
    """
    backoff = _INITIAL_BACKOFF
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = request_fn()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "%s request error (attempt %d/%d): %s",
                provider_name, attempt + 1, _MAX_RETRIES + 1, exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(
                f"{provider_name} request failed after "
                f"{_MAX_RETRIES + 1} attempts: {exc}"
            ) from exc

        if response.status_code not in _TRANSIENT_STATUS_CODES:
            return response

        # Transient error — honour Retry-After if provided.
        retry_after = response.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else backoff
        logger.warning(
            "%s returned HTTP %d (attempt %d/%d). "
            "Retrying in %.1f s.",
            provider_name, response.status_code,
            attempt + 1, _MAX_RETRIES + 1, wait,
        )

        if attempt < _MAX_RETRIES:
            time.sleep(wait)
            backoff *= 2
        else:
            raise RuntimeError(
                f"{provider_name} rate limit (HTTP {response.status_code}) "
                f"persisted after {_MAX_RETRIES + 1} attempts."
            )

    # Should never reach here, but satisfy the type checker.
    raise RuntimeError(f"{provider_name} request failed.")  # pragma: no cover


# ---------------------------------------------------------------------------
# Internal helpers — Serper
# ---------------------------------------------------------------------------


def _search_serper(
    query: str,
    num_results: int = _DEFAULT_NUM_RESULTS,
) -> list[dict[str, str]]:
    """Execute a search via the Serper.dev API.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of result dicts with ``title``, ``snippet``, and ``url``.

    Raises:
        RuntimeError: If the API key is missing, the request fails,
            or a rate limit (HTTP 429) is hit.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key or api_key == "your_serper_api_key_here":
        raise RuntimeError(
            "SERPER_API_KEY is not set. Add it to your .env file."
        )

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    logger.info("Searching Serper: '%s' (num=%d).", query[:80], num_results)

    def _do_request():
        return requests.post(
            _SERPER_URL,
            headers=headers,
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )

    response = _retry_with_backoff(_do_request, "Serper")

    if response.status_code != 200:
        raise RuntimeError(
            f"Serper returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to parse Serper JSON response."
        ) from exc

    return _parse_serper_results(data, num_results)


def _parse_serper_results(
    data: dict[str, Any],
    num_results: int,
) -> list[dict[str, str]]:
    """Parse Serper API response into normalised result dicts.

    Args:
        data: Raw JSON response from Serper.
        num_results: Maximum results to extract.

    Returns:
        List of normalised result dicts.
    """
    results: list[dict[str, str]] = []

    organic = data.get("organic", [])
    for item in organic[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
        })

    logger.info("Serper returned %d result(s).", len(results))
    return results


# ---------------------------------------------------------------------------
# Internal helpers — Tavily
# ---------------------------------------------------------------------------


def _search_tavily(
    query: str,
    num_results: int = _DEFAULT_NUM_RESULTS,
) -> list[dict[str, str]]:
    """Execute a search via the Tavily API (fallback provider).

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of result dicts with ``title``, ``snippet``, and ``url``.

    Raises:
        RuntimeError: If the API key is missing or the request fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",
    }

    logger.info(
        "Searching Tavily (fallback): '%s' (max=%d).",
        query[:80], num_results,
    )

    def _do_request():
        return requests.post(
            _TAVILY_URL,
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )

    response = _retry_with_backoff(_do_request, "Tavily")

    if response.status_code != 200:
        raise RuntimeError(
            f"Tavily returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to parse Tavily JSON response."
        ) from exc

    return _parse_tavily_results(data, num_results)


def _parse_tavily_results(
    data: dict[str, Any],
    num_results: int,
) -> list[dict[str, str]]:
    """Parse Tavily API response into normalised result dicts.

    Args:
        data: Raw JSON response from Tavily.
        num_results: Maximum results to extract.

    Returns:
        List of normalised result dicts.
    """
    results: list[dict[str, str]] = []

    items = data.get("results", [])
    for item in items[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "url": item.get("url", ""),
        })

    logger.info("Tavily returned %d result(s).", len(results))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    query: str,
    num_results: int = _DEFAULT_NUM_RESULTS,
) -> list[dict[str, str]]:
    """Execute a web search with automatic fallback.

    Tries Serper.dev first.  If Serper fails (rate limit, network
    error, missing key), falls back to Tavily.  If both fail, raises
    ``RuntimeError``.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return (3–5
            recommended).  Defaults to 5.

    Returns:
        List of result dicts, each containing:
            - ``title`` (str): Page title.
            - ``snippet`` (str): Text snippet or description.
            - ``url`` (str): Full URL of the result.

    Raises:
        TypeError: If *query* is not a string.
        ValueError: If *query* is empty.
        RuntimeError: If both search providers fail.
    """
    if not isinstance(query, str):
        raise TypeError(
            f"Expected a string query, got {type(query).__name__}."
        )
    if not query.strip():
        raise ValueError("Search query must not be empty.")

    logger.info("Starting web search: '%s'.", query[:80])

    # Try Serper first
    try:
        results = _search_serper(query, num_results=num_results)
        if results:
            return results
        logger.warning("Serper returned zero results — trying Tavily.")
    except RuntimeError as exc:
        logger.warning("Serper failed: %s — falling back to Tavily.", exc)

    # Fallback to Tavily
    try:
        results = _search_tavily(query, num_results=num_results)
        return results
    except RuntimeError as exc:
        logger.error("Tavily also failed: %s.", exc)
        raise RuntimeError(
            f"Both search providers failed for query: '{query[:80]}'. "
            f"Last error: {exc}"
        ) from exc
