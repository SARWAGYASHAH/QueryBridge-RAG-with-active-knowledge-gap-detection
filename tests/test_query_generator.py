"""
test_query_generator.py — Unit tests for the query generation module.

Tests cover:
    - LLM-based query generation with mocked responses.
    - Near-duplicate detection between queries.
    - Validation logic (length, duplicate filtering).
    - Fallback generation when LLM fails.
    - Min/max query count enforcement.
    - Input validation (TypeError, ValueError).
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from querybridge.query_generator import (
    MAX_QUERIES,
    MIN_QUERIES,
    MIN_QUERY_LENGTH,
    _are_near_duplicates,
    _generate_fallback_queries,
    _validate_queries,
    generate_queries,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gap_result(
    gap_type: str = "missing",
    missing_info: str = "Training data details are not covered.",
) -> dict[str, Any]:
    """Create a mock gap_result dict."""
    return {
        "gap_detected": True,
        "gap_type": gap_type,
        "missing_info": missing_info,
        "action": "search",
    }


# ---------------------------------------------------------------------------
# Near-duplicate detection tests
# ---------------------------------------------------------------------------


class TestNearDuplicates:
    """Tests for the _are_near_duplicates function."""

    def test_identical_queries_are_duplicates(self) -> None:
        assert _are_near_duplicates(
            "transformer architecture details",
            "transformer architecture details",
        ) is True

    def test_case_insensitive_duplicates(self) -> None:
        assert _are_near_duplicates(
            "Transformer Architecture Details",
            "transformer architecture details",
        ) is True

    def test_different_queries_are_not_duplicates(self) -> None:
        assert _are_near_duplicates(
            "transformer attention mechanism paper",
            "GPT-4 training data and compute cost",
        ) is False

    def test_partially_overlapping_queries(self) -> None:
        """Queries with some overlap but below threshold → not duplicates."""
        result = _are_near_duplicates(
            "transformer self attention mechanism explained",
            "transformer training data requirements official",
        )
        # Only "transformer" overlaps — well below threshold
        assert result is False


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateQueries:
    """Tests for the _validate_queries function."""

    def test_filters_short_queries(self) -> None:
        queries = ["ok", "too short", "a valid search query about transformers"]
        result = _validate_queries(queries, "original question about AI")
        assert all(len(q) >= MIN_QUERY_LENGTH for q in result)

    def test_filters_original_duplicates(self) -> None:
        """Queries too similar to the original are removed."""
        original = "what is the transformer architecture"
        queries = [
            "what is the transformer architecture",  # exact dup
            "GPT-4 training compute requirements official",
            "self attention mechanism multi head explained",
        ]
        result = _validate_queries(queries, original)
        assert original not in result
        assert len(result) == 2

    def test_filters_inter_duplicates(self) -> None:
        """Near-duplicate queries among themselves are collapsed."""
        queries = [
            "transformer architecture parameters official paper",
            "transformer architecture parameters official source",
            "self attention mechanism multi head explained clearly",
        ]
        result = _validate_queries(queries, "unrelated original query here")
        # First two are near-duplicates — only one should survive
        assert len(result) == 2

    def test_preserves_valid_diverse_queries(self) -> None:
        queries = [
            "transformer self attention mechanism explained",
            "GPT-4 training data and compute cost",
            "BERT vs GPT architecture comparison paper",
        ]
        result = _validate_queries(queries, "unrelated original query text")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Fallback generation tests
# ---------------------------------------------------------------------------


class TestFallbackGeneration:
    """Tests for _generate_fallback_queries."""

    def test_generates_at_least_three_queries(self) -> None:
        queries = _generate_fallback_queries(
            "what is attention",
            "missing",
            "Training details are not covered.",
        )
        assert len(queries) >= 3

    def test_contradictory_type_adds_official_source(self) -> None:
        queries = _generate_fallback_queries(
            "model parameter count",
            "contradictory",
            "Conflicting claims about parameters.",
        )
        assert any("official" in q.lower() for q in queries)

    def test_uses_missing_info_as_query(self) -> None:
        """Missing info description should appear as a query variant."""
        queries = _generate_fallback_queries(
            "transformer architecture",
            "partial",
            "Training data sources are not mentioned in retrieved passages.",
        )
        # At least one query should contain content from missing_info
        assert any("training" in q.lower() for q in queries)

    def test_respects_max_queries(self) -> None:
        queries = _generate_fallback_queries(
            "test query",
            "missing",
            "Everything is missing from the knowledge base.",
        )
        assert len(queries) <= MAX_QUERIES


# ---------------------------------------------------------------------------
# LLM integration tests (mocked)
# ---------------------------------------------------------------------------


class TestGenerateQueriesWithMockedLLM:
    """Tests for generate_queries with mocked Groq API calls."""

    @patch("querybridge.query_generator._get_llm_client")
    def test_successful_generation(self, mock_client_fn: MagicMock) -> None:
        """LLM returns valid diverse queries."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            "transformer architecture official paper specifications",
            "vaswani et al 2017 model parameter count details",
            "attention is all you need model dimensions layers",
            "transformer base vs large model comparison official",
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = generate_queries(
            "How many parameters does the transformer have?",
            _make_gap_result(gap_type="contradictory"),
        )
        assert len(result) >= MIN_QUERIES
        assert len(result) <= MAX_QUERIES
        assert all(isinstance(q, str) for q in result)

    @patch("querybridge.query_generator._get_llm_client")
    def test_llm_failure_uses_fallback(
        self,
        mock_client_fn: MagicMock,
    ) -> None:
        """When LLM fails, fallback queries are generated."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "API timeout"
        )
        mock_client_fn.return_value = mock_client

        result = generate_queries(
            "What is self-attention in transformers?",
            _make_gap_result(),
        )
        assert len(result) >= MIN_QUERIES
        assert all(isinstance(q, str) for q in result)

    @patch("querybridge.query_generator._get_llm_client")
    def test_supplements_when_too_few_valid(
        self,
        mock_client_fn: MagicMock,
    ) -> None:
        """When LLM returns queries that mostly fail validation,
        fallbacks supplement to meet MIN_QUERIES."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Two queries are too short, one is valid
        mock_response.choices[0].message.content = json.dumps([
            "short",
            "tiny",
            "a perfectly valid search query about neural networks",
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = generate_queries(
            "How do neural networks learn?",
            _make_gap_result(),
        )
        assert len(result) >= MIN_QUERIES

    @patch("querybridge.query_generator._get_llm_client")
    def test_caps_at_max_queries(
        self,
        mock_client_fn: MagicMock,
    ) -> None:
        """Even if LLM returns more, output is capped at MAX_QUERIES."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            "transformer architecture deep dive official paper",
            "self attention mechanism mathematical explanation",
            "multi head attention implementation details code",
            "positional encoding in transformers explained",
            "transformer encoder decoder architecture comparison",
            "feed forward network in transformer blocks",
            "layer normalization residual connections transformers",
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = generate_queries(
            "Explain the complete transformer architecture",
            _make_gap_result(),
        )
        assert len(result) <= MAX_QUERIES


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for type checking in generate_queries."""

    def test_query_not_string_raises(self) -> None:
        with pytest.raises(TypeError, match="string query"):
            generate_queries(
                123,  # type: ignore
                _make_gap_result(),
            )

    def test_gap_result_not_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="gap_result dict"):
            generate_queries(
                "valid query",
                "not a dict",  # type: ignore
            )

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            generate_queries(
                "   ",
                _make_gap_result(),
            )
