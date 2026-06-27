"""
test_gap_detector.py — Unit tests for the gap detection module.

Tests cover:
    - High-confidence fast path returning ``none`` gap.
    - Empty retrieval producing ``missing`` gap.
    - Rule-based fallback for partial and contradictory gaps.
    - Action determination logic (answer / search / escalate).
    - LLM-based gap analysis with mocked Groq responses.
    - Misclassification guard: partial must not be labelled missing
      when chunks are present and confidence is moderate.
    - Input validation (TypeError on bad arguments).
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from querybridge.gap_detector import (
    CRITICAL_CONFIDENCE,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    SEVERE_PENALTY,
    _chunks_have_partial_relevance,
    _classify_from_llm_response,
    _classify_rule_based,
    _determine_action,
    detect_gap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunks(n: int = 3, score: float = 0.8) -> list[dict[str, Any]]:
    """Create n dummy chunks for testing."""
    return [
        {
            "text": f"This is test chunk number {i} with enough content "
                    f"to be meaningful for gap analysis testing purposes.",
            "score": score,
            "source": f"doc_{i}.pdf",
        }
        for i in range(n)
    ]


def _make_score_result(
    confidence: float = 0.85,
    label: str = "high",
) -> dict[str, Any]:
    """Create a mock score_result dict."""
    return {
        "confidence": confidence,
        "label": label,
        "reason": "Test reason.",
        "signals": {
            "retrieval_similarity": 0.8,
            "context_coverage": 0.9,
            "source_agreement": 0.5,
            "contradiction_penalty": 0.0,
        },
    }


def _make_contradiction_result(
    found: bool = False,
    penalty: float = 0.0,
) -> dict[str, Any]:
    """Create a mock contradiction_result dict."""
    return {
        "contradictions_found": found,
        "conflicting_pairs": [],
        "penalty": penalty,
    }


# ---------------------------------------------------------------------------
# Fast-path tests
# ---------------------------------------------------------------------------


class TestFastPaths:
    """Tests for the optimised fast paths in detect_gap."""

    def test_high_confidence_no_contradictions(self) -> None:
        """High confidence + zero penalty → no gap, action=answer."""
        result = detect_gap(
            query="What is attention?",
            chunks=_make_chunks(),
            score_result=_make_score_result(confidence=0.85),
            contradiction_result=_make_contradiction_result(),
        )
        assert result["gap_detected"] is False
        assert result["gap_type"] == "none"
        assert result["action"] == "answer"

    def test_empty_chunks_returns_missing(self) -> None:
        """No chunks → gap_type=missing, action=search."""
        result = detect_gap(
            query="What is attention?",
            chunks=[],
            score_result=_make_score_result(confidence=0.0, label="low"),
            contradiction_result=_make_contradiction_result(),
        )
        assert result["gap_detected"] is True
        assert result["gap_type"] == "missing"
        assert result["action"] in ("search", "escalate")


# ---------------------------------------------------------------------------
# Rule-based classification tests
# ---------------------------------------------------------------------------


class TestRuleBasedClassification:
    """Tests for the _classify_rule_based fallback."""

    def test_severe_contradiction_returns_contradictory(self) -> None:
        """High contradiction penalty → contradictory gap."""
        gap_type, missing_info, action = _classify_rule_based(
            confidence=0.5,
            contradiction_penalty=SEVERE_PENALTY,
            chunks=_make_chunks(),
        )
        assert gap_type == "contradictory"
        assert action == "search"

    def test_high_confidence_returns_none(self) -> None:
        """High confidence + no contradiction → no gap."""
        gap_type, _, action = _classify_rule_based(
            confidence=HIGH_CONFIDENCE,
            contradiction_penalty=0.0,
            chunks=_make_chunks(),
        )
        assert gap_type == "none"
        assert action == "answer"

    def test_medium_confidence_returns_partial(self) -> None:
        """Medium confidence → partial gap.

        This is the key test: with chunks present and confidence
        between LOW and HIGH thresholds, the gap must be 'partial'
        — NOT 'missing'.
        """
        gap_type, _, action = _classify_rule_based(
            confidence=0.55,
            contradiction_penalty=0.0,
            chunks=_make_chunks(),
        )
        assert gap_type == "partial", (
            f"Expected 'partial' for medium confidence, got '{gap_type}'. "
            f"This is the partial-vs-missing misclassification bug."
        )
        assert action == "search"

    def test_low_confidence_returns_missing(self) -> None:
        """Low confidence → missing gap."""
        gap_type, _, action = _classify_rule_based(
            confidence=0.2,
            contradiction_penalty=0.0,
            chunks=_make_chunks(),
        )
        assert gap_type == "missing"
        assert action == "search"

    def test_no_chunks_returns_missing(self) -> None:
        """Empty chunk list → missing gap."""
        gap_type, _, action = _classify_rule_based(
            confidence=0.0,
            contradiction_penalty=0.0,
            chunks=[],
        )
        assert gap_type == "missing"


# ---------------------------------------------------------------------------
# Action determination tests
# ---------------------------------------------------------------------------


class TestDetermineAction:
    """Tests for the _determine_action function."""

    def test_none_gap_returns_answer(self) -> None:
        assert _determine_action("none", 0.85, 0.0) == "answer"

    def test_partial_gap_returns_search(self) -> None:
        assert _determine_action("partial", 0.55, 0.0) == "search"

    def test_missing_gap_returns_search(self) -> None:
        assert _determine_action("missing", 0.3, 0.0) == "search"

    def test_contradictory_severe_low_confidence_returns_escalate(
        self,
    ) -> None:
        """Severe contradiction + low confidence → escalate."""
        action = _determine_action(
            "contradictory",
            confidence=0.3,
            contradiction_penalty=SEVERE_PENALTY,
        )
        assert action == "escalate"

    def test_critical_confidence_returns_escalate(self) -> None:
        """Confidence below critical threshold → escalate."""
        action = _determine_action(
            "missing",
            confidence=CRITICAL_CONFIDENCE - 0.01,
            contradiction_penalty=0.0,
        )
        assert action == "escalate"

    def test_contradictory_high_confidence_returns_search(self) -> None:
        """Contradictory but confidence above low → search, not escalate."""
        action = _determine_action(
            "contradictory",
            confidence=0.55,
            contradiction_penalty=SEVERE_PENALTY,
        )
        assert action == "search"


# ---------------------------------------------------------------------------
# LLM response classification tests
# ---------------------------------------------------------------------------


class TestClassifyFromLLMResponse:
    """Tests for _classify_from_llm_response."""

    def test_full_coverage_returns_none(self) -> None:
        gap_type, _ = _classify_from_llm_response(
            {"coverage": "full", "missing_aspects": "Nothing"},
            contradiction_penalty=0.0,
        )
        assert gap_type == "none"

    def test_partial_coverage_returns_partial(self) -> None:
        gap_type, missing = _classify_from_llm_response(
            {
                "coverage": "partial",
                "missing_aspects": "Training details are not covered.",
            },
            contradiction_penalty=0.0,
        )
        assert gap_type == "partial"
        assert "Training details" in missing

    def test_no_coverage_no_chunks_returns_missing(self) -> None:
        """LLM reports none + no chunks → truly missing."""
        gap_type, _ = _classify_from_llm_response(
            {"coverage": "none", "missing_aspects": "Everything is missing."},
            contradiction_penalty=0.0,
            chunks=None,
        )
        assert gap_type == "missing"

    def test_no_coverage_low_score_chunks_returns_missing(self) -> None:
        """LLM reports none + chunks with low scores → still missing."""
        low_score_chunks = [
            {"text": "Irrelevant content", "score": 0.2, "source": "a.pdf"},
        ]
        gap_type, _ = _classify_from_llm_response(
            {"coverage": "none", "missing_aspects": "Everything is missing."},
            contradiction_penalty=0.0,
            chunks=low_score_chunks,
        )
        assert gap_type == "missing"

    def test_no_coverage_with_relevant_chunks_returns_partial(self) -> None:
        """Regression: LLM says 'none' but chunks have decent scores.

        This is the key misclassification fix — when the retriever
        found passages with reasonable similarity, the gap should be
        'partial' (some related content found) not 'missing' (nothing
        found at all).
        """
        relevant_chunks = _make_chunks(n=3, score=0.65)
        gap_type, _ = _classify_from_llm_response(
            {"coverage": "none", "missing_aspects": "Topic not found."},
            contradiction_penalty=0.0,
            chunks=relevant_chunks,
        )
        assert gap_type == "partial", (
            f"Expected 'partial' when chunks have relevant scores, "
            f"got '{gap_type}'. This is the partial-vs-missing fix."
        )

    def test_severe_contradiction_overrides_coverage(self) -> None:
        """Even with full coverage, severe contradictions → contradictory."""
        gap_type, _ = _classify_from_llm_response(
            {"coverage": "full", "missing_aspects": "Nothing"},
            contradiction_penalty=SEVERE_PENALTY,
        )
        assert gap_type == "contradictory"


# ---------------------------------------------------------------------------
# LLM integration tests (mocked)
# ---------------------------------------------------------------------------


class TestDetectGapWithMockedLLM:
    """Tests for detect_gap with mocked Groq API calls."""

    @patch("querybridge.gap_detector._get_llm_client")
    def test_llm_partial_coverage(self, mock_client_fn: MagicMock) -> None:
        """LLM reports partial coverage → gap_type=partial."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "coverage": "partial",
            "missing_aspects": "The query asks about training data but "
                               "the passages only cover architecture.",
            "reasoning": "Architecture is covered but training is not.",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = detect_gap(
            query="What is the architecture and training data?",
            chunks=_make_chunks(),
            score_result=_make_score_result(confidence=0.55, label="medium"),
            contradiction_result=_make_contradiction_result(),
        )
        assert result["gap_detected"] is True
        assert result["gap_type"] == "partial"
        assert result["action"] == "search"

    @patch("querybridge.gap_detector._get_llm_client")
    def test_llm_full_coverage_with_contradictions(
        self,
        mock_client_fn: MagicMock,
    ) -> None:
        """LLM says full but contradictions are severe → contradictory."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "coverage": "full",
            "missing_aspects": "Nothing",
            "reasoning": "All aspects are covered.",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = detect_gap(
            query="How many parameters does the model have?",
            chunks=_make_chunks(),
            score_result=_make_score_result(confidence=0.45, label="medium"),
            contradiction_result=_make_contradiction_result(
                found=True, penalty=0.25,
            ),
        )
        assert result["gap_type"] == "contradictory"
        assert result["gap_detected"] is True

    @patch("querybridge.gap_detector._get_llm_client")
    def test_llm_failure_falls_back_to_rules(
        self,
        mock_client_fn: MagicMock,
    ) -> None:
        """When the LLM call raises, fall back to rule-based."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "API timeout"
        )
        mock_client_fn.return_value = mock_client

        result = detect_gap(
            query="What is self-attention?",
            chunks=_make_chunks(),
            score_result=_make_score_result(confidence=0.55, label="medium"),
            contradiction_result=_make_contradiction_result(),
        )
        # Rule-based with 0.55 confidence → partial
        assert result["gap_detected"] is True
        assert result["gap_type"] == "partial"
        assert result["action"] == "search"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for type checking in detect_gap."""

    def test_chunks_not_list_raises(self) -> None:
        with pytest.raises(TypeError, match="list of chunk dicts"):
            detect_gap(
                query="test",
                chunks="not a list",  # type: ignore
                score_result=_make_score_result(),
                contradiction_result=_make_contradiction_result(),
            )

    def test_score_result_not_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="score_result dict"):
            detect_gap(
                query="test",
                chunks=_make_chunks(),
                score_result="bad",  # type: ignore
                contradiction_result=_make_contradiction_result(),
            )

    def test_contradiction_result_not_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="contradiction_result dict"):
            detect_gap(
                query="test",
                chunks=_make_chunks(),
                score_result=_make_score_result(),
                contradiction_result=42,  # type: ignore
            )
