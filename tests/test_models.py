"""Unit tests for data models."""

import pytest
from datetime import datetime
from uuid import UUID

from synthetic_data_flywheel.models import (
    SyntheticPair,
    QualityScores,
    JudgmentResult,
    CycleState,
    FlywheelConfig,
)


class TestSyntheticPair:
    """Tests for SyntheticPair model."""
    
    def test_create_synthetic_pair(self):
        """Test creating a synthetic pair."""
        pair = SyntheticPair(
            instruction="What is 2+2?",
            input="",
            output="4",
            category="math",
        )
        assert pair.instruction == "What is 2+2?"
        assert pair.output == "4"
        assert pair.category == "math"
        assert isinstance(pair.id, UUID)
    
    def test_synthetic_pair_to_dict(self):
        """Test converting pair to dictionary."""
        pair = SyntheticPair(
            instruction="Test",
            input="",
            output="Result",
        )
        d = pair.to_dict()
        assert d["instruction"] == "Test"
        assert d["output"] == "Result"
        assert "id" in d
        assert "created_at" in d
    
    def test_synthetic_pair_from_dict(self):
        """Test creating pair from dictionary."""
        data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "instruction": "Test",
            "input": "",
            "output": "Result",
            "category": "test",
            "created_at": "2024-01-01T00:00:00",
        }
        pair = SyntheticPair.from_dict(data)
        assert pair.instruction == "Test"
        assert pair.output == "Result"


class TestQualityScores:
    """Tests for QualityScores model."""
    
    def test_valid_scores(self):
        """Test valid quality scores."""
        scores = QualityScores(
            coherence=8.5,
            accuracy=9.0,
            helpfulness=7.5,
            overall=8.3,
        )
        assert scores.coherence == 8.5
        assert scores.accuracy == 9.0
    
    def test_invalid_score_too_high(self):
        """Test that scores above 10 are rejected."""
        with pytest.raises(ValueError):
            QualityScores(coherence=11.0)
    
    def test_invalid_score_negative(self):
        """Test that negative scores are rejected."""
        with pytest.raises(ValueError):
            QualityScores(coherence=-1.0)


class TestJudgmentResult:
    """Tests for JudgmentResult model."""
    
    def test_judgment_result_creation(self):
        """Test creating a judgment result."""
        scores = QualityScores(coherence=8.0, accuracy=9.0, helpfulness=8.0, overall=8.3)
        result = JudgmentResult(
            pair_id="test-id",
            scores=scores,
            passed=True,
            judge_model="test-model",
        )
        assert result.passed is True
        assert result.quality_tier == "high"
    
    def test_quality_tier_calculation(self):
        """Test quality tier calculation."""
        # High tier
        scores_high = QualityScores(coherence=8.0, accuracy=8.0, helpfulness=8.0, overall=8.0)
        result_high = JudgmentResult(pair_id="1", scores=scores_high, passed=True)
        assert result_high.quality_tier == "high"
        
        # Medium tier
        scores_med = QualityScores(coherence=6.0, accuracy=6.0, helpfulness=6.0, overall=6.0)
        result_med = JudgmentResult(pair_id="2", scores=scores_med, passed=True)
        assert result_med.quality_tier == "medium"
        
        # Low tier
        scores_low = QualityScores(coherence=4.0, accuracy=4.0, helpfulness=4.0, overall=4.0)
        result_low = JudgmentResult(pair_id="3", scores=scores_low, passed=False)
        assert result_low.quality_tier == "low"


class TestCycleState:
    """Tests for CycleState model."""
    
    def test_cycle_state_creation(self):
        """Test creating a cycle state."""
        cycle = CycleState(
            cycle_id=1,
            status="completed",
            seeds=["seed1", "seed2"],
            generated_pairs=[],
            judgments=[],
            passed_pairs=[],
            passed_judgments=[],
        )
        assert cycle.cycle_id == 1
        assert cycle.status == "completed"
    
    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        cycle = CycleState(
            cycle_id=1,
            status="completed",
            seeds=[],
            generated_pairs=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
            judgments=[],
            passed_pairs=[{"id": "1"}],
            passed_judgments=[],
        )
        assert cycle.pass_rate == 1/3
    
    def test_avg_quality_score(self):
        """Test average quality score calculation."""
        cycle = CycleState(
            cycle_id=1,
            status="completed",
            seeds=[],
            generated_pairs=[],
            judgments=[
                {"pair_id": "1", "scores": {"overall": 8.0}, "passed": True, "judge_model": "m", "judgment_reasoning": "r", "judged_at": "2024-01-01T00:00:00"},
                {"pair_id": "2", "scores": {"overall": 6.0}, "passed": True, "judge_model": "m", "judgment_reasoning": "r", "judged_at": "2024-01-01T00:00:00"},
            ],
            passed_pairs=[],
            passed_judgments=[],
        )
        assert cycle.avg_quality_score == 7.0


class TestFlywheelConfig:
    """Tests for FlywheelConfig model."""
    
    def test_config_creation(self):
        """Test creating flywheel config."""
        config = FlywheelConfig(
            openrouter_api_key="test-key",
            openrouter_model="test-model",
            ollama_model="test-judge",
        )
        assert config.openrouter_api_key == "test-key"
        assert config.openrouter_model == "test-model"
        assert config.ollama_model == "test-judge"
