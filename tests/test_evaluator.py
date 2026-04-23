"""Unit tests for evaluator module."""

import pytest
from synthetic_data_flywheel.evaluator import Evaluator, create_evaluator
from synthetic_data_flywheel.models import SyntheticPair, QualityScores, JudgmentResult


class TestEvaluator:
    """Tests for Evaluator."""
    
    def test_compute_exact_match(self):
        """Test exact match computation."""
        evaluator = Evaluator()
        
        predictions = ["A", "B", "C"]
        references = ["A", "B", "D"]
        
        accuracy = evaluator.compute_exact_match(predictions, references)
        
        assert accuracy == 2/3
    
    def test_compute_exact_match_empty(self):
        """Test exact match with empty lists."""
        evaluator = Evaluator()
        
        accuracy = evaluator.compute_exact_match([], [])
        
        assert accuracy == 0.0
    
    def test_compute_exact_match_mismatched_lengths(self):
        """Test exact match with mismatched lengths."""
        evaluator = Evaluator()
        
        accuracy = evaluator.compute_exact_match(["A"], ["A", "B"])
        
        assert accuracy == 0.0
    
    def test_compute_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        evaluator = Evaluator()
        
        predictions = ["Hello", "WORLD"]
        references = ["hello", "world"]
        
        accuracy = evaluator.compute_exact_match(predictions, references)
        
        assert accuracy == 1.0
    
    def test_evaluate_judgments(self):
        """Test evaluating judgments."""
        evaluator = Evaluator()
        
        judgments = [
            JudgmentResult(
                pair_id="1",
                scores=QualityScores(coherence=8.0, accuracy=8.0, helpfulness=8.0, overall=8.0),
                passed=True,
                judge_model="test",
            ),
            JudgmentResult(
                pair_id="2",
                scores=QualityScores(coherence=6.0, accuracy=6.0, helpfulness=6.0, overall=6.0),
                passed=False,
                judge_model="test",
            ),
            JudgmentResult(
                pair_id="3",
                scores=QualityScores(coherence=9.0, accuracy=9.0, helpfulness=9.0, overall=9.0),
                passed=True,
                judge_model="test",
            ),
        ]
        
        result = evaluator.evaluate_judgments(judgments)
        
        assert result["total"] == 3
        assert result["passed"] == 2
        assert result["failed"] == 1
        assert result["pass_rate"] == 2/3
        assert result["avg_coherence"] == (8.0 + 6.0 + 9.0) / 3
        assert result["avg_accuracy"] == (8.0 + 6.0 + 9.0) / 3
    
    def test_evaluate_judgments_empty(self):
        """Test evaluating empty judgments."""
        evaluator = Evaluator()
        
        result = evaluator.evaluate_judgments([])
        
        assert result["total"] == 0
        assert result["pass_rate"] == 0.0
    
    def test_evaluate_dataset(self):
        """Test evaluating dataset."""
        evaluator = Evaluator()
        
        pairs = [
            SyntheticPair(instruction="Test1", input="", output="Result1", category="math"),
            SyntheticPair(instruction="Test2", input="", output="Result2", category="math"),
            SyntheticPair(instruction="Test3", input="", output="Result3", category="science"),
        ]
        
        result = evaluator.evaluate_dataset(pairs)
        
        assert result["total_samples"] == 3
        assert result["categories"]["math"] == 2
        assert result["categories"]["science"] == 1
    
    def test_evaluate_dataset_empty(self):
        """Test evaluating empty dataset."""
        evaluator = Evaluator()
        
        result = evaluator.evaluate_dataset([])
        
        assert "error" in result
    
    def test_evaluate_dataset_with_judgments(self):
        """Test evaluating dataset with judgments."""
        evaluator = Evaluator()
        
        pairs = [
            SyntheticPair(instruction="Test1", input="", output="Result1"),
            SyntheticPair(instruction="Test2", input="", output="Result2"),
        ]
        
        judgments = [
            JudgmentResult(
                pair_id=str(pairs[0].id),
                scores=QualityScores(coherence=8.0, accuracy=8.0, helpfulness=8.0, overall=8.0),
                passed=True,
                judge_model="test",
            ),
            JudgmentResult(
                pair_id=str(pairs[1].id),
                scores=QualityScores(coherence=6.0, accuracy=6.0, helpfulness=6.0, overall=6.0),
                passed=False,
                judge_model="test",
            ),
        ]
        
        result = evaluator.evaluate_dataset(pairs, judgments)
        
        assert result["total_samples"] == 2
        assert "judgments" in result
        assert result["judgments"]["total"] == 2


class TestCreateEvaluator:
    """Tests for create_evaluator factory."""
    
    def test_create_evaluator(self):
        """Test factory creates evaluator."""
        evaluator = create_evaluator()
        assert isinstance(evaluator, Evaluator)
