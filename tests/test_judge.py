"""Unit tests for judge module."""

import pytest
from unittest.mock import MagicMock, patch

from synthetic_data_flywheel.judge import (
    OllamaClient,
    QualityJudge,
    create_judge,
)
from synthetic_data_flywheel.models import SyntheticPair, QualityScores, JudgmentResult


class TestOllamaClient:
    """Tests for OllamaClient."""
    
    def test_generate_method(self):
        """Test generate method with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test judgment"}
        
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client.post = MagicMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            client = OllamaClient(base_url="http://localhost:11434")
            with client:
                result = client.generate("Test prompt", "test-model")
        
        assert result == "Test judgment"
    
    def test_check_health(self):
        """Test health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client.get = MagicMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            client = OllamaClient()
            with client:
                is_healthy = client.check_health()
        
        assert is_healthy is True


class TestQualityJudge:
    """Tests for QualityJudge."""
    
    def test_parse_judgment_valid_json(self):
        """Test parsing valid judgment JSON."""
        judge = QualityJudge()
        response = '{"passed": true, "coherence": 8.5, "accuracy": 9.0, "helpfulness": 8.0, "overall": 8.5, "reasoning": "Good"}'
        
        result = judge._parse_judgment(response)
        
        assert result["passed"] is True
        assert result["coherence"] == 8.5
        assert result["accuracy"] == 9.0
    
    def test_parse_judgment_markdown_json(self):
        """Test parsing JSON in markdown code block."""
        judge = QualityJudge()
        response = '```json\n{"passed": true, "coherence": 8.0, "accuracy": 8.0, "helpfulness": 8.0, "overall": 8.0, "reasoning": "Test"}\n```'
        
        result = judge._parse_judgment(response)
        
        assert result["passed"] is True
        assert result["coherence"] == 8.0
    
    def test_parse_judgment_invalid_json(self):
        """Test parsing invalid JSON returns fallback."""
        judge = QualityJudge()
        response = "This is not valid JSON"
        
        result = judge._parse_judgment(response)
        
        assert result["passed"] is False
        assert result["overall"] == 5.0
    
    def test_judge_single_pair(self):
        """Test judging a single pair."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"passed": true, "coherence": 8.0, "accuracy": 8.0, "helpfulness": 8.0, "overall": 8.0, "reasoning": "Good quality"}'
        }
        
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client.post = MagicMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            judge = QualityJudge()
            pair = SyntheticPair(instruction="Test", input="", output="Result")
            
            with judge:
                result = judge.judge(pair)
        
        assert isinstance(result, JudgmentResult)
        assert result.passed is True
        assert result.scores.overall == 8.0
    
    def test_judge_batch(self):
        """Test batch judging."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"passed": true, "coherence": 8.0, "accuracy": 8.0, "helpfulness": 8.0, "overall": 8.0, "reasoning": "Good"}'
        }
        
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client.post = MagicMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            judge = QualityJudge()
            pairs = [
                SyntheticPair(instruction="Test1", input="", output="Result1"),
                SyntheticPair(instruction="Test2", input="", output="Result2"),
            ]
            
            with judge:
                results = judge.judge_batch(pairs)
        
        assert len(results) == 2
        assert all(isinstance(r, JudgmentResult) for r in results)
    
    def test_filter_pairs(self):
        """Test filtering pairs based on judgments."""
        judge = QualityJudge()
        
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
                scores=QualityScores(coherence=4.0, accuracy=4.0, helpfulness=4.0, overall=4.0),
                passed=False,
                judge_model="test",
            ),
        ]
        
        passed, failed = judge.filter_pairs(pairs, judgments)
        
        assert len(passed) == 1
        assert len(failed) == 1
        assert passed[0].instruction == "Test1"
        assert failed[0].instruction == "Test2"


class TestCreateJudge:
    """Tests for create_judge factory."""
    
    def test_create_judge(self):
        """Test factory creates judge."""
        judge = create_judge()
        assert isinstance(judge, QualityJudge)
