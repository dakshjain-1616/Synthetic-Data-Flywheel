"""Unit tests for generator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synthetic_data_flywheel.generator import (
    PromptTemplate,
    OpenRouterClient,
    create_generator,
)
from synthetic_data_flywheel.models import SyntheticPair


class TestPromptTemplate:
    """Tests for PromptTemplate."""
    
    def test_get_qa_template(self):
        """Test getting QA template."""
        template = PromptTemplate.get("QA")
        assert "{seed}" in template
        assert "question" in template.lower()
    
    def test_get_instruction_template(self):
        """Test getting instruction template."""
        template = PromptTemplate.get("INSTRUCTION")
        assert "{seed}" in template
        assert "instruction" in template.lower()
    
    def test_get_reasoning_template(self):
        """Test getting reasoning template."""
        template = PromptTemplate.get("REASONING")
        assert "{seed}" in template
        assert "reasoning" in template.lower()
    
    def test_get_creative_template(self):
        """Test getting creative template."""
        template = PromptTemplate.get("CREATIVE")
        assert "{seed}" in template
    
    def test_get_unknown_template(self):
        """Test getting unknown template defaults to instruction."""
        template = PromptTemplate.get("UNKNOWN")
        assert "{seed}" in template


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""
    
    @pytest.mark.asyncio
    async def test_generate_method(self):
        """Test generate method with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            client = OpenRouterClient(api_key="test-key")
            async with client:
                result = await client.generate("Test prompt")
        
        assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_generate_synthetic_pair(self):
        """Test generating synthetic pair."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"instruction": "Test", "input": "", "output": "Result"}'}}]
        }
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            client = OpenRouterClient(api_key="test-key")
            async with client:
                pair = await client.generate_synthetic_pair("Seed", "INSTRUCTION")
        
        assert isinstance(pair, SyntheticPair)
        assert pair.instruction == "Test"
        assert pair.output == "Result"
    
    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Test batch generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"instruction": "Test", "input": "", "output": "Result"}'}}]
        }
        
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            client = OpenRouterClient(api_key="test-key")
            async with client:
                pairs = await client.generate_batch(["seed1", "seed2"], "INSTRUCTION")
        
        assert len(pairs) == 2
        assert all(isinstance(p, SyntheticPair) for p in pairs)


class TestCreateGenerator:
    """Tests for create_generator factory."""
    
    def test_create_generator(self):
        """Test factory creates generator."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            generator = create_generator()
            assert isinstance(generator, OpenRouterClient)
