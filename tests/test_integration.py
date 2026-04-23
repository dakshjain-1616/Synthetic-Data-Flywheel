"""Integration test - End-to-end dry run with mocked services."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from synthetic_data_flywheel.engine import create_engine
from synthetic_data_flywheel.models import SyntheticPair, QualityScores, JudgmentResult


class TestIntegration:
    """Integration tests for the full flywheel pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_flywheel_cycle(self, tmp_path):
        """Test a complete flywheel cycle with mocked services."""
        
        # Mock generator response
        mock_gen_response = MagicMock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            "choices": [{"message": {"content": '{"instruction": "Test instruction", "input": "", "output": "Test output", "category": "test"}'}}]
        }
        
        # Mock judge response
        mock_judge_response = MagicMock()
        mock_judge_response.status_code = 200
        mock_judge_response.json.return_value = {
            "response": '{"passed": true, "coherence": 8.0, "accuracy": 8.0, "helpfulness": 8.0, "overall": 8.0, "reasoning": "Good quality"}'
        }
        
        with patch("httpx.AsyncClient") as mock_async_client, \
             patch("httpx.Client") as mock_sync_client, \
             patch("synthetic_data_flywheel.config.get_settings") as mock_settings:
            
            # Setup async client mock
            mock_async = AsyncMock()
            mock_async.__aenter__ = AsyncMock(return_value=mock_async)
            mock_async.__aexit__ = AsyncMock(return_value=None)
            mock_async.post = AsyncMock(return_value=mock_gen_response)
            mock_async_client.return_value = mock_async
            
            # Setup sync client mock
            mock_sync = MagicMock()
            mock_sync.__enter__ = MagicMock(return_value=mock_sync)
            mock_sync.__exit__ = MagicMock(return_value=None)
            mock_sync.post = MagicMock(return_value=mock_judge_response)
            mock_sync_client.return_value = mock_sync
            
            # Setup settings mock
            mock_settings.return_value = MagicMock(
                data_dir=str(tmp_path / "data"),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                report_output_dir=str(tmp_path / "reports"),
                notebook_output_dir=str(tmp_path / "notebooks"),
                openrouter_api_key="test-key",
                openrouter_model="test-model",
                ollama_base_url="http://localhost:11434",
                ollama_model="test-model",
                quality_min_score=7.0,
                max_cycles=1,
                huggingface_token=None,
            )
            
            # Create engine and run
            engine = create_engine(
                seeds=["Test seed 1", "Test seed 2"],
                checkpoint_dir=str(tmp_path / "checkpoints"),
                max_cycles=1,
            )
            
            cycles = await engine.run_full_loop()
            
            # Verify results
            assert len(cycles) == 1
            assert cycles[0].status == "completed"
            assert cycles[0].cycle_id == 1
            
            # Verify checkpoint was saved
            checkpoints = list((tmp_path / "checkpoints").glob("*.json"))
            assert len(checkpoints) == 1
            
            # Verify summary
            summary = engine.get_summary()
            assert summary["total_cycles"] == 1
            assert summary["total_passed_pairs"] > 0
    
    def test_checkpoint_save_and_load(self, tmp_path):
        """Test checkpoint save and load functionality."""
        
        with patch("synthetic_data_flywheel.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                data_dir=str(tmp_path / "data"),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                report_output_dir=str(tmp_path / "reports"),
                notebook_output_dir=str(tmp_path / "notebooks"),
                max_cycles=5,
            )
            
            engine = create_engine(
                seeds=["Test"],
                checkpoint_dir=str(tmp_path / "checkpoints"),
            )
            
            # Manually add a cycle
            from synthetic_data_flywheel.models import CycleState
            cycle = CycleState(
                cycle_id=1,
                status="completed",
                seeds=["Test"],
                generated_pairs=[],
                judgments=[],
                passed_pairs=[],
                passed_judgments=[],
                eval_metrics={"pass_rate": 0.8},
            )
            engine.cycles.append(cycle)
            engine.current_cycle = 1
            
            # Save checkpoint
            checkpoint_path = engine._save_checkpoint()
            assert checkpoint_path.exists()
            
            # Create new engine and load
            engine2 = create_engine(seeds=[])
            loaded = engine2.load_checkpoint(str(checkpoint_path))
            
            assert loaded is True
            assert len(engine2.cycles) == 1
            assert engine2.cycles[0].cycle_id == 1
    
    def test_feedback_extraction(self, tmp_path):
        """Test failure seed extraction for feedback loop."""
        
        with patch("synthetic_data_flywheel.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                data_dir=str(tmp_path / "data"),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                report_output_dir=str(tmp_path / "reports"),
                notebook_output_dir=str(tmp_path / "notebooks"),
                max_cycles=5,
            )
            
            engine = create_engine(seeds=["Initial seed"])
            
            # Create a cycle with some failed judgments
            from synthetic_data_flywheel.models import CycleState
            from uuid import uuid4
            
            pair_id = str(uuid4())
            cycle = CycleState(
                cycle_id=1,
                status="completed",
                seeds=["Initial seed"],
                generated_pairs=[{
                    "id": pair_id,
                    "instruction": "Failed instruction",
                    "input": "",
                    "output": "Failed output",
                    "created_at": "2024-01-01T00:00:00",
                }],
                judgments=[{
                    "pair_id": pair_id,
                    "scores": {"coherence": 4.0, "accuracy": 4.0, "helpfulness": 4.0, "overall": 4.0},
                    "passed": False,
                    "judge_model": "test",
                    "judgment_reasoning": "Poor quality",
                    "judged_at": "2024-01-01T00:00:00",
                }],
                passed_pairs=[],
                passed_judgments=[],
            )
            engine.cycles.append(cycle)
            
            # Extract failure seeds
            failure_seeds = engine._extract_failure_seeds(1)
            
            assert len(failure_seeds) > 0
            assert "Failed instruction" in failure_seeds
