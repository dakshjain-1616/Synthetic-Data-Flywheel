"""Unit tests for dataset manager module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from synthetic_data_flywheel.dataset_manager import (
    DatasetManager,
    create_dataset_manager,
)
from synthetic_data_flywheel.models import SyntheticPair


class TestDatasetManager:
    """Tests for DatasetManager."""
    
    def test_initialization(self, tmp_path):
        """Test dataset manager initialization."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            assert dm.data_dir == tmp_path
    
    def test_pairs_to_dicts(self, tmp_path):
        """Test converting pairs to dictionaries."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            pairs = [
                SyntheticPair(instruction="Test1", input="", output="Result1"),
                SyntheticPair(instruction="Test2", input="", output="Result2"),
            ]
            
            dicts = dm.pairs_to_dicts(pairs)
            
            assert len(dicts) == 2
            assert dicts[0]["instruction"] == "Test1"
            assert dicts[1]["instruction"] == "Test2"
    
    def test_dicts_to_pairs(self, tmp_path):
        """Test converting dictionaries to pairs."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            data = [
                {"id": "123e4567-e89b-12d3-a456-426614174000", "instruction": "Test1", "input": "", "output": "Result1", "created_at": "2024-01-01T00:00:00"},
                {"id": "123e4567-e89b-12d3-a456-426614174001", "instruction": "Test2", "input": "", "output": "Result2", "created_at": "2024-01-01T00:00:00"},
            ]
            
            pairs = dm.dicts_to_pairs(data)
            
            assert len(pairs) == 2
            assert pairs[0].instruction == "Test1"
            assert pairs[1].instruction == "Test2"
    
    def test_save_and_load_local(self, tmp_path):
        """Test saving and loading pairs locally."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            pairs = [
                SyntheticPair(instruction="Test1", input="", output="Result1"),
                SyntheticPair(instruction="Test2", input="", output="Result2"),
            ]
            
            # Save
            path = dm.save_local(pairs, filename="test.json")
            assert path.exists()
            
            # Load
            loaded = dm.load_local(filename="test.json")
            assert len(loaded) == 2
            assert loaded[0].instruction == "Test1"
    
    def test_save_and_load_with_split(self, tmp_path):
        """Test saving and loading with split."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            pairs = [SyntheticPair(instruction="Test", input="", output="Result")]
            
            path = dm.save_local(pairs, filename="test.json", split="train")
            assert path.exists()
            assert "train" in str(path)
            
            loaded = dm.load_local(filename="test.json", split="train")
            assert len(loaded) == 1
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file returns empty list."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            loaded = dm.load_local(filename="nonexistent.json")
            assert loaded == []
    
    def test_create_train_test_split(self, tmp_path):
        """Test train/test/val split."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            pairs = [SyntheticPair(instruction=f"Test{i}", input="", output=f"Result{i}") for i in range(10)]
            
            train, test, val = dm.create_train_test_split(pairs, test_size=0.2, val_size=0.1)
            
            assert len(train) == 7  # 70%
            assert len(test) == 2   # 20%
            assert len(val) == 1    # 10%
    
    def test_get_dataset_info(self, tmp_path):
        """Test getting dataset info."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            pairs = [
                SyntheticPair(instruction="Test1", input="", output="Result1", category="math"),
                SyntheticPair(instruction="Test2", input="", output="Result2", category="math"),
                SyntheticPair(instruction="Test3", input="", output="Result3", category="science"),
            ]
            
            info = dm.get_dataset_info(pairs)
            
            assert info["count"] == 3
            assert info["categories"]["math"] == 2
            assert info["categories"]["science"] == 1
    
    def test_get_dataset_info_empty(self, tmp_path):
        """Test getting info for empty dataset."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = DatasetManager()
            
            info = dm.get_dataset_info([])
            
            assert info["count"] == 0


class TestCreateDatasetManager:
    """Tests for create_dataset_manager factory."""
    
    def test_create_manager(self, tmp_path):
        """Test factory creates manager."""
        with patch("synthetic_data_flywheel.dataset_manager.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(data_dir=str(tmp_path))
            dm = create_dataset_manager()
            assert isinstance(dm, DatasetManager)
