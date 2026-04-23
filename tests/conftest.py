"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    return tmp_path / "data"


@pytest.fixture
def test_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory for tests."""
    return tmp_path / "checkpoints"


@pytest.fixture
def test_report_dir(tmp_path):
    """Create a temporary report directory for tests."""
    return tmp_path / "reports"
