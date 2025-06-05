"""
Pytest configuration and fixtures.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.utils.config import AppConfig
from src.utils.security import SecurityConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return AppConfig(
        debug=True,
        environment="testing",
        projects_root=temp_dir / "projects",
        cache_root=temp_dir / "cache",
        logs_root=temp_dir / "logs",
        security=SecurityConfig(
            secret_key="test_secret_key_32_chars_long_123",
            api_rate_limit=10,
            max_file_size=1024
        )
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = Mock()
    mock.generate.return_value = {"response": "test response"}
    return mock