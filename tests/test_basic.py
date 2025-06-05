"""
Basic functionality tests.
"""
import pytest


def test_basic_import():
    """Test basic imports work."""
    try:
        import src
        assert True
    except ImportError:
        pytest.fail("Cannot import src package")


def test_models_import():
    """Test model imports work."""
    try:
        from src.models.project import ProjectMetadata
        assert True
    except ImportError:
        pytest.fail("Cannot import models")


if __name__ == "__main__":
    pytest.main([__file__])