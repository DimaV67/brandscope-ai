"""
Test data models.
"""
import pytest
from datetime import datetime

from src.models.project import ProjectMetadata, ProjectConfig, StageStatus, StageStatusEnum


class TestProjectMetadata:
    """Test ProjectMetadata model."""
    
    def test_valid_project_metadata(self):
        """Test creating valid project metadata."""
        metadata = ProjectMetadata(
            project_id="test_project_123",
            display_name="Test Project",
            brand="TestBrand",
            category="speakers"
        )
        
        assert metadata.project_id == "test_project_123"
        assert metadata.brand == "TestBrand"
        assert metadata.category == "speakers"
        assert isinstance(metadata.created_date, datetime)
    
    def test_project_id_validation(self):
        """Test project ID validation."""
        # Test path traversal prevention
        with pytest.raises(ValueError, match="path traversal"):
            ProjectMetadata(
                project_id="../dangerous_path",
                display_name="Test",
                brand="Test",
                category="test"
            )
        
        # Test invalid characters
        with pytest.raises(ValueError, match="alphanumeric"):
            ProjectMetadata(
                project_id="test<script>",
                display_name="Test",
                brand="Test", 
                category="test"
            )
    
    def test_brand_validation(self):
        """Test brand name validation."""
        with pytest.raises(ValueError, match="HTML/script"):
            ProjectMetadata(
                project_id="test_project",
                display_name="Test",
                brand="Brand<script>alert('xss')</script>",
                category="test"
            )


class TestStageStatus:
    """Test StageStatus model."""
    
    def test_default_stage_status(self):
        """Test default stage status."""
        status = StageStatus()
        
        assert status.status == StageStatusEnum.PENDING
        assert status.started_date is None
        assert status.completion_date is None
        assert status.outputs == []
        assert status.retry_count == 0
    
    def test_output_validation(self):
        """Test output path validation."""
        status = StageStatus(
            outputs=["valid_file.json", "../dangerous_path", "/absolute/path", "normal.txt"]
        )
        
        # Should filter out dangerous paths
        assert status.outputs == ["valid_file.json", "normal.txt"]


class TestProjectConfig:
    """Test ProjectConfig model."""
    
    def test_valid_project_config(self):
        """Test creating valid project config."""
        metadata = ProjectMetadata(
            project_id="test_project",
            display_name="Test Project",
            brand="TestBrand",
            category="speakers"
        )
        
        config = ProjectConfig(project_metadata=metadata)
        
        assert config.project_metadata.project_id == "test_project"
        assert "stage1" in config.stage_status
        assert "stage2" in config.stage_status
        assert "stage3" in config.stage_status
    
    def test_stage_validation(self):
        """Test stage status validation."""
        metadata = ProjectMetadata(
            project_id="test_project",
            display_name="Test Project",
            brand="TestBrand",
            category="speakers"
        )
        
        # Test missing stages
        with pytest.raises(ValueError, match="Missing required stages"):
            ProjectConfig(
                project_metadata=metadata,
                stage_status={"stage1": StageStatus()}  # Missing stage2, stage3
            )


if __name__ == "__main__":
    pytest.main([__file__])