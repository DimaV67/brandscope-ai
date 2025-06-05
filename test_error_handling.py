"""
Test error handling specifically.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.project_manager import ProjectManager
from src.models.brand import BrandContext, ProductInfo, PriceTierEnum
from src.models.project import StageStatusEnum
from src.stage1.prompt_generator import Stage1Generator
from src.utils.config import AppConfig
from src.utils.security import SecurityConfig


def test_error_handling():
    """Test error handling in Stage 1."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        
        # Create test config
        test_config = AppConfig(
            debug=True,
            environment="testing",
            projects_root=temp_path / "projects",
            cache_root=temp_path / "cache", 
            logs_root=temp_path / "logs",
            security=SecurityConfig(
                secret_key="test_secret_key_32_chars_long_123",
                api_rate_limit=10,
                max_file_size=1024 * 1024
            )
        )
        
        # Create brand context
        brand_context = BrandContext(
            brand_name="ErrorTest",
            products=[
                ProductInfo(
                    name="Test Product",
                    product_type="test",
                    price_tier=PriceTierEnum.MIDRANGE
                )
            ]
        )
        
        with patch('src.utils.config.get_config', return_value=test_config):
            with patch('src.core.project_manager.get_config', return_value=test_config):
                # Create project
                manager = ProjectManager()
                project = manager.create_project(
                    brand_context=brand_context,
                    category="test_category"
                )
                
                print(f"✅ Project created: {project.project_id}")
                
                # Remove required file to cause error
                brand_context_path = project.get_file_path("inputs/brand_context.json")
                brand_context_path.unlink()
                print(f"🗑️ Removed: {brand_context_path}")
                
                # Try to run Stage 1 - should fail
                generator = Stage1Generator(project)
                
                try:
                    result = generator.execute_full_pipeline()
                    print("❌ Expected error but pipeline succeeded")
                    return False
                except Exception as e:
                    print(f"✅ Expected error occurred: {str(e)}")
                
                # Check if status was updated
                project._load_config()  # Reload to get updated status
                stage1_status = project.config.stage_status['stage1']
                
                print(f"📊 Stage 1 status: {stage1_status.status}")
                print(f"❌ Error message: {stage1_status.error_message}")
                
                if stage1_status.status == StageStatusEnum.FAILED:
                    print("✅ Status correctly updated to FAILED")
                    return True
                else:
                    print(f"❌ Status should be FAILED but is {stage1_status.status}")
                    return False


if __name__ == "__main__":
    success = test_error_handling()
    if success:
        print("🎉 Error handling test passed!")
    else:
        print("💥 Error handling test failed!")