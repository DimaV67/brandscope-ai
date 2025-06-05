"""
Simple error handling test.
"""
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from src.core.project_manager import ProjectManager
from src.models.brand import BrandContext, ProductInfo, PriceTierEnum
from src.models.project import StageStatusEnum
from src.stage1.prompt_generator import Stage1Generator
from src.utils.config import AppConfig
from src.utils.security import SecurityConfig


def test_simple_error():
    """Test error handling with minimal setup."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        
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
                
                # Check initial status
                initial_status = project.config.stage_status['stage1'].status
                print(f"📊 Initial status: {initial_status}")
                
                # Remove the brand context file to cause error
                brand_file = project.get_file_path("inputs/brand_context.json")
                brand_file.unlink()
                print(f"🗑️ Removed brand context file")
                
                # Create generator and run (should fail)
                generator = Stage1Generator(project)
                
                try:
                    result = generator.execute_full_pipeline()
                    print("❌ Pipeline should have failed but didn't")
                    return False
                except Exception as e:
                    print(f"✅ Pipeline failed as expected: {type(e).__name__}")
                
                # Check status after failure
                # Reload the project config from disk
                project._load_config()
                final_status = project.config.stage_status['stage1']
                
                print(f"📊 Final status: {final_status.status}")
                print(f"❌ Error message: {final_status.error_message}")
                print(f"🔄 Retry count: {final_status.retry_count}")
                
                # Check if status was properly updated
                success = (
                    final_status.status == StageStatusEnum.FAILED and
                    final_status.error_message is not None and
                    len(final_status.error_message) > 0
                )
                
                if success:
                    print("✅ Status correctly updated to FAILED")
                else:
                    print("❌ Status was not properly updated")
                    
                    # Debug: show the actual config file
                    config_file = project.config_path
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        print(f"📄 Config file stage1 status: {config_data['stage_status']['stage1']}")
                
                return success


if __name__ == "__main__":
    success = test_simple_error()
    print("🎉 Success!" if success else "💥 Failed!")