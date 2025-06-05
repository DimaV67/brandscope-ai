"""
Manual test script for Stage 1 orchestration.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.project_manager import ProjectManager
from src.models.brand import BrandContext, ProductInfo, PriceTierEnum, CompetitiveContext
from src.stage1.prompt_generator import Stage1Generator
from src.utils.config import AppConfig
from src.utils.security import SecurityConfig


def test_stage1_manually():
    """Manual test of Stage 1 orchestration."""
    
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
            brand_name="SonoTest",
            products=[
                ProductInfo(
                    name="SonoTest One",
                    product_type="smart_speaker",
                    price_tier=PriceTierEnum.PREMIUM,
                    key_features=["wireless", "alexa", "premium_sound"]
                )
            ],
            competitive_context=CompetitiveContext(
                primary_competitors=["Apple HomePod", "Amazon Echo"]
            )
        )
        
        with patch('src.utils.config.get_config', return_value=test_config):
            with patch('src.core.project_manager.get_config', return_value=test_config):
                # Create project
                manager = ProjectManager()
                project = manager.create_project(
                    brand_context=brand_context,
                    category="speakers"
                )
                
                print(f"✅ Project created: {project.project_id}")
                
                # Run Stage 1
                generator = Stage1Generator(project)
                result = generator.execute_full_pipeline()
                
                print(f"✅ Stage 1 completed: {result['status']}")
                print(f"📁 Artifacts: {len(result['artifacts'])}")
                print(f"🎯 Archetypes: {len(result['execution_package']['customer_archetypes'])}")
                print(f"❓ Queries: {len(result['execution_package']['execution_queries'])}")
                
                # Check files were created
                stage1_dir = project.get_file_path("stage1_outputs")
                json_files = list(stage1_dir.glob("*.json"))
                print(f"📄 JSON files created: {len(json_files)}")
                
                guide_path = project.get_file_path("stage2_execution/manual_execution_guide.md")
                if guide_path.exists():
                    print(f"📋 Execution guide created: {guide_path.stat().st_size} bytes")
                
                return True


if __name__ == "__main__":
    success = test_stage1_manually()
    if success:
        print("🎉 Manual test passed!")
    else:
        print("❌ Manual test failed!")