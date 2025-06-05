"""
Simple test script for Stage 1 orchestration.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from core.project_manager import ProjectManager
    from models.brand import BrandContext, ProductInfo, PriceTierEnum, CompetitiveContext
    from stage1.prompt_generator import Stage1Generator
    from utils.config import AppConfig
    from utils.security import SecurityConfig
    from utils.logging import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_stage1_simple():
    """Simple test of Stage 1 orchestration."""
    
    # Setup logging
    setup_logging()
    
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
            brand_name="TestSpeakers",
            products=[
                ProductInfo(
                    name="TestSpeaker Pro",
                    product_type="smart_speaker",
                    price_tier=PriceTierEnum.PREMIUM,
                    key_features=["wireless", "voice_control", "premium_sound"]
                )
            ],
            competitive_context=CompetitiveContext(
                primary_competitors=["Brand A", "Brand B"]
            )
        )
        
        print("🔄 Setting up test environment...")
        
        with patch('utils.config.get_config', return_value=test_config):
            with patch('core.project_manager.get_config', return_value=test_config):
                try:
                    # Create project
                    print("📁 Creating test project...")
                    manager = ProjectManager()
                    project = manager.create_project(
                        brand_context=brand_context,
                        category="speakers"
                    )
                    
                    print(f"✅ Project created: {project.project_id}")
                    
                    # Run Stage 1
                    print("🚀 Running Stage 1 pipeline...")
                    generator = Stage1Generator(project)
                    result = generator.execute_full_pipeline()
                    
                    print(f"✅ Stage 1 completed: {result['status']}")
                    
                    # Validate results
                    exec_package = result['execution_package']
                    print(f"📊 Results summary:")
                    print(f"   - Archetypes: {len(exec_package['customer_archetypes'])}")
                    print(f"   - Queries: {len(exec_package['execution_queries'])}")
                    print(f"   - Artifacts: {len(result['artifacts'])}")
                    
                    # Check file creation
                    stage1_dir = project.get_file_path("stage1_outputs")
                    json_files = list(stage1_dir.glob("*.json"))
                    print(f"   - JSON files: {len(json_files)}")
                    
                    guide_path = project.get_file_path("stage2_execution/manual_execution_guide.md")
                    guide_exists = guide_path.exists()
                    print(f"   - Execution guide: {'✅ Created' if guide_exists else '❌ Missing'}")
                    
                    # Show a sample query
                    if exec_package['execution_queries']:
                        sample_query = exec_package['execution_queries'][0]
                        print(f"\n📝 Sample query:")
                        print(f"   \"{sample_query['styled_query']}\"")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Error during execution: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False


if __name__ == "__main__":
    print("🎯 Testing Stage 1 Orchestration")
    print("=" * 40)
    
    success = test_stage1_simple()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)