"""
Integration test for Stage 1 orchestration.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.project_manager import ProjectManager, BrandAuditProject
from src.models.brand import BrandContext, ProductInfo, PriceTierEnum, CompetitiveContext
from src.models.project import StageStatusEnum
from src.stage1.prompt_generator import Stage1Generator
from src.utils.config import AppConfig
from src.utils.security import SecurityConfig


class TestStage1Orchestration:
    """Test Stage 1 complete orchestration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
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
                max_file_size=1024 * 1024
            )
        )
    
    @pytest.fixture
    def brand_context(self):
        """Create test brand context."""
        return BrandContext(
            brand_name="TestBrand",
            products=[
                ProductInfo(
                    name="TestProduct One",
                    product_type="premium_speaker",
                    price_tier=PriceTierEnum.PREMIUM,
                    price_range="$200-$400",
                    key_features=["wireless", "smart_assistant", "premium_audio"]
                ),
                ProductInfo(
                    name="TestProduct Two", 
                    product_type="portable_speaker",
                    price_tier=PriceTierEnum.MIDRANGE,
                    price_range="$100-$200",
                    key_features=["portable", "waterproof", "long_battery"]
                )
            ],
            competitive_context=CompetitiveContext(
                primary_competitors=["Competitor A", "Competitor B", "Competitor C"]
            ),
            brand_positioning="Premium audio with smart features",
            target_markets=["urban_professionals", "tech_enthusiasts"]
        )
    
    @pytest.fixture
    def test_project(self, test_config, brand_context, temp_dir):
        """Create test project."""
        with patch('src.utils.config.get_config', return_value=test_config):
            with patch('src.core.project_manager.get_config', return_value=test_config):
                # Create project manager
                manager = ProjectManager()
                
                # Create project
                project = manager.create_project(
                    brand_context=brand_context,
                    category="speakers",
                    customer_narrative="Test customer looking for premium speakers"
                )
                
                return project
    
    def test_stage1_full_pipeline_success(self, test_project, test_config):
        """Test successful Stage 1 pipeline execution."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            # Create Stage 1 generator
            generator = Stage1Generator(test_project)
            
            # Execute pipeline
            result = generator.execute_full_pipeline()
            
            # Validate result structure
            assert result['status'] == 'success'
            assert 'artifacts' in result
            assert 'execution_package' in result
            assert result['next_steps'] == 'stage2_manual_execution'
            
            # Check artifacts were created
            artifacts = result['artifacts']
            expected_artifacts = [
                'category_intelligence',
                'archetypes', 
                'queries',
                'execution_package'
            ]
            
            for artifact in expected_artifacts:
                assert artifact in artifacts
                artifact_path = Path(artifacts[artifact])
                assert artifact_path.exists()
                assert artifact_path.stat().st_size > 0
            
            # Validate execution package structure
            exec_package = result['execution_package']
            assert 'metadata' in exec_package
            assert 'customer_archetypes' in exec_package
            assert 'execution_queries' in exec_package
            assert 'prompts' in exec_package
            
            # Check metadata
            metadata = exec_package['metadata']
            assert metadata['project_id'] == test_project.project_id
            assert metadata['brand'] == "TestBrand"
            assert metadata['category'] == "speakers"
            assert 'generation_timestamp' in metadata
            assert 'correlation_id' in metadata
            
            # Check archetypes
            archetypes = exec_package['customer_archetypes']
            assert len(archetypes) > 0
            assert all('name' in arch for arch in archetypes)
            assert all('attributes' in arch for arch in archetypes)
            
            # Check queries
            queries = exec_package['execution_queries']
            assert len(queries) > 0
            assert all('styled_query' in query for query in queries)
            assert all('category' in query for query in queries)
            
            # Check prompts
            prompts = exec_package['prompts']
            assert 'natural_ai' in prompts
            assert 'controlled_ai' in prompts
            assert len(prompts['natural_ai']) > 50  # Should be substantial
            assert len(prompts['controlled_ai']) > 50
            
            # Check project status was updated
            status = test_project.get_status()
            assert status['stage1_complete'] is True
    
    def test_stage1_creates_execution_guide(self, test_project, test_config):
        """Test that Stage 1 creates execution guide."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            generator = Stage1Generator(test_project)
            result = generator.execute_full_pipeline()
            
            # Check execution guide was created
            guide_path = test_project.get_file_path("stage2_execution/manual_execution_guide.md")
            assert guide_path.exists()
            
            # Read and validate guide content
            with open(guide_path, 'r', encoding='utf-8') as f:
                guide_content = f.read()
            
            # Check for required sections
            assert "# TestBrand speakers Brand Audit" in guide_content
            assert "## Customer Archetypes" in guide_content
            assert "## Execution Instructions" in guide_content
            assert "### Time Estimate" in guide_content
            assert "### Platform Testing Order" in guide_content
            assert "### Query Execution Priority" in guide_content
            assert "## Quality Checklist" in guide_content
            
            # Check that specific content is included
            assert "TestBrand" in guide_content
            assert "speakers" in guide_content
            assert "Claude" in guide_content
            assert "ChatGPT" in guide_content
    
    def test_stage1_file_structure(self, test_project, test_config):
        """Test that Stage 1 creates proper file structure."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            generator = Stage1Generator(test_project)
            generator.execute_full_pipeline()
            
            # Check stage1_outputs directory
            stage1_dir = test_project.get_file_path("stage1_outputs")
            assert stage1_dir.exists()
            assert stage1_dir.is_dir()
            
            # Check expected files exist
            json_files = list(stage1_dir.glob("*.json"))
            assert len(json_files) >= 4  # At least 4 JSON files
            
            # Validate JSON files can be parsed
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    assert len(data) > 0
            
            # Check stage2_execution directory
            stage2_dir = test_project.get_file_path("stage2_execution")
            assert stage2_dir.exists()
            assert stage2_dir.is_dir()
            
            # Check subdirectories were created
            natural_dir = stage2_dir / "natural_dataset"
            controlled_dir = stage2_dir / "controlled_dataset"
            assert natural_dir.exists()
            assert controlled_dir.exists()
    
    def test_stage1_error_handling(self, test_project, test_config):
        """Test Stage 1 error handling."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            # Remove required input file to trigger error
            brand_context_path = test_project.get_file_path("inputs/brand_context.json")
            brand_context_path.unlink()
            
            generator = Stage1Generator(test_project)
            
            # Should raise StageExecutionError
            with pytest.raises(Exception):  # Will be StageExecutionError when imported
                generator.execute_full_pipeline()
            
            # Reload project config to get updated status
            test_project._load_config()
            
            # Check project status was updated with error
            project_config = test_project.config
            stage1_status = project_config.stage_status['stage1']
            assert stage1_status.status == StageStatusEnum.FAILED
            assert stage1_status.error_message is not None
            assert len(stage1_status.error_message) > 0
    
    def test_stage1_quality_metrics(self, test_project, test_config):
        """Test Stage 1 quality metrics calculation."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            generator = Stage1Generator(test_project)
            result = generator.execute_full_pipeline()
            
            # Check quality metrics are present
            exec_package = result['execution_package']
            quality_metrics = exec_package['quality_metrics']
            
            assert 'archetype_confidence' in quality_metrics
            assert 'query_authenticity' in quality_metrics
            assert 'framework_compliance' in quality_metrics
            
            # Validate metric ranges
            assert 0.0 <= quality_metrics['archetype_confidence'] <= 1.0
            assert 0.0 <= quality_metrics['query_authenticity'] <= 10.0
            assert isinstance(quality_metrics['framework_compliance'], bool)
    
    def test_stage1_prompt_templates(self, test_project, test_config):
        """Test that Stage 1 generates proper prompt templates."""
        
        with patch('src.utils.config.get_config', return_value=test_config):
            generator = Stage1Generator(test_project)
            result = generator.execute_full_pipeline()
            
            prompts = result['execution_package']['prompts']
            
            # Test natural AI prompt
            natural_prompt = prompts['natural_ai']
            assert "helpful shopping assistant" in natural_prompt
            assert "Customer context:" in natural_prompt
            assert "{styled_query}" in natural_prompt
            
            # Test controlled AI prompt  
            controlled_prompt = prompts['controlled_ai']
            assert "SEARCH MANDATE" in controlled_prompt
            assert "HALLUCINATION PREVENTION" in controlled_prompt
            assert "Customer context:" in controlled_prompt
            assert "{styled_query}" in controlled_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])