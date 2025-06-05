# src/migration/phase2_migration.py
"""
Migration helper for Phase 2: Switch from mock to LLM components.
Provides backwards compatibility and gradual migration path.
"""
import asyncio
import sys
import os
from typing import Dict, Any, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm.ollama_client import OllamaConfig
from stage1.llm.attribute_extractor import LLMAttributeExtractor
from stage1.llm.archetype_builder import LLMArchetypeBuilder
from stage1.llm.query_generator import LLMQueryGenerator
from utils.logging import get_logger

logger = get_logger(__name__)

class ComponentMode(Enum):
    """Component execution modes"""
    MOCK_ONLY = "mock"
    LLM_ONLY = "llm"
    LLM_WITH_FALLBACK = "llm_fallback"
    HYBRID = "hybrid"  # Use both and compare

@dataclass
class MigrationConfig:
    """Configuration for component migration"""
    mode: ComponentMode = ComponentMode.LLM_WITH_FALLBACK
    ollama_config: Optional[OllamaConfig] = None
    enable_performance_logging: bool = True
    enable_quality_comparison: bool = False
    fallback_timeout: float = 30.0  # seconds

class ComponentFactory:
    """Factory for creating components based on migration configuration"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self._mock_components = {}
        self._llm_components = {}
    
    def create_attribute_extractor(self):
        """Create attribute extractor based on migration mode"""
        
        if self.config.mode == ComponentMode.MOCK_ONLY:
            return self._get_mock_extractor()
        
        elif self.config.mode == ComponentMode.LLM_ONLY:
            return LLMAttributeExtractor(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=False
            )
        
        elif self.config.mode == ComponentMode.LLM_WITH_FALLBACK:
            return LLMAttributeExtractor(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=True
            )
        
        elif self.config.mode == ComponentMode.HYBRID:
            return HybridAttributeExtractor(
                ollama_config=self.config.ollama_config,
                enable_comparison=self.config.enable_quality_comparison
            )
        
        else:
            raise ValueError(f"Unknown component mode: {self.config.mode}")
    
    def create_archetype_builder(self):
        """Create archetype builder based on migration mode"""
        
        if self.config.mode == ComponentMode.MOCK_ONLY:
            return self._get_mock_builder()
        
        elif self.config.mode == ComponentMode.LLM_ONLY:
            return LLMArchetypeBuilder(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=False
            )
        
        elif self.config.mode == ComponentMode.LLM_WITH_FALLBACK:
            return LLMArchetypeBuilder(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=True
            )
        
        elif self.config.mode == ComponentMode.HYBRID:
            return HybridArchetypeBuilder(
                ollama_config=self.config.ollama_config,
                enable_comparison=self.config.enable_quality_comparison
            )
        
        else:
            raise ValueError(f"Unknown component mode: {self.config.mode}")
    
    def create_query_generator(self):
        """Create query generator based on migration mode"""
        
        if self.config.mode == ComponentMode.MOCK_ONLY:
            return self._get_mock_generator()
        
        elif self.config.mode == ComponentMode.LLM_ONLY:
            return LLMQueryGenerator(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=False
            )
        
        elif self.config.mode == ComponentMode.LLM_WITH_FALLBACK:
            return LLMQueryGenerator(
                ollama_config=self.config.ollama_config,
                fallback_to_mock=True
            )
        
        elif self.config.mode == ComponentMode.HYBRID:
            return HybridQueryGenerator(
                ollama_config=self.config.ollama_config,
                enable_comparison=self.config.enable_quality_comparison
            )
        
        else:
            raise ValueError(f"Unknown component mode: {self.config.mode}")
    
    def _get_mock_extractor(self):
        """Get mock attribute extractor with caching"""
        if "extractor" not in self._mock_components:
            from stage1.attribute_extractor import AttributeExtractor
            self._mock_components["extractor"] = AttributeExtractor()
        return self._mock_components["extractor"]
    
    def _get_mock_builder(self):
        """Get mock archetype builder with caching"""
        if "builder" not in self._mock_components:
            from stage1.archetype_builder import ArchetypeBuilder
            self._mock_components["builder"] = ArchetypeBuilder()
        return self._mock_components["builder"]
    
    def _get_mock_generator(self):
        """Get mock query generator with caching"""
        if "generator" not in self._mock_components:
            from stage1.query_generator import QueryGenerator
            self._mock_components["generator"] = QueryGenerator()
        return self._mock_components["generator"]

class HybridAttributeExtractor:
    """Hybrid extractor that runs both mock and LLM for comparison"""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None, enable_comparison: bool = True):
        self.mock_extractor = self._get_mock_extractor()
        self.llm_extractor = LLMAttributeExtractor(ollama_config, fallback_to_mock=True)
        self.enable_comparison = enable_comparison
    
    async def generate_category_intelligence(self, category: str, brand_context, customer_narrative: Optional[str] = None):
        """Generate intelligence using both components and compare results"""
        
        # Run mock (synchronous)
        mock_result = self.mock_extractor.generate_category_intelligence(
            category, brand_context, customer_narrative
        )
        
        # Run LLM (asynchronous)
        llm_result = await self.llm_extractor.generate_category_intelligence(
            category, brand_context, customer_narrative
        )
        
        if self.enable_comparison:
            comparison = self._compare_results(mock_result, llm_result)
            logger.info("Hybrid attribute extraction comparison", metadata=comparison)
        
        # Return LLM result (preferred) with mock as metadata
        llm_result["_hybrid_metadata"] = {
            "mock_result": mock_result,
            "comparison_enabled": self.enable_comparison
        }
        
        return llm_result
    
    def _compare_results(self, mock_result: Dict, llm_result: Dict) -> Dict:
        """Compare mock vs LLM results"""
        return {
            "mock_attributes": len(mock_result.get("universal_attributes", {})),
            "llm_attributes": len(llm_result.get("universal_attributes", {})),
            "llm_has_insights": "category_insights" in llm_result,
            "llm_enhanced": len(llm_result.get("universal_attributes", {})) > len(mock_result.get("universal_attributes", {}))
        }
    
    def _get_mock_extractor(self):
        """Get mock extractor"""
        from stage1.attribute_extractor import AttributeExtractor
        return AttributeExtractor()

class HybridArchetypeBuilder:
    """Hybrid builder that runs both mock and LLM for comparison"""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None, enable_comparison: bool = True):
        self.mock_builder = self._get_mock_builder()
        self.llm_builder = LLMArchetypeBuilder(ollama_config, fallback_to_mock=True)
        self.enable_comparison = enable_comparison
    
    async def generate_archetypes(self, category_intelligence: Dict, brand_context):
        """Generate archetypes using both components and compare results"""
        
        # Run mock (synchronous)
        mock_result = self.mock_builder.generate_archetypes(category_intelligence, brand_context)
        
        # Run LLM (asynchronous)
        llm_result = await self.llm_builder.generate_archetypes(category_intelligence, brand_context)
        
        if self.enable_comparison:
            comparison = self._compare_results(mock_result, llm_result)
            logger.info("Hybrid archetype generation comparison", metadata=comparison)
        
        # Return LLM result with mock as metadata
        llm_result["_hybrid_metadata"] = {
            "mock_result": mock_result,
            "comparison_enabled": self.enable_comparison
        }
        
        return llm_result
    
    def _compare_results(self, mock_result: Dict, llm_result: Dict) -> Dict:
        """Compare mock vs LLM archetype results"""
        mock_archetypes = mock_result.get("ranked_archetypes", [])
        llm_archetypes = llm_result.get("ranked_archetypes", [])
        
        return {
            "mock_archetype_count": len(mock_archetypes),
            "llm_archetype_count": len(llm_archetypes),
            "llm_has_behavioral_data": any(
                "ai_query_patterns" in arch for arch in llm_archetypes
            ),
            "avg_confidence_improvement": (
                llm_result.get("generation_metadata", {}).get("avg_confidence", 0.8) -
                mock_result.get("generation_metadata", {}).get("avg_confidence", 0.8)
            )
        }
    
    def _get_mock_builder(self):
        """Get mock builder"""
        from stage1.archetype_builder import ArchetypeBuilder
        return ArchetypeBuilder()

class HybridQueryGenerator:
    """Hybrid generator that runs both mock and LLM for comparison"""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None, enable_comparison: bool = True):
        self.mock_generator = self._get_mock_generator()
        self.llm_generator = LLMQueryGenerator(ollama_config, fallback_to_mock=True)
        self.enable_comparison = enable_comparison
    
    async def generate_query_package(self, top_archetypes: list, category_intelligence: Dict, brand_context):
        """Generate queries using both components and compare results"""
        
        # Run mock (synchronous)
        mock_result = self.mock_generator.generate_query_package(
            top_archetypes, category_intelligence, brand_context
        )
        
        # Run LLM (asynchronous)
        llm_result = await self.llm_generator.generate_query_package(
            top_archetypes, category_intelligence, brand_context
        )
        
        if self.enable_comparison:
            comparison = self._compare_results(mock_result, llm_result)
            logger.info("Hybrid query generation comparison", metadata=comparison)
        
        # Return LLM result with mock as metadata
        llm_result["_hybrid_metadata"] = {
            "mock_result": mock_result,
            "comparison_enabled": self.enable_comparison
        }
        
        return llm_result
    
    def _compare_results(self, mock_result: Dict, llm_result: Dict) -> Dict:
        """Compare mock vs LLM query results"""
        mock_queries = mock_result.get("styled_queries", [])
        llm_queries = llm_result.get("styled_queries", [])
        
        mock_auth = mock_result.get("generation_metadata", {}).get("avg_authenticity", 7.0)
        llm_auth = llm_result.get("generation_metadata", {}).get("avg_authenticity", 7.0)
        
        return {
            "mock_query_count": len(mock_queries),
            "llm_query_count": len(llm_queries),
            "authenticity_improvement": llm_auth - mock_auth,
            "llm_has_refinements": any(q.get("refined", False) for q in llm_queries),
            "categories_coverage_improvement": (
                len(llm_result.get("generation_metadata", {}).get("categories_covered", [])) -
                len(mock_result.get("generation_metadata", {}).get("categories_covered", []))
            )
        }
    
    def _get_mock_generator(self):
        """Get mock generator"""
        from stage1.query_generator import QueryGenerator
        return QueryGenerator()

class MigrationManager:
    """Manages the overall migration process"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.factory = ComponentFactory(config)
    
    async def validate_migration_readiness(self) -> Dict[str, Any]:
        """Validate that the system is ready for migration"""
        
        readiness = {
            "ollama_available": False,
            "models_ready": False,
            "components_functional": False,
            "performance_acceptable": False,
            "ready_for_migration": False
        }
        
        # Check Ollama availability
        from llm.ollama_client import test_ollama_connection
        
        try:
            ollama_status = await test_ollama_connection(self.config.ollama_config)
            readiness["ollama_available"] = ollama_status["healthy"]
            readiness["models_ready"] = len(ollama_status.get("models", [])) > 0
            
            if ollama_status.get("test_generation"):
                test_gen = ollama_status["test_generation"]
                readiness["components_functional"] = test_gen.get("success", False)
                readiness["performance_acceptable"] = (
                    test_gen.get("response_time", 999) < self.config.fallback_timeout
                )
        
        except Exception as e:
            logger.error(f"Migration readiness check failed: {e}")
            readiness["error"] = str(e)
        
        # Overall readiness assessment
        readiness["ready_for_migration"] = all([
            readiness["ollama_available"],
            readiness["models_ready"],
            readiness["components_functional"]
        ])
        
        return readiness
    
    def get_component_set(self) -> Dict[str, Any]:
        """Get a complete set of components for the current migration mode"""
        
        return {
            "attribute_extractor": self.factory.create_attribute_extractor(),
            "archetype_builder": self.factory.create_archetype_builder(),
            "query_generator": self.factory.create_query_generator(),
            "mode": self.config.mode.value,
            "config": self.config
        }
    
    async def run_migration_test(self, test_scenario: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a comprehensive migration test"""
        
        if not test_scenario:
            test_scenario = {
                "category": "smartphones",
                "brand_name": "TestBrand",
                "competitors": ["Apple", "Samsung", "Google"],
                "positioning": "Innovation leader"
            }
        
        from models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum
        
        # Create sample product
        sample_product = ProductInfo(
            name=f"{test_scenario['brand_name']} {test_scenario['category'].title()}",
            product_type=test_scenario["category"],
            price_tier=PriceTierEnum.MIDRANGE,
            price_range="$100 - $500", 
            key_features=["High quality", "Reliable", "User-friendly"]
        )
        
        brand_context = BrandContext(
            brand_name=test_scenario["brand_name"],
            products=[sample_product],
            competitive_context=CompetitiveContext(
                primary_competitors=test_scenario["competitors"],
                market_position="challenger"
            ),
            brand_positioning=test_scenario.get("positioning", "Market challenger"),
            target_markets=["general consumers", "category enthusiasts"]
        )
        
        components = self.get_component_set()
        
        try:
            # Test full pipeline
            import time
            start_time = time.time()
            
            # Stage 1: Attribute Extraction
            extractor = components["attribute_extractor"]
            if hasattr(extractor, 'generate_category_intelligence') and asyncio.iscoroutinefunction(extractor.generate_category_intelligence):
                category_intelligence = await extractor.generate_category_intelligence(
                    test_scenario["category"], brand_context
                )
            else:
                category_intelligence = extractor.generate_category_intelligence(
                    test_scenario["category"], brand_context
                )
            
            # Stage 2: Archetype Building
            builder = components["archetype_builder"]
            if hasattr(builder, 'generate_archetypes') and asyncio.iscoroutinefunction(builder.generate_archetypes):
                archetypes = await builder.generate_archetypes(category_intelligence, brand_context)
            else:
                archetypes = builder.generate_archetypes(category_intelligence, brand_context)
            
            # Stage 3: Query Generation
            generator = components["query_generator"]
            if hasattr(generator, 'generate_query_package') and asyncio.iscoroutinefunction(generator.generate_query_package):
                queries = await generator.generate_query_package(
                    archetypes["top_archetypes"], category_intelligence, brand_context
                )
            else:
                queries = generator.generate_query_package(
                    archetypes["top_archetypes"], category_intelligence, brand_context
                )
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "total_time": total_time,
                "results": {
                    "category_intelligence": category_intelligence,
                    "archetypes": archetypes,
                    "queries": queries
                },
                "mode": self.config.mode.value,
                "performance_metrics": {
                    "queries_generated": len(queries.get("styled_queries", [])),
                    "archetypes_generated": len(archetypes.get("ranked_archetypes", [])),
                    "avg_authenticity": queries.get("generation_metadata", {}).get("avg_authenticity", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Migration test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": self.config.mode.value
            }

# Convenience functions for easy migration
def create_migration_config(
    mode: str = "llm_fallback",
    ollama_host: str = "localhost",
    ollama_port: int = 11434
) -> MigrationConfig:
    """Create migration configuration with sensible defaults"""
    
    mode_enum = ComponentMode(mode)
    ollama_config = OllamaConfig(host=ollama_host, port=ollama_port)
    
    return MigrationConfig(
        mode=mode_enum,
        ollama_config=ollama_config,
        enable_performance_logging=True,
        enable_quality_comparison=(mode == "hybrid")
    )

def get_production_components(config: Optional[MigrationConfig] = None):
    """Get production-ready components with recommended settings"""
    
    if config is None:
        config = create_migration_config(mode="llm_fallback")
    
    manager = MigrationManager(config)
    return manager.get_component_set()

async def quick_migration_test(mode: str = "llm_fallback") -> bool:
    """Quick test to verify migration is working"""
    
    config = create_migration_config(mode=mode)
    manager = MigrationManager(config)
    
    # Check readiness
    readiness = await manager.validate_migration_readiness()
    if not readiness["ready_for_migration"]:
        print(f"❌ Migration not ready: {readiness}")
        return False
    
    # Run test
    test_result = await manager.run_migration_test()
    if test_result["success"]:
        print(f"✅ Migration test successful in {test_result['total_time']:.2f}s")
        return True
    else:
        print(f"❌ Migration test failed: {test_result.get('error', 'Unknown error')}")
        return False

# Example usage script
async def main():
    """Example migration workflow"""
    print("🚀 BrandScope Phase 2 Migration")
    
    # Test different modes
    modes = ["llm_fallback", "hybrid", "llm_only"]
    
    for mode in modes:
        print(f"\n🔍 Testing {mode} mode...")
        success = await quick_migration_test(mode)
        if success:
            print(f"✅ {mode} mode: READY")
        else:
            print(f"❌ {mode} mode: NOT READY")
    
    print("\n🎯 Recommended production configuration:")
    print("   Mode: llm_fallback (LLM with mock fallback)")
    print("   This provides the best balance of quality and reliability")

if __name__ == "__main__":
    asyncio.run(main())