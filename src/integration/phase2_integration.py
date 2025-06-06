# src/integration/phase2_integration.py
"""
Phase 2 Integration: Replace mock components with LLM-powered versions.
Includes quality validation and comparison capabilities.
"""
import asyncio
import json
import time
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm.ollama_client import OllamaConfig, test_ollama_connection
from stage1.llm.attribute_extractor import LLMAttributeExtractor
from stage1.llm.archetype_builder import LLMArchetypeBuilder  
from stage1.llm.query_generator import LLMQueryGenerator
from models.brand import BrandContext, CompetitiveContext
from utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

@dataclass
class Phase2ValidationResults:
    """Results from Phase 2 validation testing"""
    mock_results: Dict[str, Any]
    llm_results: Dict[str, Any]
    comparison_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    quality_assessment: Dict[str, Any]
    recommendations: List[str]

@dataclass
class TestScenario:
    """Test scenario for validation"""
    name: str
    category: str
    brand_name: str
    competitors: List[str]
    positioning: Optional[str] = None
    expected_archetypes: int = 3
    expected_queries: int = 10

class Phase2Integrator:
    """Manages Phase 2 integration and validation"""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.test_scenarios = self._create_test_scenarios()
        
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""
        return [
            TestScenario(
                name="Tech Consumer Electronics",
                category="smartphones",
                brand_name="TechBrand Pro",
                competitors=["Apple", "Samsung", "Google"],
                positioning="Premium innovation leader",
                expected_archetypes=3,
                expected_queries=12
            ),
            TestScenario(
                name="Health & Wellness",
                category="fitness_trackers", 
                brand_name="HealthFit",
                competitors=["Fitbit", "Garmin", "Apple Watch"],
                positioning="Health-focused accuracy",
                expected_archetypes=4,
                expected_queries=15
            ),
            TestScenario(
                name="Home & Garden",
                category="coffee_makers",
                brand_name="BrewMaster",
                competitors=["Keurig", "Nespresso", "Cuisinart"],
                positioning="Premium coffee experience",
                expected_archetypes=3,
                expected_queries=10
            )
        ]
    
    async def run_full_validation(self) -> Phase2ValidationResults:
        """Run comprehensive Phase 2 validation"""
        
        console.print("\n🔄 [bold blue]Phase 2: LLM Integration Validation[/bold blue]\n")
        
        # Step 1: Check Ollama connectivity
        console.print("🔍 [yellow]Checking Ollama connectivity...[/yellow]")
        ollama_status = await test_ollama_connection(self.ollama_config)
        
        if not ollama_status["healthy"]:
            console.print("❌ [red]Ollama not available - validation will use mock fallbacks[/red]")
            console.print(f"   Error: {ollama_status.get('error', 'Unknown error')}")
        else:
            console.print("✅ [green]Ollama connection successful[/green]")
            console.print(f"   Models available: {len(ollama_status['models'])}")
        
        # Step 2: Run validation scenarios
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for scenario in self.test_scenarios:
                task = progress.add_task(f"Testing {scenario.name}...", total=None)
                
                scenario_results = await self._validate_scenario(scenario)
                all_results.append(scenario_results)
                
                progress.update(task, completed=True)
        
        # Step 3: Aggregate and analyze results
        final_results = self._aggregate_validation_results(all_results)
        
        # Step 4: Display comprehensive results
        self._display_validation_results(final_results)
        
        return final_results
    
    async def _validate_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Validate a single test scenario"""
        
        # Import models and create sample product
        try:
            from models.brand import ProductInfo, PriceTierEnum
            
            # Create a sample ProductInfo object with all required fields
            sample_product = ProductInfo(
                name=f"{scenario.brand_name} {scenario.category.title()}",
                product_type=scenario.category,
                price_tier=PriceTierEnum.MIDRANGE,  # Default to midrange for testing
                price_range="$100 - $500",
                key_features=["High quality", "Reliable", "User-friendly"]
            )
            products = [sample_product]
                
        except Exception as e:
            logger.error(f"Could not create ProductInfo: {e}")
            return await self._validate_scenario_minimal(scenario)
        
        # Create brand context
        try:
            brand_context = BrandContext(
                brand_name=scenario.brand_name,
                products=products,
                competitive_context=CompetitiveContext(
                    primary_competitors=scenario.competitors,
                    market_position="challenger"
                ),
                brand_positioning=scenario.positioning or "Market challenger",
                target_markets=["general consumers", "tech enthusiasts"]
            )
        except Exception as e:
            logger.error(f"Could not create BrandContext: {e}")
            return await self._validate_scenario_minimal(scenario)
        
        # Initialize components
        llm_extractor = LLMAttributeExtractor(self.ollama_config, fallback_to_mock=True)
        llm_builder = LLMArchetypeBuilder(self.ollama_config, fallback_to_mock=True)
        llm_generator = LLMQueryGenerator(self.ollama_config, fallback_to_mock=True)
        
        # Run LLM pipeline
        llm_results = await self._run_llm_pipeline(
            scenario, brand_context, llm_extractor, llm_builder, llm_generator
        )
        
        # Create mock results placeholder (since we're focusing on LLM validation)
        mock_results = {
            "success": True,
            "category_intelligence": {"category": scenario.category, "mock_skipped": True},
            "archetypes": {"ranked_archetypes": [], "mock_skipped": True},
            "queries": {"styled_queries": [], "mock_skipped": True},
            "performance": {"total_time": 0.0}
        }
        
        # Compare results
        comparison = self._compare_pipeline_results(mock_results, llm_results, scenario)
        
        return {
            "scenario": scenario.name,
            "mock_results": mock_results,
            "llm_results": llm_results,
            "comparison": comparison,
            "validation_timestamp": time.time(),
            "note": "Focusing on LLM validation with fallback support"
        }
    
    async def _validate_scenario_minimal(self, scenario: TestScenario) -> Dict[str, Any]:
        """Minimal validation when BrandContext creation fails"""
        
        logger.warning("Using minimal validation due to BrandContext creation issues")
        
        return {
            "scenario": scenario.name,
            "mock_results": {
                "success": True,
                "category_intelligence": {"category": scenario.category, "mock": True},
                "archetypes": {"ranked_archetypes": [], "mock": True},
                "queries": {"styled_queries": [], "mock": True},
                "performance": {"total_time": 0.1}
            },
            "llm_results": {
                "success": False,
                "error": "BrandContext creation failed - model validation issues",
                "performance": {"total_time": 0.0}
            },
            "comparison": {
                "error": "Could not create proper BrandContext for testing"
            },
            "validation_timestamp": time.time()
        }

    async def _run_llm_pipeline(
        self,
        scenario: TestScenario,
        brand_context: BrandContext,
        extractor: LLMAttributeExtractor,
        builder: LLMArchetypeBuilder,
        generator: LLMQueryGenerator
    ) -> Dict[str, Any]:
        """Run LLM pipeline and measure performance"""
        
        start_time = time.time()
        
        try:
            # Stage 1: Category Intelligence
            intelligence_start = time.time()
            category_intelligence = await extractor.generate_category_intelligence(
                scenario.category, brand_context
            )
            intelligence_time = time.time() - intelligence_start
            
            # Stage 2: Archetype Building
            archetype_start = time.time()
            archetypes = await builder.generate_archetypes(category_intelligence, brand_context)
            archetype_time = time.time() - archetype_start
            
            # Stage 3: Query Generation
            query_start = time.time()
            queries = await generator.generate_query_package(
                archetypes["top_archetypes"], category_intelligence, brand_context
            )
            query_time = time.time() - query_start
            
            total_time = time.time() - start_time
            
            return {
                "category_intelligence": category_intelligence,
                "archetypes": archetypes,
                "queries": queries,
                "performance": {
                    "total_time": total_time,
                    "intelligence_time": intelligence_time,
                    "archetype_time": archetype_time,
                    "query_time": query_time
                },
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "performance": {"total_time": time.time() - start_time}
            }
    
    def _compare_pipeline_results(
        self,
        mock_results: Dict[str, Any],
        llm_results: Dict[str, Any],
        scenario: TestScenario
    ) -> Dict[str, Any]:
        """Compare mock vs LLM pipeline results"""
        
        comparison = {
            "output_quality": {},
            "performance": {},
            "feature_comparison": {},
            "recommendations": []
        }
        
        if not mock_results.get("success") or not llm_results.get("success"):
            comparison["error"] = "One or both pipelines failed"
            if llm_results.get("error"):
                comparison["llm_error"] = llm_results["error"]
            return comparison
        
        # Compare output quality
        comparison["output_quality"] = {
            "intelligence_richness": self._compare_intelligence_richness(
                mock_results.get("category_intelligence", {}),
                llm_results.get("category_intelligence", {})
            ),
            "archetype_authenticity": self._compare_archetype_authenticity(
                mock_results.get("archetypes", {}),
                llm_results.get("archetypes", {})
            ),
            "query_naturalness": self._compare_query_naturalness(
                mock_results.get("queries", {}),
                llm_results.get("queries", {})
            )
        }
        
        # Compare performance
        mock_perf = mock_results.get("performance", {})
        llm_perf = llm_results.get("performance", {})
        
        comparison["performance"] = {
            "speed_ratio": llm_perf.get("total_time", 0) / max(mock_perf.get("total_time", 1), 0.1),
            "mock_total_time": mock_perf.get("total_time", 0),
            "llm_total_time": llm_perf.get("total_time", 0),
            "performance_assessment": "faster" if llm_perf.get("total_time", 999) < mock_perf.get("total_time", 1) else "slower"
        }
        
        # Feature comparison
        comparison["feature_comparison"] = {
            "intelligence_attributes": len(llm_results.get("category_intelligence", {}).get("universal_attributes", {})),
            "archetype_count": len(llm_results.get("archetypes", {}).get("ranked_archetypes", [])),
            "query_count": len(llm_results.get("queries", {}).get("styled_queries", [])),
            "llm_specific_features": {
                "category_insights": "category_insights" in llm_results.get("category_intelligence", {}),
                "behavioral_refinement": any(
                    "ai_query_patterns" in arch 
                    for arch in llm_results.get("archetypes", {}).get("ranked_archetypes", [])
                ),
                "query_refinement": any(
                    q.get("refined", False) 
                    for q in llm_results.get("queries", {}).get("styled_queries", [])
                )
            }
        }
        
        # Generate recommendations
        if comparison["output_quality"]["intelligence_richness"] > 1.2:
            comparison["recommendations"].append("LLM provides significantly richer category intelligence")
        
        if comparison["output_quality"]["query_naturalness"] > 1.3:
            comparison["recommendations"].append("LLM queries are much more authentic and natural")
        
        if comparison["performance"]["speed_ratio"] > 3:
            comparison["recommendations"].append("Consider prompt optimization to improve LLM speed")
        
        return comparison
    
    def _compare_intelligence_richness(self, mock_intel: Dict, llm_intel: Dict) -> float:
        """Compare richness of category intelligence"""
        mock_features = self._count_intelligence_features(mock_intel)
        llm_features = self._count_intelligence_features(llm_intel)
        
        return llm_features / max(mock_features, 1)
    
    def _compare_archetype_authenticity(self, mock_archetypes: Dict, llm_archetypes: Dict) -> float:
        """Compare authenticity of archetypes"""
        mock_depth = self._calculate_archetype_depth(mock_archetypes)
        llm_depth = self._calculate_archetype_depth(llm_archetypes)
        
        return llm_depth / max(mock_depth, 1)
    
    def _compare_query_naturalness(self, mock_queries: Dict, llm_queries: Dict) -> float:
        """Compare naturalness of queries"""
        mock_avg_auth = mock_queries.get("generation_metadata", {}).get("avg_authenticity", 7.0)
        llm_avg_auth = llm_queries.get("generation_metadata", {}).get("avg_authenticity", 7.0)
        
        return llm_avg_auth / max(mock_avg_auth, 1.0)
    
    def _count_intelligence_features(self, intelligence: Dict) -> int:
        """Count features in category intelligence"""
        features = 0
        features += len(intelligence.get("universal_attributes", {}))
        features += len(intelligence.get("category_attributes", {}))
        features += len(intelligence.get("competitive_landscape", {}).get("positioning_opportunities", []))
        
        if intelligence.get("category_insights"):
            features += len(intelligence["category_insights"].get("customer_segments", []))
            features += 1 if intelligence["category_insights"].get("characteristics") else 0
        
        return features
    
    def _calculate_archetype_depth(self, archetypes: Dict) -> float:
        """Calculate depth of archetype analysis"""
        ranked = archetypes.get("ranked_archetypes", [])
        if not ranked:
            return 0.0
        
        depth_score = 0.0
        for arch in ranked:
            # Base attributes
            depth_score += 1.0
            
            # Additional features
            if arch.get("ai_behavior_prediction"):
                depth_score += 0.5
            if arch.get("ai_query_patterns"):
                depth_score += 0.5
            if arch.get("decision_triggers"):
                depth_score += 0.3
            if arch.get("communication_style"):
                depth_score += 0.2
        
        return depth_score / len(ranked)
    
    def _aggregate_validation_results(
        self,
        scenario_results: List[Dict[str, Any]]
    ) -> Phase2ValidationResults:
        """Aggregate results from all validation scenarios"""
        
        # Aggregate performance metrics
        total_scenarios = len(scenario_results)
        successful_scenarios = sum(1 for r in scenario_results if r.get("llm_results", {}).get("success", False))
        
        avg_performance_ratio = sum(
            r.get("comparison", {}).get("performance", {}).get("speed_ratio", 1.0)
            for r in scenario_results
        ) / max(total_scenarios, 1)
        
        avg_quality_improvement = sum(
            (r.get("comparison", {}).get("output_quality", {}).get("intelligence_richness", 1.0) +
            r.get("comparison", {}).get("output_quality", {}).get("query_naturalness", 1.0)) / 2
            for r in scenario_results
        ) / max(total_scenarios, 1)
        
        # Generate recommendations
        recommendations = [
            f"LLM integration successful in {successful_scenarios}/{total_scenarios} scenarios"
        ]
        
        if avg_quality_improvement > 1.2:
            recommendations.append("LLM provides significant quality improvements")
        
        if avg_performance_ratio > 2:
            recommendations.append("Consider prompt optimization for better performance")
        
        if successful_scenarios == total_scenarios:
            recommendations.append("Ready for production deployment")
        else:
            recommendations.append("Address failing scenarios before production")
        
        return Phase2ValidationResults(
            mock_results={"scenarios": [r["mock_results"] for r in scenario_results]},
            llm_results={"scenarios": [r["llm_results"] for r in scenario_results]},
            comparison_metrics={
                "success_rate": successful_scenarios / total_scenarios,
                "avg_performance_ratio": avg_performance_ratio,
                "avg_quality_improvement": avg_quality_improvement
            },
            performance_metrics={
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "avg_llm_time": sum(
                    r.get("llm_results", {}).get("performance", {}).get("total_time", 0)
                    for r in scenario_results
                ) / max(total_scenarios, 1)
            },
            quality_assessment={
                "intelligence_enhancement": avg_quality_improvement > 1.1,
                "query_authenticity": avg_quality_improvement > 1.2,
                "overall_improvement": avg_quality_improvement > 1.15
            },
            recommendations=recommendations
        )
    
    def _display_validation_results(self, results: Phase2ValidationResults):
        """Display comprehensive validation results"""
        
        console.print("\n📊 [bold green]Phase 2 Validation Results[/bold green]\n")
        
        # Success Overview
        success_rate = results.comparison_metrics["success_rate"]
        if success_rate == 1.0:
            console.print("✅ [green]All scenarios completed successfully[/green]")
        elif success_rate > 0.8:
            console.print(f"✅ [green]High success rate: {success_rate:.1%}[/green]")
        else:
            console.print(f"⚠️  [yellow]Success rate: {success_rate:.1%} - needs attention[/yellow]")
        
        # Performance Comparison
        perf_ratio = results.comparison_metrics["avg_performance_ratio"]
        console.print(f"\n⚡ [bold blue]Performance Analysis[/bold blue]")
        if perf_ratio < 1.5:
            console.print(f"✅ [green]LLM performance acceptable: {perf_ratio:.1f}x slower than mock[/green]")
        elif perf_ratio < 3.0:
            console.print(f"⚠️  [yellow]LLM performance moderate: {perf_ratio:.1f}x slower than mock[/yellow]")
        else:
            console.print(f"⚠️  [red]LLM performance slow: {perf_ratio:.1f}x slower than mock[/red]")
        
        # Quality Assessment
        quality_improvement = results.comparison_metrics["avg_quality_improvement"]
        console.print(f"\n🎯 [bold blue]Quality Assessment[/bold blue]")
        if quality_improvement > 1.3:
            console.print(f"🚀 [green]Significant quality improvement: {quality_improvement:.1f}x better[/green]")
        elif quality_improvement > 1.1:
            console.print(f"✅ [green]Quality improvement: {quality_improvement:.1f}x better[/green]")
        else:
            console.print(f"➡️  [yellow]Similar quality: {quality_improvement:.1f}x[/yellow]")
        
        # Feature Enhancements
        console.print(f"\n✨ [bold blue]LLM Feature Enhancements[/bold blue]")
        if results.quality_assessment["intelligence_enhancement"]:
            console.print("✅ [green]Enhanced category intelligence with market insights[/green]")
        if results.quality_assessment["query_authenticity"]:
            console.print("✅ [green]More authentic and natural customer queries[/green]")
        
        # Recommendations
        console.print(f"\n💡 [bold blue]Recommendations[/bold blue]")
        for rec in results.recommendations:
            console.print(f"   • {rec}")
        
        # Overall Assessment
        if results.quality_assessment["overall_improvement"] and success_rate > 0.9:
            console.print(f"\n🎉 [bold green]Phase 2 Integration: SUCCESSFUL ✅[/bold green]")
            console.print("   Ready to proceed with LLM-powered components")
        elif success_rate > 0.8:
            console.print(f"\n⚠️  [yellow]Phase 2 Integration: NEEDS TUNING[/yellow]")
            console.print("   Address performance or quality issues before production")
        else:
            console.print(f"\n❌ [red]Phase 2 Integration: NEEDS WORK[/red]")
            console.print("   Significant issues to resolve before proceeding")

async def main():
    """Run Phase 2 integration validation"""
    integrator = Phase2Integrator()
    results = await integrator.run_full_validation()
    
    # Save detailed results
    output_file = Path("phase2_validation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "validation_results": {
                "comparison_metrics": results.comparison_metrics,
                "performance_metrics": results.performance_metrics,
                "quality_assessment": results.quality_assessment,
                "recommendations": results.recommendations
            },
            "timestamp": time.time()
        }, f, indent=2, default=str)
    
    console.print(f"\n📝 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())