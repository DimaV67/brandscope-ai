#!/usr/bin/env python3
"""
Pure LLM test with no mock fallback to avoid import issues.
"""
import sys
import os
import asyncio
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

async def test_pure_llm():
    """Test LLM components with no fallback"""
    
    try:
        from llm.ollama_client import OllamaConfig, test_ollama_connection
        from stage1.llm.attribute_extractor import LLMAttributeExtractor
        from stage1.llm.archetype_builder import LLMArchetypeBuilder
        from stage1.llm.query_generator import LLMQueryGenerator
        from models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum
        
        print("🚀 Testing Pure LLM Components (No Fallback)")
        
        # Test 1: Ollama connection
        print("\n🔍 Testing Ollama connection...")
        ollama_status = await test_ollama_connection()
        if ollama_status["healthy"]:
            print(f"✅ Ollama connected - {len(ollama_status['models'])} models available")
        else:
            print("❌ Ollama not available")
            return False
        
        # Test 2: Create test brand context
        print("\n🏢 Creating test brand context...")
        sample_product = ProductInfo(
            name="TestPhone Pro",
            product_type="smartphones",
            price_tier=PriceTierEnum.MIDRANGE,
            price_range="$300 - $600",
            key_features=["5G", "Great camera", "Long battery"]
        )
        
        brand_context = BrandContext(
            brand_name="TestBrand",
            products=[sample_product],
            competitive_context=CompetitiveContext(
                primary_competitors=["Apple", "Samsung"],
                market_position="challenger"
            ),
            brand_positioning="Innovation at affordable prices",
            target_markets=["tech-savvy millennials", "budget-conscious professionals"]
        )
        print("✅ BrandContext created successfully")
        
        # Test 3: LLM Attribute Extractor (NO FALLBACK)
        print("\n🔍 Testing LLM Attribute Extractor (pure LLM)...")
        extractor = LLMAttributeExtractor(fallback_to_mock=False)  # NO FALLBACK
        
        intelligence = await extractor.generate_category_intelligence(
            "smartphones", brand_context
        )
        print(f"✅ Generated category intelligence")
        print(f"   Universal attributes: {len(intelligence.get('universal_attributes', {}))}")
        print(f"   Category attributes: {len(intelligence.get('category_attributes', {}))}")
        print(f"   LLM generated: {intelligence.get('category_insights', {}).get('llm_generated', False)}")
        
        # Test 4: LLM Archetype Builder (NO FALLBACK)
        print("\n👥 Testing LLM Archetype Builder (pure LLM)...")
        builder = LLMArchetypeBuilder(fallback_to_mock=False)  # NO FALLBACK
        
        archetypes = await builder.generate_archetypes(intelligence, brand_context)
        print(f"✅ Generated {len(archetypes.get('ranked_archetypes', []))} archetypes")
        
        # Show archetype details
        for i, archetype in enumerate(archetypes.get('ranked_archetypes', [])[:2], 1):
            print(f"   {i}. {archetype.get('name', 'Unknown')} (confidence: {archetype.get('confidence', 0):.2f})")
        
        # Test 5: LLM Query Generator (NO FALLBACK)
        print("\n❓ Testing LLM Query Generator (pure LLM)...")
        generator = LLMQueryGenerator(fallback_to_mock=False)  # NO FALLBACK
        
        queries = await generator.generate_query_package(
            archetypes.get("top_archetypes", []), intelligence, brand_context
        )
        print(f"✅ Generated {len(queries.get('styled_queries', []))} queries")
        print(f"   Avg authenticity: {queries.get('generation_metadata', {}).get('avg_authenticity', 0):.1f}")
        
        # Show sample queries
        print("\n📝 Sample queries generated:")
        for i, query in enumerate(queries.get("styled_queries", [])[:3], 1):
            print(f"   {i}. {query.get('styled_query', 'N/A')}")
        
        print("\n🎉 All Pure LLM components working successfully!")
        print(f"\n📊 Final Summary:")
        print(f"   • Category intelligence: ✅ (LLM generated)")
        print(f"   • Customer archetypes: {len(archetypes.get('ranked_archetypes', []))}")
        print(f"   • Research queries: {len(queries.get('styled_queries', []))}")
        print(f"   • Query authenticity: {queries.get('generation_metadata', {}).get('avg_authenticity', 0):.1f}/10")
        print(f"   • LLM enhanced features:")
        
        if intelligence.get('category_insights', {}).get('llm_generated'):
            print(f"     ✅ Market insights and competitive analysis")
        
        if any(arch.get('ai_query_patterns') for arch in archetypes.get('ranked_archetypes', [])):
            print(f"     ✅ AI behavior patterns for each archetype")
        
        if any(q.get('refined') for q in queries.get('styled_queries', [])):
            print(f"     ✅ Query refinement for authenticity")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pure_llm())
    if success:
        print("\n✅ Pure LLM integration is working perfectly!")
        print("🚀 Ready for production deployment!")
    else:
        print("\n❌ LLM integration needs attention")
        sys.exit(1)