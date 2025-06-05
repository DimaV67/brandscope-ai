#!/usr/bin/env python3
"""
Simple test of LLM components without mock comparison.
"""
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

async def test_llm_components_only():
    """Test LLM components in isolation"""
    
    try:
        # Import LLM components
        from llm.ollama_client import OllamaConfig, test_ollama_connection
        from stage1.llm.attribute_extractor import LLMAttributeExtractor
        from stage1.llm.archetype_builder import LLMArchetypeBuilder
        from stage1.llm.query_generator import LLMQueryGenerator
        from models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum
        
        print("🚀 Testing LLM Components Only")
        
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
        
        # Test 3: LLM Attribute Extractor
        print("\n🔍 Testing LLM Attribute Extractor...")
        extractor = LLMAttributeExtractor(fallback_to_mock=False)
        
        try:
            intelligence = await extractor.generate_category_intelligence(
                "smartphones", brand_context
            )
            print(f"✅ Generated category intelligence with {len(intelligence.get('universal_attributes', {}))} attributes")
        except Exception as e:
            print(f"❌ Attribute extraction failed: {e}")
            return False
        
        # Test 4: LLM Archetype Builder
        print("\n👥 Testing LLM Archetype Builder...")
        builder = LLMArchetypeBuilder(fallback_to_mock=False)
        
        try:
            archetypes = await builder.generate_archetypes(intelligence, brand_context)
            print(f"✅ Generated {len(archetypes.get('ranked_archetypes', []))} archetypes")
        except Exception as e:
            print(f"❌ Archetype building failed: {e}")
            return False
        
        # Test 5: LLM Query Generator
        print("\n❓ Testing LLM Query Generator...")
        generator = LLMQueryGenerator(fallback_to_mock=False)
        
        try:
            queries = await generator.generate_query_package(
                archetypes["top_archetypes"], intelligence, brand_context
            )
            print(f"✅ Generated {len(queries.get('styled_queries', []))} queries")
            
            # Show some example queries
            if queries.get("styled_queries"):
                print("\n📝 Sample queries generated:")
                for i, query in enumerate(queries["styled_queries"][:3], 1):
                    print(f"   {i}. {query.get('styled_query', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            return False
        
        print("\n🎉 All LLM components working successfully!")
        print(f"📊 Summary:")
        print(f"   • Category attributes: {len(intelligence.get('universal_attributes', {}))}")
        print(f"   • Customer archetypes: {len(archetypes.get('ranked_archetypes', []))}")
        print(f"   • Research queries: {len(queries.get('styled_queries', []))}")
        print(f"   • Avg authenticity: {queries.get('generation_metadata', {}).get('avg_authenticity', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_components_only())
    if success:
        print("\n✅ LLM components are ready for production!")
    else:
        print("\n❌ LLM components need attention before deployment")
        sys.exit(1)