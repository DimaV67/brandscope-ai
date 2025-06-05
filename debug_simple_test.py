#!/usr/bin/env python3
"""
Debug version to find the exact error location.
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

async def debug_archetype_builder():
    """Debug the archetype builder specifically"""
    
    try:
        from llm.ollama_client import OllamaConfig
        from stage1.llm.attribute_extractor import LLMAttributeExtractor
        from stage1.llm.archetype_builder import LLMArchetypeBuilder
        from models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum
        
        print("🔍 Debug: Testing Archetype Builder Issue")
        
        # Create minimal test data
        sample_product = ProductInfo(
            name="TestPhone",
            product_type="smartphones",
            price_tier=PriceTierEnum.MIDRANGE,
            price_range="$300 - $600",
            key_features=["5G", "Camera"]
        )
        
        brand_context = BrandContext(
            brand_name="TestBrand",
            products=[sample_product],
            competitive_context=CompetitiveContext(
                primary_competitors=["Apple", "Samsung"],
                market_position="challenger"
            ),
            brand_positioning="Innovation at affordable prices",
            target_markets=["tech users"]
        )
        
        print("✅ BrandContext created")
        print(f"   brand_name: {brand_context.brand_name}")
        print(f"   brand_positioning: {brand_context.brand_positioning}")
        print(f"   Available attributes: {dir(brand_context)}")
        
        # Test attribute extractor first
        extractor = LLMAttributeExtractor(fallback_to_mock=True)
        intelligence = await extractor.generate_category_intelligence("smartphones", brand_context)
        print("✅ Attribute extractor worked")
        
        # Now test archetype builder with detailed error catching
        print("\n🔍 Testing archetype builder...")
        builder = LLMArchetypeBuilder(fallback_to_mock=True)
        
        try:
            archetypes = await builder.generate_archetypes(intelligence, brand_context)
            print("✅ Archetype builder worked!")
            print(f"   Generated {len(archetypes.get('ranked_archetypes', []))} archetypes")
        except Exception as e:
            print(f"❌ Archetype builder failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Let's try to see where exactly it fails
            print("\n🔍 Debugging archetype builder internals...")
            
            # Test health check
            from llm.ollama_client import OllamaClient
            async with OllamaClient() as client:
                health = await client.health_check()
                print(f"   Ollama health: {health}")
                
                if health:
                    # Try to run the first step manually
                    try:
                        result = await builder._generate_initial_archetypes(client, intelligence, brand_context)
                        print(f"   Initial archetypes step: ✅ ({len(result)} archetypes)")
                    except Exception as e2:
                        print(f"   Initial archetypes step: ❌ {e2}")
                        traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_archetype_builder())