#!/usr/bin/env python3
"""
Test script to validate Stage 1 fixes.
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.ollama_client import OllamaClient, LLMRequest, test_ollama_connection
from src.stage1.llm.attribute_extractor import LLMAttributeExtractor
from src.stage1.llm.archetype_builder import LLMArchetypeBuilder
from src.stage1.llm.query_generator import LLMQueryGenerator
from src.models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

async def test_ollama_connection_robust():
    """Test Ollama connection with robust error handling."""
    print("🔍 Testing Ollama connection...")
    
    try:
        status = await test_ollama_connection()
        
        if status["healthy"]:
            print("✅ Ollama is healthy")
            print(f"   Available models: {len(status.get('models', []))}")
            
            if status.get("test_generation"):
                test_gen = status["test_generation"]
                if test_gen.get("success"):
                    print(f"✅ Test generation successful")
                    print(f"   Response: {test_gen.get('content', '')[:100]}...")
                else:
                    print(f"❌ Test generation failed: {test_gen.get('error')}")
            
            return True
        else:
            print(f"❌ Ollama health check failed: {status.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"💥 Connection test failed: {e}")
        return False

async def test_json_parsing():
    """Test improved JSON parsing."""
    print("\n🧪 Testing JSON parsing improvements...")
    
    test_cases = [
        '{"test": "success"}',  # Clean JSON
        'Here is the JSON: {"test": "success"}',  # With prefix
        '```json\n{"test": "success"}\n```',  # Markdown format
        'The response is: {"test": "success"} - hope this helps!',  # With suffix
        '{"test": "partial"',  # Broken JSON - should fail gracefully
    ]
    
    async with OllamaClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            try:
                parsed = client._extract_json_from_response(test_case)
                print(f"✅ Test {i}: Parsed successfully - {parsed}")
            except Exception as e:
                print(f"❌ Test {i}: Failed to parse - {e}")

async def test_attribute_extractor():
    """Test enhanced attribute extractor."""
    print("\n🏗️  Testing Attribute Extractor...")
    
    # Create test brand context
    brand_context = BrandContext(
        brand_name="Wonderful",
        products=[
            ProductInfo(
                name="Wonderful Pistachios",
                product_type="pistachios",
                price_tier=PriceTierEnum.PREMIUM,
                key_features=["roasted", "salted", "premium quality"]
            )
        ],
        competitive_context=CompetitiveContext(
            primary_competitors=["Blue Diamond", "Planters", "Private Label"]
        ),
        brand_positioning="Premium pistachio brand focused on quality and taste"
    )
    
    extractor = LLMAttributeExtractor()
    
    try:
        result = await extractor.generate_category_intelligence(
            category="pistachios",
            brand_context=brand_context
        )
        
        print(f"✅ Attribute extraction successful")
        print(f"   Category: {result.get('category')}")
        print(f"   Universal attributes: {len(result.get('universal_attributes', {}))}")
        print(f"   Customer segments: {len(result.get('category_insights', {}).get('customer_segments', []))}")
        print(f"   LLM generated: {result.get('category_insights', {}).get('llm_generated', False)}")
        print(f"   Tokens used: {extractor.total_tokens_used}")
        
        return result
        
    except Exception as e:
        print(f"❌ Attribute extraction failed: {e}")
        return None

async def test_archetype_builder(category_intelligence):
    """Test enhanced archetype builder."""
    print("\n👥 Testing Archetype Builder...")
    
    if not category_intelligence:
        print("⚠️  Skipping archetype test - no category intelligence")
        return None
    
    brand_context = BrandContext(
        brand_name="Wonderful",
        products=[
            ProductInfo(
                name="Wonderful Pistachios",
                product_type="pistachios", 
                price_tier=PriceTierEnum.PREMIUM,
                key_features=["roasted", "salted", "premium quality"]
            )
        ],
        competitive_context=CompetitiveContext(
            primary_competitors=["Blue Diamond", "Planters", "Private Label"]
        ),
        brand_positioning="Premium pistachio brand focused on quality and taste"
    )
    
    builder = LLMArchetypeBuilder()
    
    try:
        result = await builder.generate_archetypes(
            category_intelligence=category_intelligence,
            brand_context=brand_context
        )
        
        archetypes = result.get("ranked_archetypes", [])
        top_archetypes = result.get("top_archetypes", [])
        
        print(f"✅ Archetype generation successful")
        print(f"   Total archetypes: {len(archetypes)}")
        print(f"   Top archetypes: {len(top_archetypes)}")
        print(f"   LLM generated: {result.get('generation_metadata', {}).get('llm_generated', False)}")
        print(f"   Avg confidence: {result.get('generation_metadata', {}).get('avg_confidence', 0):.2f}")
        print(f"   Tokens used: {builder.total_tokens_used}")
        
        if archetypes:
            print(f"   Sample archetype: {archetypes[0].get('name', 'Unknown')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Archetype generation failed: {e}")
        return None

async def test_query_generator(archetypes, category_intelligence):
    """Test enhanced query generator."""
    print("\n❓ Testing Query Generator...")
    
    if not archetypes or not category_intelligence:
        print("⚠️  Skipping query test - missing dependencies")
        return None
    
    brand_context = BrandContext(
        brand_name="Wonderful",
        products=[
            ProductInfo(
                name="Wonderful Pistachios",
                product_type="pistachios",
                price_tier=PriceTierEnum.PREMIUM, 
                key_features=["roasted", "salted", "premium quality"]
            )
        ],
        competitive_context=CompetitiveContext(
            primary_competitors=["Blue Diamond", "Planters", "Private Label"]
        ),
        brand_positioning="Premium pistachio brand focused on quality and taste"
    )
    
    generator = LLMQueryGenerator()
    
    try:
        result = await generator.generate_query_package(
            top_archetypes=archetypes.get("top_archetypes", []),
            category_intelligence=category_intelligence,
            brand_context=brand_context
        )
        
        queries = result.get("styled_queries", [])
        metadata = result.get("generation_metadata", {})
        
        print(f"✅ Query generation successful")
        print(f"   Total queries: {len(queries)}")
        print(f"   Avg authenticity: {metadata.get('avg_authenticity', 0):.2f}")
        print(f"   Categories covered: {len(metadata.get('categories_covered', []))}")
        print(f"   LLM generated: {metadata.get('llm_generated', False)}")
        print(f"   Tokens used: {generator.total_tokens_used}")
        
        if queries:
            print(f"   Sample query: {queries[0].get('styled_query', 'Unknown')[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Query generation failed: {e}")
        return None

async def test_full_pipeline():
    """Test complete Stage 1 pipeline."""
    print("\n🚀 Testing Full Pipeline Integration...")
    
    try:
        # Test each component
        category_intelligence = await test_attribute_extractor()
        archetypes = await test_archetype_builder(category_intelligence)
        queries = await test_query_generator(archetypes, category_intelligence)
        
        # Validate pipeline results
        success_count = 0
        total_components = 3
        
        if category_intelligence and category_intelligence.get('category_insights'):
            success_count += 1
            print("   ✅ Category intelligence: PASS")
        else:
            print("   ❌ Category intelligence: FAIL")
        
        if archetypes and len(archetypes.get('ranked_archetypes', [])) > 0:
            success_count += 1
            print("   ✅ Archetype generation: PASS")
        else:
            print("   ❌ Archetype generation: FAIL")
        
        if queries and len(queries.get('styled_queries', [])) > 0:
            success_count += 1
            print("   ✅ Query generation: PASS")
        else:
            print("   ❌ Query generation: FAIL")
        
        success_rate = success_count / total_components
        print(f"\n📊 Pipeline Success Rate: {success_rate:.1%} ({success_count}/{total_components})")
        
        if success_rate >= 0.67:  # At least 2/3 components working
            print("🎉 Pipeline integration: SUCCESSFUL")
            return True
        else:
            print("💥 Pipeline integration: FAILED")
            return False
            
    except Exception as e:
        print(f"💥 Pipeline test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 BrandScope Stage 1 Fix Validation")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Track results
    test_results = {}
    
    # Test 1: Ollama Connection
    test_results["ollama"] = await test_ollama_connection_robust()
    
    # Test 2: JSON Parsing
    if test_results["ollama"]:
        await test_json_parsing()
    
    # Test 3: Full Pipeline
    test_results["pipeline"] = await test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.title():15} {status}")
    
    overall_success = all(test_results.values())
    
    if overall_success:
        print("\n🎉 All tests PASSED! Stage 1 fixes are working correctly.")
        print("\n📋 Next Steps:")
        print("1. Deploy the fixed components to your main codebase")
        print("2. Run the actual Stage 1 pipeline with: python -m src.main")
        print("3. Check the generated files in stage1_outputs/")
    else:
        print("\n💥 Some tests FAILED. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        if not test_results.get("ollama"):
            print("- Ensure Ollama is running: ollama serve")
            print("- Check if models are installed: ollama list")
            print("- Try pulling llama3.2: ollama pull llama3.2")
        
        print("- Check logs in logs/ directory for detailed error information")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)