#!/usr/bin/env python3
"""
Simplified test to debug archetype generation specifically.
"""
import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.ollama_client import OllamaClient, LLMRequest
from src.models.brand import BrandContext, CompetitiveContext, ProductInfo, PriceTierEnum

async def test_simple_archetype_generation():
    """Test archetype generation with a very simple prompt."""
    print("🧪 Testing Simple Archetype Generation")
    print("=" * 50)
    
    # Test the simplest possible archetype generation
    simple_prompt = """Create 2 customer types for pistachios:

{
    "archetypes": [
        {
            "archetype_id": "ARCH_001",
            "name": "Premium Customer",
            "description": "Wants high quality pistachios"
        },
        {
            "archetype_id": "ARCH_002", 
            "name": "Budget Customer",
            "description": "Wants affordable pistachios"
        }
    ]
}"""
    
    async with OllamaClient() as client:
        print("🔍 Testing direct prompt...")
        
        request = LLMRequest(
            prompt=simple_prompt,
            model="llama3.2",
            temperature=0.1,
            max_tokens=500,
            system_prompt="Return only JSON. No explanations."
        )
        
        response = await client.generate(request)
        
        print(f"✅ Response received: {response.success}")
        print(f"📄 Content: {response.content[:200]}...")
        
        if response.success:
            try:
                # Try to extract JSON
                parsed = client._extract_json_from_response(response.content)
                print(f"✅ JSON parsed successfully: {json.dumps(parsed, indent=2)}")
                return True
            except Exception as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"📄 Full content: {response.content}")
                return False
        else:
            print(f"❌ LLM request failed: {response.error}")
            return False

async def test_step_by_step_generation():
    """Test each step of the archetype generation process."""
    print("\n🔬 Testing Step-by-Step Generation")
    print("=" * 50)
    
    # Create simple brand context
    brand_context = BrandContext(
        brand_name="Wonderful",
        products=[
            ProductInfo(
                name="Wonderful Pistachios",
                product_type="pistachios",
                price_tier=PriceTierEnum.PREMIUM,
                key_features=["roasted", "salted"]
            )
        ],
        competitive_context=CompetitiveContext(
            primary_competitors=["Blue Diamond", "Planters"]
        )
    )
    
    # Mock category intelligence
    category_intelligence = {
        "category": "pistachios",
        "universal_attributes": {
            "COREB1": ["HEALTH_FOCUSED", "QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
            "MODIFIERE1": ["SMART_SHOPPER", "HEALTH_IDENTITY", "STATUS_SIGNAL"],
            "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"]
        },
        "category_insights": {
            "customer_segments": [
                {"segment_name": "Health-Conscious", "size_percentage": 40},
                {"segment_name": "Premium Quality", "size_percentage": 35},
                {"segment_name": "Value Shoppers", "size_percentage": 25}
            ]
        }
    }
    
    # Test using the enhanced archetype builder
    try:
        from src.stage1.llm.archetype_builder import LLMArchetypeBuilder
        
        builder = LLMArchetypeBuilder()
        result = await builder.generate_archetypes(category_intelligence, brand_context)
        
        archetypes = result.get("ranked_archetypes", [])
        print(f"✅ Generated {len(archetypes)} archetypes")
        
        for i, archetype in enumerate(archetypes, 1):
            print(f"   {i}. {archetype.get('name', 'Unknown')}")
            print(f"      ID: {archetype.get('archetype_id', 'Unknown')}")
            print(f"      Description: {archetype.get('description', 'No description')[:100]}...")
        
        return len(archetypes) > 0
        
    except Exception as e:
        print(f"❌ Archetype builder failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_different_models():
    """Test archetype generation with different models."""
    print("\n🤖 Testing Different Models")
    print("=" * 50)
    
    simple_prompt = """Generate customer archetypes for pistachios. Return only this JSON:

{
    "archetypes": [
        {
            "archetype_id": "ARCH_001",
            "name": "Health-Focused Customer",
            "description": "Customer who buys pistachios for health benefits"
        }
    ]
}"""
    
    models = ["llama3.2:latest", "mistral:latest", "phi3:latest"]
    results = {}
    
    async with OllamaClient() as client:
        # Check available models first
        available_models = await client.list_models()
        available_names = [m["name"] for m in available_models]
        print(f"Available models: {available_names}")
        
        for model in models:
            if model not in available_names:
                print(f"⚠️  Skipping {model} - not available")
                continue
                
            print(f"\n🧪 Testing {model}...")
            
            request = LLMRequest(
                prompt=simple_prompt,
                model=model,
                temperature=0.1,
                max_tokens=500,
                system_prompt="Return JSON only."
            )
            
            try:
                response = await client.generate(request)
                
                if response.success:
                    try:
                        parsed = client._extract_json_from_response(response.content)
                        archetypes = parsed.get("archetypes", [])
                        results[model] = len(archetypes)
                        print(f"✅ {model}: Generated {len(archetypes)} archetypes")
                    except Exception as e:
                        results[model] = 0
                        print(f"❌ {model}: JSON parsing failed - {e}")
                        print(f"   Content: {response.content[:100]}...")
                else:
                    results[model] = 0
                    print(f"❌ {model}: Request failed - {response.error}")
                    
            except Exception as e:
                results[model] = 0
                print(f"💥 {model}: Exception - {e}")
    
    print(f"\n📊 Model Results: {results}")
    return any(count > 0 for count in results.values())

async def main():
    """Run all archetype-specific tests."""
    print("🎯 Archetype Generation Debug Tests")
    print("=" * 60)
    
    tests = [
        ("Simple Prompt", test_simple_archetype_generation),
        ("Step-by-Step", test_step_by_step_generation),
        ("Different Models", test_different_models)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🚀 Running {test_name} test...")
            results[test_name] = await test_func()
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    overall_success = any(results.values())
    
    if overall_success:
        print("\n🎉 At least one test passed! Archetype generation is working.")
    else:
        print("\n💥 All tests failed. There's a fundamental issue with archetype generation.")
        print("\n🔧 Debugging suggestions:")
        print("1. Check if Ollama models are working properly: ollama run llama3.2")
        print("2. Try a simpler prompt manually in Ollama")
        print("3. Check if there are model-specific issues")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)