# tests/test_ollama_integration.py
import asyncio
import json
import time
import sys
import os
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.ollama_client import (
    OllamaClient, 
    OllamaConfig, 
    LLMRequest,
    ModelManager,
    test_ollama_connection,
    quick_generate
)

console = Console()

class OllamaIntegrationTester:
    """Comprehensive Ollama integration testing"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.results = {}
    
    async def run_full_test_suite(self) -> Dict:
        """Run complete test suite for Ollama integration"""
        console.print("\n🚀 [bold blue]BrandScope Ollama Integration Test Suite[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Test 1: Basic Connection
            task1 = progress.add_task("Testing Ollama connection...", total=None)
            await self._test_connection()
            progress.update(task1, completed=True)
            
            # Test 2: Model Management
            task2 = progress.add_task("Testing model management...", total=None)
            await self._test_model_management()
            progress.update(task2, completed=True)
            
            # Test 3: Basic Generation
            task3 = progress.add_task("Testing basic generation...", total=None)
            await self._test_basic_generation()
            progress.update(task3, completed=True)
            
            # Test 4: Streaming
            task4 = progress.add_task("Testing streaming generation...", total=None)
            await self._test_streaming()
            progress.update(task4, completed=True)
            
            # Test 5: Brand Analysis Prompts
            task5 = progress.add_task("Testing brand analysis prompts...", total=None)
            await self._test_brand_analysis_prompts()
            progress.update(task5, completed=True)
            
            # Test 6: Performance Benchmarks
            task6 = progress.add_task("Running performance benchmarks...", total=None)
            await self._test_performance()
            progress.update(task6, completed=True)
        
        self._display_results()
        return self.results
    
    async def _test_connection(self):
        """Test basic Ollama connection"""
        try:
            status = await test_ollama_connection(self.config)
            self.results["connection"] = {
                "healthy": status["healthy"],
                "models_available": len(status["models"]),
                "test_generation_success": status.get("test_generation", {}).get("success", False),
                "response_time": status.get("test_generation", {}).get("response_time"),
                "error": None
            }
        except Exception as e:
            self.results["connection"] = {
                "healthy": False,
                "error": str(e)
            }
    
    async def _test_model_management(self):
        """Test model management capabilities"""
        try:
            async with OllamaClient(self.config) as client:
                manager = ModelManager(client)
                
                # List available models
                models = await client.list_models()
                
                # Test model info retrieval
                model_info = None
                if models:
                    model_info = await client.get_model_info(models[0]["name"])
                
                self.results["model_management"] = {
                    "models_found": len(models),
                    "model_names": [m["name"] for m in models],
                    "model_info_available": model_info is not None,
                    "recommended_setup_ready": len(models) >= 2,
                    "error": None
                }
        except Exception as e:
            self.results["model_management"] = {"error": str(e)}
    
    async def _test_basic_generation(self):
        """Test basic text generation"""
        test_cases = [
            {
                "name": "Simple Response",
                "prompt": "What is artificial intelligence in one sentence?",
                "expected_min_length": 10
            },
            {
                "name": "Structured Output",
                "prompt": "List 3 benefits of AI in JSON format with keys 'benefit' and 'description'",
                "expected_min_length": 50
            },
            {
                "name": "Creative Task",
                "prompt": "Write a creative tagline for a brand monitoring AI tool",
                "expected_min_length": 5
            }
        ]
        
        results = []
        
        try:
            async with OllamaClient(self.config) as client:
                for test_case in test_cases:
                    start_time = time.time()
                    
                    request = LLMRequest(
                        prompt=test_case["prompt"],
                        temperature=0.7,
                        max_tokens=200
                    )
                    
                    response = await client.generate(request)
                    end_time = time.time()
                    
                    results.append({
                        "name": test_case["name"],
                        "success": response.success,
                        "content_length": len(response.content) if response.success else 0,
                        "meets_min_length": len(response.content) >= test_case["expected_min_length"] if response.success else False,
                        "response_time": end_time - start_time,
                        "tokens_used": response.tokens_used,
                        "error": response.error
                    })
            
            self.results["basic_generation"] = {
                "test_cases": results,
                "success_rate": sum(1 for r in results if r["success"]) / len(results),
                "avg_response_time": sum(r["response_time"] for r in results) / len(results),
                "error": None
            }
            
        except Exception as e:
            self.results["basic_generation"] = {"error": str(e)}
    
    async def _test_streaming(self):
        """Test streaming generation"""
        try:
            async with OllamaClient(self.config) as client:
                request = LLMRequest(
                    prompt="Count from 1 to 10, one number per line.",
                    temperature=0.1,
                    stream=True
                )
                
                chunks_received = 0
                total_content = ""
                start_time = time.time()
                
                async for chunk in client.stream_generate(request):
                    if chunk.success:
                        chunks_received += 1
                        total_content += chunk.content
                    else:
                        break
                
                end_time = time.time()
                
                self.results["streaming"] = {
                    "chunks_received": chunks_received,
                    "total_content_length": len(total_content),
                    "streaming_worked": chunks_received > 1,
                    "total_time": end_time - start_time,
                    "error": None
                }
                
        except Exception as e:
            self.results["streaming"] = {"error": str(e)}
    
    async def _test_brand_analysis_prompts(self):
        """Test specific prompts for brand analysis tasks"""
        brand_prompts = [
            {
                "task": "attribute_extraction",
                "prompt": """Analyze the brand "Apple" and extract key attributes in this JSON format:
{
  "quality": {"score": 0-10, "reasoning": "explanation"},
  "innovation": {"score": 0-10, "reasoning": "explanation"},
  "trustworthiness": {"score": 0-10, "reasoning": "explanation"}
}"""
            },
            {
                "task": "archetype_analysis", 
                "prompt": """What customer archetype would be most interested in Apple products? 
Describe in 2-3 sentences focusing on demographics, psychographics, and motivations."""
            },
            {
                "task": "query_generation",
                "prompt": """Generate 3 different ways a potential customer might ask about Apple iPhones:
1. Direct question
2. Comparison question  
3. Problem-focused question"""
            }
        ]
        
        results = []
        
        try:
            async with OllamaClient(self.config) as client:
                for prompt_test in brand_prompts:
                    request = LLMRequest(
                        prompt=prompt_test["prompt"],
                        temperature=0.3,  # Lower temp for analytical tasks
                        max_tokens=300
                    )
                    
                    response = await client.generate(request)
                    
                    results.append({
                        "task": prompt_test["task"],
                        "success": response.success,
                        "response_quality": self._assess_response_quality(
                            prompt_test["task"], 
                            response.content if response.success else ""
                        ),
                        "content": response.content[:200] + "..." if response.success and len(response.content) > 200 else response.content,
                        "error": response.error
                    })
            
            self.results["brand_analysis"] = {
                "prompt_tests": results,
                "success_rate": sum(1 for r in results if r["success"]) / len(results),
                "avg_quality": sum(r["response_quality"] for r in results) / len(results),
                "error": None
            }
            
        except Exception as e:
            self.results["brand_analysis"] = {"error": str(e)}
    
    async def _test_performance(self):
        """Run performance benchmarks"""
        try:
            async with OllamaClient(self.config) as client:
                # Test concurrent requests
                concurrent_tasks = []
                for i in range(3):
                    request = LLMRequest(
                        prompt=f"Generate a short brand description for company {i+1}.",
                        temperature=0.7,
                        max_tokens=100
                    )
                    concurrent_tasks.append(client.generate(request))
                
                start_time = time.time()
                responses = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
                end_time = time.time()
                
                successful_responses = [r for r in responses if hasattr(r, 'success') and r.success]
                
                self.results["performance"] = {
                    "concurrent_requests": len(concurrent_tasks),
                    "successful_responses": len(successful_responses),
                    "total_time": end_time - start_time,
                    "avg_time_per_request": (end_time - start_time) / len(concurrent_tasks),
                    "concurrency_success_rate": len(successful_responses) / len(concurrent_tasks),
                    "error": None
                }
                
        except Exception as e:
            self.results["performance"] = {"error": str(e)}
    
    def _assess_response_quality(self, task: str, content: str) -> float:
        """Simple response quality assessment"""
        if not content:
            return 0.0
        
        quality_indicators = {
            "attribute_extraction": ["{", "}", "score", "reasoning"],
            "archetype_analysis": ["customer", "demographic", "psychographic"],
            "query_generation": ["1.", "2.", "3.", "?"]
        }
        
        indicators = quality_indicators.get(task, [])
        found_indicators = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        return min(found_indicators / len(indicators), 1.0) if indicators else 0.5
    
    def _display_results(self):
        """Display comprehensive test results"""
        console.print("\n📊 [bold green]Test Results Summary[/bold green]\n")
        
        # Connection Status
        conn = self.results.get("connection", {})
        if conn.get("healthy"):
            console.print("✅ [green]Ollama Connection: HEALTHY[/green]")
            if conn.get("response_time"):
                console.print(f"   Response time: {conn['response_time']:.2f}s")
        else:
            console.print("❌ [red]Ollama Connection: FAILED[/red]")
            if conn.get("error"):
                console.print(f"   Error: {conn['error']}")
        
        # Model Management
        models = self.results.get("model_management", {})
        if models.get("models_found", 0) > 0:
            console.print(f"✅ [green]Models Available: {models['models_found']}[/green]")
            console.print(f"   Models: {', '.join(models.get('model_names', [])[:3])}")
        else:
            console.print("⚠️  [yellow]No models found - run 'ollama pull llama3.2'[/yellow]")
        
        # Generation Tests
        gen = self.results.get("basic_generation", {})
        if gen.get("success_rate", 0) > 0.8:
            console.print(f"✅ [green]Generation Success Rate: {gen['success_rate']:.1%}[/green]")
            console.print(f"   Avg response time: {gen.get('avg_response_time', 0):.2f}s")
        else:
            console.print(f"⚠️  [yellow]Generation Success Rate: {gen.get('success_rate', 0):.1%}[/yellow]")
        
        # Brand Analysis
        brand = self.results.get("brand_analysis", {})
        if brand.get("success_rate", 0) > 0.8:
            console.print(f"✅ [green]Brand Analysis: {brand['success_rate']:.1%} success[/green]")
            console.print(f"   Avg quality score: {brand.get('avg_quality', 0):.1%}")
        else:
            console.print(f"⚠️  [yellow]Brand Analysis needs tuning[/yellow]")
        
        # Streaming
        stream = self.results.get("streaming", {})
        if stream.get("streaming_worked"):
            console.print("✅ [green]Streaming: WORKING[/green]")
        else:
            console.print("❌ [red]Streaming: FAILED[/red]")
        
        # Performance
        perf = self.results.get("performance", {})
        if perf.get("concurrency_success_rate", 0) > 0.8:
            console.print(f"✅ [green]Concurrency: {perf['concurrency_success_rate']:.1%}[/green]")
        else:
            console.print(f"⚠️  [yellow]Concurrency issues detected[/yellow]")
        
        # Recommendations
        console.print("\n💡 [bold blue]Recommendations:[/bold blue]")
        
        if not conn.get("healthy"):
            console.print("   • Start Ollama: 'ollama serve'")
        
        if models.get("models_found", 0) < 2:
            console.print("   • Install models: 'ollama pull llama3.2' and 'ollama pull mistral'")
        
        if gen.get("avg_response_time", 0) > 10:
            console.print("   • Consider lighter models for faster responses")
        
        if brand.get("avg_quality", 0) < 0.7:
            console.print("   • Prompt engineering needed for brand analysis tasks")
        
        console.print("\n🎯 [bold green]Ready for Phase 2 integration![/bold green]")

async def main():
    """Run the test suite"""
    tester = OllamaIntegrationTester()
    results = await tester.run_full_test_suite()
    
    # Save results for analysis
    with open("ollama_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n📝 Results saved to: ollama_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())