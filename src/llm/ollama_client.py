# src/llm/ollama_client.py
import httpx
import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available Ollama model types for different tasks"""
    GENERAL = "llama3.2"  # General purpose reasoning
    CREATIVE = "mistral"  # Creative content generation
    ANALYSIS = "codellama"  # Analytical tasks
    FAST = "phi3"  # Quick responses

@dataclass
class OllamaConfig:
    """Ollama configuration settings"""
    host: str = "localhost"
    port: int = 11434
    timeout: int = 300
    max_retries: int = 3
    default_model: str = "llama3.2"
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

@dataclass
class LLMRequest:
    """Standardized LLM request format"""
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: Optional[str] = None
    stream: bool = False
    
    def to_ollama_format(self, default_model: str) -> Dict:
        """Convert to Ollama API format"""
        payload = {
            "model": self.model or default_model,
            "prompt": self.prompt,
            "stream": self.stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty
            }
        }
        
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
            
        if self.system_prompt:
            payload["system"] = self.system_prompt
            
        return payload

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    model: str
    provider: str = "ollama"
    usage: Optional[Dict] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    @property
    def tokens_used(self) -> int:
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0

class OllamaClient:
    """Production-ready Ollama client with comprehensive error handling"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        self._available_models: Optional[List[str]] = None
        self._last_health_check: Optional[float] = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion with automatic retry logic"""
        start_time = time.time()
        
        try:
            payload = request.to_ollama_format(self.config.default_model)
            
            logger.debug(f"Sending request to Ollama: {payload['model']}")
            
            response = await self.client.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            end_time = time.time()
            
            return LLMResponse(
                content=result.get("response", ""),
                model=result.get("model", payload["model"]),
                provider="ollama",
                usage={
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                metadata={
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                    "eval_duration": result.get("eval_duration", 0),
                    "response_time": end_time - start_time,
                    "done": result.get("done", True)
                }
            )
            
        except httpx.RequestError as e:
            logger.error(f"Ollama request failed: {e}")
            return LLMResponse(
                content="",
                model=request.model or self.config.default_model,
                error=f"Request failed: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            return LLMResponse(
                content="",
                model=request.model or self.config.default_model,
                error=f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                content="",
                model=request.model or self.config.default_model,
                error=f"Unexpected error: {str(e)}"
            )
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[LLMResponse, None]:
        """Stream generation with proper async handling"""
        try:
            request.stream = True
            payload = request.to_ollama_format(self.config.default_model)
            
            logger.debug(f"Starting stream from Ollama: {payload['model']}")
            
            async with self.client.stream(
                "POST",
                f"{self.config.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            
                            if chunk.get("response"):
                                yield LLMResponse(
                                    content=chunk["response"],
                                    model=chunk.get("model", payload["model"]),
                                    provider="ollama",
                                    metadata={
                                        "done": chunk.get("done", False),
                                        "is_stream": True
                                    }
                                )
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield LLMResponse(
                content="",
                model=request.model or self.config.default_model,
                error=f"Stream failed: {str(e)}"
            )
    
    async def list_models(self) -> List[Dict]:
        """Get available models with caching"""
        try:
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            self._available_models = [model["name"] for model in models_data.get("models", [])]
            
            return models_data.get("models", [])
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        try:
            response = await self.client.post(
                f"{self.config.base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if Ollama is responsive with caching"""
        current_time = time.time()
        
        # Cache health check for 30 seconds
        if (self._last_health_check and 
            current_time - self._last_health_check < 30):
            return True
        
        try:
            response = await self.client.get(
                f"{self.config.base_url}/api/tags",
                timeout=10.0
            )
            
            is_healthy = response.status_code == 200
            if is_healthy:
                self._last_health_check = current_time
            
            return is_healthy
            
        except Exception:
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = await self.client.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name},
                timeout=600.0  # 10 minutes for model download
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary"""
        if self._available_models is None:
            await self.list_models()
        
        if model_name in (self._available_models or []):
            return True
        
        logger.info(f"Model {model_name} not found locally, attempting to pull...")
        return await self.pull_model(model_name)
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Convenience functions for common operations
async def test_ollama_connection(config: Optional[OllamaConfig] = None) -> Dict:
    """Test Ollama connection and return status"""
    async with OllamaClient(config) as client:
        status = {
            "healthy": await client.health_check(),
            "models": [],
            "test_generation": None
        }
        
        if status["healthy"]:
            status["models"] = await client.list_models()
            
            # Test basic generation
            test_request = LLMRequest(
                prompt="Hello! Respond with exactly: 'Ollama connection successful'",
                temperature=0.1,
                max_tokens=10
            )
            
            response = await client.generate(test_request)
            status["test_generation"] = {
                "success": response.success,
                "content": response.content,
                "error": response.error,
                "response_time": response.metadata.get("response_time") if response.metadata else None
            }
        
        return status

async def quick_generate(
    prompt: str, 
    model: str = "llama3.2",
    temperature: float = 0.7,
    config: Optional[OllamaConfig] = None
) -> str:
    """Quick generation utility function"""
    async with OllamaClient(config) as client:
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature
        )
        
        response = await client.generate(request)
        
        if response.success:
            return response.content
        else:
            raise RuntimeError(f"Generation failed: {response.error}")

# Model management utilities
class ModelManager:
    """Utility class for managing Ollama models"""
    
    RECOMMENDED_MODELS = {
        "general": "llama3.2",
        "creative": "mistral",
        "analysis": "codellama", 
        "fast": "phi3",
        "reasoning": "llama3.2:70b"  # If you have enough RAM
    }
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    async def setup_recommended_models(self) -> Dict[str, bool]:
        """Download recommended models for different tasks"""
        results = {}
        
        for task, model in self.RECOMMENDED_MODELS.items():
            logger.info(f"Setting up {task} model: {model}")
            results[task] = await self.client.ensure_model_available(model)
        
        return results
    
    async def get_optimal_model(self, task_type: str = "general") -> str:
        """Get the best available model for a task"""
        recommended = self.RECOMMENDED_MODELS.get(task_type, "llama3.2")
        
        if await self.client.ensure_model_available(recommended):
            return recommended
        
        # Fallback to any available model
        models = await self.client.list_models()
        if models:
            return models[0]["name"]
        
        raise RuntimeError("No models available")