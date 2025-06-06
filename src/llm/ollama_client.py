# src/llm/ollama_client.py
import httpx
import asyncio
import json
import re
import logging
from typing import Dict, List, Optional, AsyncGenerator, Union, Tuple
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available Ollama model types for different tasks"""
    GENERAL = "llama3.2"
    CREATIVE = "mistral"
    ANALYSIS = "codellama"
    FAST = "phi3"

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
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout))
        self._available_models: Optional[List[str]] = None
        self._last_health_check: Optional[float] = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Improved JSON extraction with multiple strategies."""
        if not content or not content.strip():
            raise json.JSONDecodeError("Empty content", content, 0)
        
        content = content.strip()
        
        # Strategy 0: Handle the specific meta-instruction pattern we're seeing
        # If the response starts with {"context": "customer psychology expert", "instructions":
        # This indicates the model is returning instructions instead of content
        if content.startswith('{"context":') or content.startswith('{"instructions":'):
            logger.warning("Model returned meta-instructions instead of content")
            raise json.JSONDecodeError("Meta-instructions detected, not actual content", content, 0)
        
        # Strategy 1: Try direct JSON parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for markdown code blocks with JSON
        markdown_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON objects in the text
        # Look for balanced braces
        def find_json_objects(text: str) -> List[str]:
            objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    brace_count = 1
                    start = i
                    i += 1
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        i += 1
                    if brace_count == 0:
                        candidate = text[start:i]
                        # Skip if it looks like meta-instructions
                        if not ('"context":' in candidate or '"instructions":' in candidate):
                            objects.append(candidate)
                else:
                    i += 1
            return objects
        
        json_candidates = find_json_objects(content)
        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to clean and parse the entire content
        # Remove common prefixes and suffixes
        cleaned = content
        prefixes_to_remove = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON response is:",
            "JSON:",
            "```json",
            "```"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing markdown
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Try to find and extract just the JSON part
        # Look for anything that starts with { and ends with }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = content[start_idx:end_idx + 1]
            # Skip if it looks like meta-instructions
            if not ('"context":' in candidate or '"instructions":' in candidate):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        
        logger.warning(f"Could not extract JSON from response. Content: {content[:200]}...")
        raise json.JSONDecodeError("No valid JSON object found", content, 0)

    async def generate_json_with_self_correction(
        self, request: LLMRequest, retries: int = 2
    ) -> Tuple[Dict[str, Any], int]:
        """Enhanced JSON generation with better self-correction."""
        total_tokens = 0
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                # Enhance the prompt to be more explicit about JSON requirements
                if attempt == 0:
                    enhanced_prompt = f"""{request.prompt}

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON
2. Do not include any explanatory text before or after the JSON
3. Do not use markdown code blocks (no ```)
4. Ensure all JSON is properly formatted with correct syntax
5. Use double quotes for all strings
6. End with a complete JSON object"""
                    
                    enhanced_request = LLMRequest(
                        prompt=enhanced_prompt,
                        model=request.model,
                        temperature=max(0.1, request.temperature - 0.2),  # Lower temperature for structured output
                        max_tokens=request.max_tokens,
                        system_prompt="You are a JSON generator. Always respond with valid JSON only.",
                        top_p=0.9,
                        top_k=40
                    )
                else:
                    # For retries, use the correction prompt
                    enhanced_request = request
                
                response = await self.generate(enhanced_request)
                total_tokens += response.tokens_used
                
                if not response.success:
                    last_error = f"LLM request failed: {response.error}"
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    continue

                # Try to extract JSON
                try:
                    parsed_json = self._extract_json_from_response(response.content)
                    logger.debug(f"Successfully parsed JSON on attempt {attempt + 1}")
                    return parsed_json, total_tokens
                except json.JSONDecodeError as e:
                    last_error = f"JSON parsing error: {e}"
                    logger.warning(f"Attempt {attempt + 1} JSON parse failed: {last_error}")
                    
                    if attempt < retries:
                        # Create a correction prompt
                        correction_prompt = f"""The previous response failed to parse as valid JSON.

ERROR: {last_error}
FAULTY RESPONSE: 
{response.content}

Please provide a corrected version that is ONLY valid JSON with no additional text.
Original request was: {request.prompt[:500]}...

Respond with ONLY the corrected JSON object."""
                        
                        request = LLMRequest(
                            prompt=correction_prompt,
                            model="llama3.2",  # Use a reliable model for correction
                            temperature=0.1,
                            max_tokens=request.max_tokens,
                            system_prompt="Fix the JSON. Respond only with valid JSON."
                        )
                    
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"Attempt {attempt + 1} failed with exception: {last_error}")
        
        logger.error(f"All JSON generation attempts failed. Last error: {last_error}")
        return {}, total_tokens
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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
    
    async def health_check(self) -> bool:
        """Check if Ollama is responsive with caching."""
        current_time = time.time()
        
        # Use cached result if recent
        if self._last_health_check and (current_time - self._last_health_check < 30):
            return True
        
        try:
            response = await self.client.get(f"{self.config.base_url}/", timeout=5.0)
            if response.status_code == 200:
                self._last_health_check = current_time
                return True
            return False
        except Exception:
            return False
    
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
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Test function
async def test_ollama_connection(config: Optional[OllamaConfig] = None) -> Dict:
    """Test Ollama connection and return status"""
    async with OllamaClient(config) as client:
        status = {
            "healthy": await client.health_check(),
            "models": [],
            "test_generation": None,
            "error": None
        }
        
        if status["healthy"]:
            try:
                status["models"] = await client.list_models()
                
                # Test basic generation
                test_request = LLMRequest(
                    prompt="Respond with exactly this JSON: {\"test\": \"success\", \"status\": \"working\"}",
                    temperature=0.1,
                    max_tokens=50
                )
                
                response = await client.generate(test_request)
                status["test_generation"] = {
                    "success": response.success,
                    "content": response.content,
                    "error": response.error,
                    "response_time": response.metadata.get("response_time") if response.metadata else None
                }
            except Exception as e:
                status["error"] = str(e)
        else:
            status["error"] = "Health check failed"
        
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