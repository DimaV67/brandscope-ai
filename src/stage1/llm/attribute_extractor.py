# src/stage1/llm/attribute_extractor.py
"""
LLM-powered attribute extractor with fixed import handling.
"""
import json
import re
# FIXED: Added 'Any' to the typing import
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from src.models.brand import BrandContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CategoryAnalysisPrompts:
    # ... (prompts are unchanged from the last version)
    BASE_ANALYSIS = """You are a market research expert...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    COMPETITIVE_LANDSCAPE = """Analyze the competitive landscape...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    UNIVERSAL_ATTRIBUTES_MAPPING = """Map the {category} category...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""


class LLMAttributeExtractor:
    """LLM-powered attribute extractor with robust fallback handling."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = CategoryAnalysisPrompts()
        self.total_tokens_used = 0
        self.mock_extractor = InternalMockExtractor()

    async def generate_category_intelligence(
        self, category: str, brand_context: BrandContext, customer_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("Generating category intelligence with LLM")
        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(category, brand_context, customer_narrative)
                
                # These now call the robust client method
                category_analysis = await self._analyze_category_characteristics(client, category, brand_context)
                competitive_analysis = await self._analyze_competitive_landscape(client, category, brand_context)
                universal_mapping = await self._map_universal_attributes(client, category, category_analysis)
                category_attributes = await self._generate_category_attributes(client, category)
                
                intelligence = self._combine_intelligence(
                    category, brand_context, category_analysis, competitive_analysis,
                    universal_mapping, category_attributes
                )
                logger.info("Category intelligence generated successfully")
                return intelligence
        except Exception as e:
            logger.error(f"LLM category intelligence generation failed: {e}", exc_info=True)
            return self._fallback_to_mock(category, brand_context, customer_narrative)

    async def _run_json_generation(self, client: OllamaClient, request: LLMRequest) -> Dict[str, Any]:
        """Helper to run self-correcting JSON generation and count tokens."""
        parsed_json, tokens_used = await client.generate_json_with_self_correction(request)
        self.total_tokens_used += tokens_used
        return parsed_json

    async def _analyze_category_characteristics(self, client: OllamaClient, category: str, brand_context: BrandContext) -> Dict[str, Any]:
        prompt = self.prompts.BASE_ANALYSIS.format(
            category=category, brand_name=brand_context.brand_name,
            competitors=", ".join(brand_context.competitive_context.primary_competitors),
            positioning=getattr(brand_context, 'brand_positioning', 'Not specified')
        )
        request = LLMRequest(prompt=prompt, model="codellama", temperature=0.2, max_tokens=1024, system_prompt="You are a market research expert.")
        return await self._run_json_generation(client, request)

    async def _analyze_competitive_landscape(self, client: OllamaClient, category: str, brand_context: BrandContext) -> Dict[str, Any]:
        prompt = self.prompts.COMPETITIVE_LANDSCAPE.format(
            category=category, brand_name=brand_context.brand_name,
            competitors=", ".join(brand_context.competitive_context.primary_competitors),
            positioning=getattr(brand_context, 'brand_positioning', 'Not specified')
        )
        request = LLMRequest(prompt=prompt, model="codellama", temperature=0.2, max_tokens=800, system_prompt="You are a competitive intelligence expert.")
        return await self._run_json_generation(client, request)

    async def _map_universal_attributes(self, client: OllamaClient, category: str, category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompts.UNIVERSAL_ATTRIBUTES_MAPPING.format(category=category)
        if category_analysis.get("customer_segments"):
            prompt += f"\n\nCategory Insights:\n{json.dumps(category_analysis.get('customer_segments', []), indent=2)}"
        request = LLMRequest(prompt=prompt, model="codellama", temperature=0.1, max_tokens=512, system_prompt="You are a customer psychology expert.")
        return await self._run_json_generation(client, request)

    async def _generate_category_attributes(self, client: OllamaClient, category: str) -> Dict[str, List[str]]:
        prompt = f"""Based on the {category} category analysis, generate 3-5 category-specific attributes in this format:
{{ "{category.upper()}_USE_CASE": ["primary_use", "secondary_use"], "{category.upper()}_EXPERTISE": ["beginner", "expert"] }}
IMPORTANT: Your entire response must be ONLY the valid JSON object described above, with no additional text, commentary, or markdown formatting."""
        request = LLMRequest(prompt=prompt, model="codellama", temperature=0.2, max_tokens=400, system_prompt="Generate category-specific behavioral attributes as valid JSON.")
        return await self._run_json_generation(client, request)

    def _combine_intelligence(self, category: str, brand_context: BrandContext, category_analysis: Dict[str, Any], competitive_analysis: Dict[str, Any], universal_mapping: Dict[str, Any], category_attributes: Dict[str, List[str]]) -> Dict[str, Any]:
        return {"category": category, "brand_context": brand_context.brand_name, "universal_attributes": universal_mapping.get("universal_attributes", {}), "category_attributes": category_attributes, "competitive_landscape": {"primary_competitors": brand_context.competitive_context.primary_competitors, "market_segments": competitive_analysis.get("market_segments", []), "positioning_opportunities": competitive_analysis.get("positioning_opportunities", [])}, "price_ranges": competitive_analysis.get("price_tiers", {}), "category_insights": {"characteristics": category_analysis.get("category_characteristics", {}), "customer_segments": category_analysis.get("customer_segments", []), "llm_generated": True}}
    
    def _fallback_to_mock(self, *args, **kwargs) -> Dict[str, Any]:
        logger.warning("LLM generation failed, using internal mock extractor.")
        return self.mock_extractor.generate_category_intelligence(*args, **kwargs)

class InternalMockExtractor:
    def generate_category_intelligence(self, *args, **kwargs) -> Dict[str, Any]:
        return {"category": "mock", "universal_attributes": {}, "category_attributes": {}, "competitive_landscape": {}, "price_ranges": {}, "category_insights": {"llm_generated": False, "mock_generated": True}}

# Factory function for backwards compatibility
def create_attribute_extractor(
    use_llm: bool = True,
    ollama_config: Optional[OllamaConfig] = None
) -> LLMAttributeExtractor:
    """Factory function to create attribute extractor."""
    return LLMAttributeExtractor(
        ollama_config=ollama_config,
        fallback_to_mock=True
    )