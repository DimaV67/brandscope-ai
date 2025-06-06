# src/stage1/llm/attribute_extractor.py
"""
Enhanced LLM-powered attribute extractor with robust error handling.
"""
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from src.models.brand import BrandContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CategoryAnalysisPrompts:
    """Structured prompts for category analysis with strict JSON requirements"""
    
    BASE_ANALYSIS = """Analyze the {category} market for brand {brand_name}.

Brand Information:
- Name: {brand_name}
- Competitors: {competitors}
- Positioning: {positioning}

Respond with ONLY this JSON structure (no additional text):
{{
    "category_characteristics": {{
        "market_maturity": "growing",
        "purchase_frequency": "monthly",
        "decision_complexity": "medium",
        "brand_importance": "high"
    }},
    "customer_segments": [
        {{
            "segment_name": "Health-Conscious Consumers",
            "size_percentage": 35,
            "key_motivations": ["nutrition", "quality"],
            "price_sensitivity": "medium"
        }},
        {{
            "segment_name": "Convenience Shoppers",
            "size_percentage": 45,
            "key_motivations": ["convenience", "taste"],
            "price_sensitivity": "high"
        }},
        {{
            "segment_name": "Premium Quality Seekers",
            "size_percentage": 20,
            "key_motivations": ["quality", "brand"],
            "price_sensitivity": "low"
        }}
    ],
    "category_specific_attributes": [
        {{
            "attribute_code": "{category_upper}_QUALITY",
            "attribute_name": "Quality Level",
            "values": ["premium", "standard", "budget"],
            "importance": "high"
        }},
        {{
            "attribute_code": "{category_upper}_USAGE",
            "attribute_name": "Usage Pattern",
            "values": ["daily", "occasional", "special"],
            "importance": "medium"
        }}
    ]
}}"""

    COMPETITIVE_LANDSCAPE = """Analyze the competitive landscape for {category} market.

Brand: {brand_name}
Competitors: {competitors}
Positioning: {positioning}

Respond with ONLY this JSON structure:
{{
    "market_segments": [
        {{
            "segment": "premium",
            "leaders": ["{first_competitor}", "{second_competitor}"],
            "positioning_gap": "ultra-premium organic positioning"
        }},
        {{
            "segment": "mainstream",
            "leaders": ["{competitors_list}"],
            "positioning_gap": "value-quality balance"
        }},
        {{
            "segment": "budget",
            "leaders": ["store brands", "generic options"],
            "positioning_gap": "affordable quality"
        }}
    ],
    "price_tiers": {{
        "BUDGET": "Under $5 per serving",
        "MIDRANGE": "$5-15 per serving",
        "PREMIUM": "Over $15 per serving"
    }},
    "positioning_opportunities": [
        {{
            "opportunity": "health-focused premium",
            "target_segment": "health-conscious consumers",
            "competitive_advantage": "superior nutritional profile"
        }},
        {{
            "opportunity": "convenient quality",
            "target_segment": "busy professionals",
            "competitive_advantage": "premium quality with convenience"
        }}
    ]
}}"""

    UNIVERSAL_ATTRIBUTES_MAPPING = """Map {category} category to our behavioral framework.

Universal Attributes Framework:
- COREB1: HEALTH_FOCUSED, QUALITY_CONNOISSEUR, BUSY_PRACTICAL
- MODIFIERE1: STATUS_SIGNAL, HEALTH_IDENTITY, SMART_SHOPPER
- MODIFIERD3: BUDGET, MIDRANGE, PREMIUM
- COREA2: RESEARCH, QUICK_STORE, ROUTINE_ONLINE
- DEMOD2: URBAN_FAST, SUBURBAN_FAMILY, HEALTH_CONSCIOUS_REGION
- COREB3: BRAND_AWARE, BRAND_NEUTRAL, BRAND_BLIND

Respond with ONLY this JSON:
{{
    "universal_attributes": {{
        "COREB1": ["HEALTH_FOCUSED", "QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
        "MODIFIERE1": ["HEALTH_IDENTITY", "SMART_SHOPPER", "STATUS_SIGNAL"],
        "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
        "COREA2": ["RESEARCH", "QUICK_STORE", "ROUTINE_ONLINE"],
        "DEMOD2": ["HEALTH_CONSCIOUS_REGION", "URBAN_FAST", "SUBURBAN_FAMILY"],
        "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL", "BRAND_BLIND"]
    }}
}}"""

class LLMAttributeExtractor:
    """Enhanced LLM-powered attribute extractor with robust fallback handling."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = CategoryAnalysisPrompts()
        self.total_tokens_used = 0
        self.mock_extractor = InternalMockExtractor()

    async def generate_category_intelligence(
        self, category: str, brand_context: BrandContext, customer_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate category intelligence using LLM analysis."""
        
        logger.info("Generating category intelligence with LLM",
                   metadata={"category": category, "brand": brand_context.brand_name})
        
        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(category, brand_context, customer_narrative)
                
                logger.info("Step 1/4: Analyzing category characteristics...")
                category_analysis = await self._analyze_category_characteristics(client, category, brand_context)
                
                if not category_analysis or not category_analysis.get("customer_segments"):
                    logger.error("Failed to generate customer segments in Step 1. Aborting intelligence generation.")
                    return self._fallback_to_mock(category, brand_context, customer_narrative)
                
                logger.info("Step 2/4: Analyzing competitive landscape...")
                competitive_analysis = await self._analyze_competitive_landscape(client, category, brand_context)
                
                logger.info("Step 3/4: Mapping universal attributes...")
                universal_mapping = await self._map_universal_attributes(client, category, category_analysis)
                
                logger.info("Step 4/4: Generating category attributes...")
                category_attributes = await self._generate_category_attributes(client, category)
                
                intelligence = self._combine_intelligence(
                    category, brand_context, category_analysis, competitive_analysis,
                    universal_mapping, category_attributes
                )
                
                logger.info("Category intelligence generated successfully",
                           metadata={
                               "segments_count": len(intelligence.get("category_insights", {}).get("customer_segments", [])),
                               "attributes_count": len(intelligence.get("universal_attributes", {})),
                               "total_tokens": self.total_tokens_used
                           })
                
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
        """Analyze fundamental category characteristics."""
        
        competitors = ", ".join(brand_context.competitive_context.primary_competitors) if brand_context.competitive_context.primary_competitors else "various market players"
        
        prompt = self.prompts.BASE_ANALYSIS.format(
            category=category,
            brand_name=brand_context.brand_name,
            competitors=competitors,
            positioning=getattr(brand_context, 'brand_positioning', 'Market competitor'),
            category_upper=category.upper().replace(' ', '_')
        )
        
        request = LLMRequest(
            prompt=prompt,
            model="llama3.2",
            temperature=0.3,
            max_tokens=1200,
            system_prompt="You are a market research expert. Respond only with valid JSON."
        )
        
        return await self._run_json_generation(client, request)

    async def _analyze_competitive_landscape(self, client: OllamaClient, category: str, brand_context: BrandContext) -> Dict[str, Any]:
        """Analyze competitive positioning and opportunities."""
        
        competitors = brand_context.competitive_context.primary_competitors
        competitors_list = ", ".join(competitors) if competitors else "various competitors"
        first_competitor = competitors[0] if competitors else "leading brand"
        second_competitor = competitors[1] if len(competitors) > 1 else "major competitor"
        
        prompt = self.prompts.COMPETITIVE_LANDSCAPE.format(
            category=category,
            brand_name=brand_context.brand_name,
            competitors=competitors_list,
            positioning=getattr(brand_context, 'brand_positioning', 'Market competitor'),
            first_competitor=first_competitor,
            second_competitor=second_competitor,
            competitors_list=competitors_list
        )
        
        request = LLMRequest(
            prompt=prompt,
            model="llama3.2",
            temperature=0.2,
            max_tokens=800,
            system_prompt="You are a competitive intelligence expert. Respond only with valid JSON."
        )
        
        result = await self._run_json_generation(client, request)
        
        # Provide fallback if generation fails
        if not result:
            return {
                "market_segments": [
                    {"segment": "premium", "leaders": competitors[:2] if len(competitors) >= 2 else ["premium brands"], "positioning_gap": "ultra-premium positioning"},
                    {"segment": "mainstream", "leaders": competitors if competitors else ["market leaders"], "positioning_gap": "value-quality balance"}
                ],
                "price_tiers": {"BUDGET": "Entry level", "MIDRANGE": "Competitive", "PREMIUM": "Premium"},
                "positioning_opportunities": [
                    {"opportunity": "quality_leader", "target_segment": "quality_focused", "competitive_advantage": "Superior quality"}
                ]
            }
        
        return result

    async def _map_universal_attributes(self, client: OllamaClient, category: str, category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Map category insights to universal attribute framework."""
        
        prompt = self.prompts.UNIVERSAL_ATTRIBUTES_MAPPING.format(category=category)
        
        request = LLMRequest(
            prompt=prompt,
            model="llama3.2",
            temperature=0.1,
            max_tokens=400,
            system_prompt="Map category to behavioral attributes. Respond only with valid JSON."
        )
        
        result = await self._run_json_generation(client, request)
        
        # Provide fallback mapping
        if not result:
            return {
                "universal_attributes": {
                    "COREB1": ["HEALTH_FOCUSED", "QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
                    "MODIFIERE1": ["SMART_SHOPPER", "HEALTH_IDENTITY", "STATUS_SIGNAL"],
                    "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
                    "COREA2": ["RESEARCH", "QUICK_STORE", "ROUTINE_ONLINE"],
                    "DEMOD2": ["URBAN_FAST", "SUBURBAN_FAMILY", "HEALTH_CONSCIOUS_REGION"],
                    "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL", "BRAND_BLIND"]
                }
            }
        
        return result

    async def _generate_category_attributes(self, client: OllamaClient, category: str) -> Dict[str, List[str]]:
        """Generate category-specific attribute codes."""
        
        category_upper = category.upper().replace(' ', '_')
        
        prompt = f"""Generate category-specific attributes for {category}.

Respond with ONLY this JSON:
{{
    "{category_upper}_USE_CASE": ["primary_use", "secondary_use", "special_occasion"],
    "{category_upper}_EXPERTISE": ["beginner", "enthusiast", "expert"],
    "{category_upper}_PRIORITY": ["function", "design", "price", "brand"],
    "{category_upper}_FREQUENCY": ["daily", "weekly", "monthly", "occasional"]
}}"""
        
        request = LLMRequest(
            prompt=prompt,
            model="llama3.2",
            temperature=0.2,
            max_tokens=300,
            system_prompt="Generate category attributes as valid JSON."
        )
        
        result = await self._run_json_generation(client, request)
        
        # Provide fallback attributes
        if not result:
            return {
                f"{category_upper}_USE_CASE": ["primary_use", "secondary_use"],
                f"{category_upper}_EXPERTISE": ["beginner", "intermediate", "expert"],
                f"{category_upper}_PRIORITY": ["function", "design", "price"]
            }
        
        return result

    def _combine_intelligence(
        self, category: str, brand_context: BrandContext, category_analysis: Dict[str, Any],
        competitive_analysis: Dict[str, Any], universal_mapping: Dict[str, Any], category_attributes: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Combine all analyses into final intelligence structure."""
        
        return {
            "category": category,
            "brand_context": brand_context.brand_name,
            "universal_attributes": universal_mapping.get("universal_attributes", {}),
            "category_attributes": category_attributes,
            "competitive_landscape": {
                "primary_competitors": brand_context.competitive_context.primary_competitors,
                "market_segments": competitive_analysis.get("market_segments", []),
                "positioning_opportunities": competitive_analysis.get("positioning_opportunities", [])
            },
            "price_ranges": competitive_analysis.get("price_tiers", {
                "BUDGET": "Entry-level pricing",
                "MIDRANGE": "Competitive pricing", 
                "PREMIUM": "Premium pricing"
            }),
            "category_insights": {
                "characteristics": category_analysis.get("category_characteristics", {}),
                "customer_segments": category_analysis.get("customer_segments", []),
                "llm_generated": True,
                "token_usage": self.total_tokens_used
            }
        }
    
    def _fallback_to_mock(self, category: str, brand_context: BrandContext, customer_narrative: Optional[str] = None) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        logger.warning("LLM generation failed, using internal mock extractor.")
        return self.mock_extractor.generate_category_intelligence(category, brand_context, customer_narrative)

class InternalMockExtractor:
    """Internal mock extractor for fallback scenarios."""
    
    def generate_category_intelligence(self, category: str, brand_context: BrandContext, customer_narrative: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock category intelligence."""
        
        category_upper = category.upper().replace(' ', '_')
        competitors = brand_context.competitive_context.primary_competitors
        
        return {
            "category": category,
            "brand_context": brand_context.brand_name,
            "universal_attributes": {
                "COREB1": ["HEALTH_FOCUSED", "QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
                "MODIFIERE1": ["SMART_SHOPPER", "HEALTH_IDENTITY", "STATUS_SIGNAL"],
                "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
                "COREA2": ["RESEARCH", "QUICK_STORE", "ROUTINE_ONLINE"],
                "DEMOD2": ["URBAN_FAST", "SUBURBAN_FAMILY", "HEALTH_CONSCIOUS_REGION"],
                "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL", "BRAND_BLIND"]
            },
            "category_attributes": {
                f"{category_upper}_USE_CASE": ["primary_use", "secondary_use"],
                f"{category_upper}_EXPERTISE": ["beginner", "enthusiast", "expert"],
                f"{category_upper}_PRIORITY": ["function", "design", "price"]
            },
            "competitive_landscape": {
                "primary_competitors": competitors,
                "market_segments": [
                    {"segment": "premium", "leaders": competitors[:2] if len(competitors) >= 2 else ["premium brands"], "positioning_gap": "ultra-premium positioning"},
                    {"segment": "mainstream", "leaders": competitors if competitors else ["market leaders"], "positioning_gap": "value-quality balance"},
                    {"segment": "budget", "leaders": ["store brands"], "positioning_gap": "affordable quality"}
                ],
                "positioning_opportunities": [
                    {"opportunity": "quality_leader", "target_segment": "quality_focused", "competitive_advantage": "Superior quality"},
                    {"opportunity": "value_leader", "target_segment": "price_conscious", "competitive_advantage": "Best value proposition"}
                ]
            },
            "price_ranges": {
                "BUDGET": "< $100",
                "MIDRANGE": "$100 - $300", 
                "PREMIUM": "> $300"
            },
            "category_insights": {
                "characteristics": {
                    "market_maturity": "growing",
                    "purchase_frequency": "monthly",
                    "decision_complexity": "medium",
                    "brand_importance": "high"
                },
                "customer_segments": [
                    {
                        "segment_name": "Health-Conscious Consumers",
                        "size_percentage": 35,
                        "key_motivations": ["nutrition", "quality"],
                        "price_sensitivity": "medium"
                    },
                    {
                        "segment_name": "Convenience Shoppers", 
                        "size_percentage": 45,
                        "key_motivations": ["convenience", "taste"],
                        "price_sensitivity": "high"
                    },
                    {
                        "segment_name": "Premium Quality Seekers",
                        "size_percentage": 20,
                        "key_motivations": ["quality", "brand"],
                        "price_sensitivity": "low"
                    }
                ],
                "llm_generated": False,
                "mock_generated": True
            }
        }

# Factory function for backwards compatibility
def create_attribute_extractor(use_llm: bool = True, ollama_config: Optional[OllamaConfig] = None) -> LLMAttributeExtractor:
    """Factory function to create attribute extractor."""
    return LLMAttributeExtractor(ollama_config=ollama_config)