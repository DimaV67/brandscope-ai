# src/stage1/llm/attribute_extractor.py
"""
LLM-powered attribute extractor replacing mock implementation.
"""
import json
import sys
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from models.brand import BrandContext
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CategoryAnalysisPrompts:
    """Structured prompts for category analysis"""
    
    BASE_ANALYSIS = """You are a market research expert analyzing the {category} category. 
    
Brand Context:
- Brand: {brand_name}
- Category: {category}
- Primary Competitors: {competitors}
- Positioning: {positioning}

Analyze this category and provide insights in the following JSON format:
{{
    "category_characteristics": {{
        "market_maturity": "emerging|growing|mature|declining",
        "purchase_frequency": "daily|weekly|monthly|yearly",
        "decision_complexity": "low|medium|high",
        "brand_importance": "low|medium|high"
    }},
    "customer_segments": [
        {{
            "segment_name": "descriptive name",
            "size_percentage": 0-100,
            "key_motivations": ["motivation1", "motivation2"],
            "price_sensitivity": "low|medium|high"
        }}
    ],
    "category_specific_attributes": [
        {{
            "attribute_code": "CATEGORY_SPECIFIC_CODE",
            "attribute_name": "Human readable name",
            "values": ["value1", "value2", "value3"],
            "importance": "low|medium|high"
        }}
    ]
}}

Focus on actionable insights that would help understand customer behavior patterns."""
    
    COMPETITIVE_LANDSCAPE = """Analyze the competitive landscape for {category} with focus on {brand_name}.

Competitors: {competitors}
Brand Positioning: {positioning}

Provide analysis in this JSON format:
{{
    "market_segments": [
        {{
            "segment": "segment_name", 
            "leaders": ["brand1", "brand2"],
            "positioning_gap": "opportunity description"
        }}
    ],
    "price_tiers": {{
        "BUDGET": "price_range_description",
        "MIDRANGE": "price_range_description", 
        "PREMIUM": "price_range_description"
    }},
    "positioning_opportunities": [
        {{
            "opportunity": "specific_positioning",
            "target_segment": "customer_segment",
            "competitive_advantage": "advantage_description"
        }}
    ]
}}"""

    UNIVERSAL_ATTRIBUTES_MAPPING = """Map the {category} category to our universal customer attribute framework.

Universal Framework:
- COREB1 (Core Behavior): HEALTH_FOCUSED, QUALITY_CONNOISSEUR, BUSY_PRACTICAL
- MODIFIERE1 (Identity): STATUS_SIGNAL, HEALTH_IDENTITY, SMART_SHOPPER  
- MODIFIERD3 (Price Tier): BUDGET, MIDRANGE, PREMIUM
- COREA2 (Shopping Style): RESEARCH, QUICK_STORE, ROUTINE_ONLINE
- DEMOD2 (Lifestyle): URBAN_FAST, SUBURBAN_FAMILY, HEALTH_CONSCIOUS_REGION
- COREB3 (Brand Preference): BRAND_AWARE, BRAND_NEUTRAL, BRAND_BLIND

Return JSON mapping category insights to universal attributes:
{{
    "universal_attributes": {{
        "COREB1": ["primary_behavior", "secondary_behavior"],
        "MODIFIERE1": ["primary_identity", "secondary_identity"],
        "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
        "COREA2": ["primary_shopping", "secondary_shopping"],
        "DEMOD2": ["primary_lifestyle", "secondary_lifestyle"],
        "COREB3": ["primary_brand_pref", "secondary_brand_pref"]
    }}
}}"""

class LLMAttributeExtractor:
    """LLM-powered attribute extractor with fallback to mock for development."""
    
    def __init__(
        self, 
        ollama_config: Optional[OllamaConfig] = None,
        fallback_to_mock: bool = True
    ):
        self.ollama_config = ollama_config or OllamaConfig()
        self.fallback_to_mock = fallback_to_mock
        self.prompts = CategoryAnalysisPrompts()
        
        # Load mock extractor for fallback
        if fallback_to_mock:
            try:
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from attribute_extractor import AttributeExtractor as MockExtractor
                self.mock_extractor = MockExtractor()
            except ImportError as e:
                logger.warning(f"Could not import mock extractor: {e}")
                self.mock_extractor = None
                self.fallback_to_mock = False  # Disable fallback if import fails
    
    async def generate_category_intelligence(
        self,
        category: str,
        brand_context: BrandContext,
        customer_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate category intelligence using LLM analysis."""
        
        logger.info("Generating category intelligence with LLM",
                   metadata={"category": category, "brand": brand_context.brand_name})
        
        try:
            async with OllamaClient(self.ollama_config) as client:
                # Check if Ollama is available
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(category, brand_context, customer_narrative)
                
                # Step 1: Base category analysis
                category_analysis = await self._analyze_category_characteristics(
                    client, category, brand_context
                )
                
                # Step 2: Competitive landscape analysis  
                competitive_analysis = await self._analyze_competitive_landscape(
                    client, category, brand_context
                )
                
                # Step 3: Universal attributes mapping
                universal_mapping = await self._map_universal_attributes(
                    client, category, category_analysis
                )
                
                # Step 4: Generate category-specific attributes
                category_attributes = await self._generate_category_attributes(
                    client, category, category_analysis
                )
                
                # Combine all analyses
                intelligence = self._combine_intelligence(
                    category,
                    brand_context,
                    category_analysis,
                    competitive_analysis,
                    universal_mapping,
                    category_attributes
                )
                
                logger.info("Category intelligence generated successfully",
                           metadata={"category": category, "attributes_count": len(intelligence.get("universal_attributes", {}))})
                
                return intelligence
                
        except Exception as e:
            logger.error(f"LLM category intelligence generation failed: {e}")
            if self.fallback_to_mock:
                logger.info("Falling back to mock implementation")
                return self._fallback_to_mock(category, brand_context, customer_narrative)
            raise
    
    async def _analyze_category_characteristics(
        self,
        client: OllamaClient,
        category: str,
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Analyze fundamental category characteristics."""
        
        prompt = self.prompts.BASE_ANALYSIS.format(
            category=category,
            brand_name=brand_context.brand_name,
            competitors=", ".join(brand_context.competitive_context.primary_competitors),
            positioning=getattr(brand_context, 'brand_positioning', 'Not specified')
        )
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for analytical tasks
            max_tokens=800,
            system_prompt="You are a market research expert. Always respond with valid JSON."
        )
        
        response = await client.generate(request)
        
        if not response.success:
            raise RuntimeError(f"Category analysis failed: {response.error}")
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed, attempting to extract: {e}")
            return self._extract_json_from_response(response.content)
    
    async def _analyze_competitive_landscape(
        self,
        client: OllamaClient, 
        category: str,
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Analyze competitive positioning and opportunities."""
        
        prompt = self.prompts.COMPETITIVE_LANDSCAPE.format(
            category=category,
            brand_name=brand_context.brand_name,
            competitors=", ".join(brand_context.competitive_context.primary_competitors),
            positioning=getattr(brand_context, 'brand_positioning', 'Not specified')
        )
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.2,
            max_tokens=600,
            system_prompt="You are a competitive intelligence expert. Always respond with valid JSON."
        )
        
        response = await client.generate(request)
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._extract_json_from_response(response.content)
        
        # Fallback competitive analysis
        return {
            "market_segments": [
                {"segment": "premium", "leaders": brand_context.competitive_context.primary_competitors[:2], "positioning_gap": "Quality differentiation"},
                {"segment": "mainstream", "leaders": brand_context.competitive_context.primary_competitors, "positioning_gap": "Value proposition"},
                {"segment": "budget", "leaders": ["Generic brands"], "positioning_gap": "Accessibility"}
            ],
            "price_tiers": {
                "BUDGET": "Entry-level pricing",
                "MIDRANGE": "Competitive pricing", 
                "PREMIUM": "Premium pricing"
            },
            "positioning_opportunities": [
                {"opportunity": "quality_leader", "target_segment": "quality_focused", "competitive_advantage": "Superior quality"}
            ]
        }
    
    async def _map_universal_attributes(
        self,
        client: OllamaClient,
        category: str,
        category_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map category insights to universal attribute framework."""
        
        prompt = self.prompts.UNIVERSAL_ATTRIBUTES_MAPPING.format(
            category=category
        )
        
        # Add category context to prompt
        if category_analysis.get("customer_segments"):
            prompt += f"\n\nCategory Insights:\n{json.dumps(category_analysis['customer_segments'], indent=2)}"
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.1,  # Very low temperature for structured mapping
            max_tokens=400,
            system_prompt="You are a customer psychology expert. Map categories to universal behavioral attributes with valid JSON."
        )
        
        response = await client.generate(request)
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._extract_json_from_response(response.content)
        
        # Fallback universal mapping
        return {
            "universal_attributes": {
                "COREB1": ["QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
                "MODIFIERE1": ["SMART_SHOPPER", "STATUS_SIGNAL"],
                "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
                "COREA2": ["RESEARCH", "QUICK_STORE"],
                "DEMOD2": ["URBAN_FAST", "SUBURBAN_FAMILY"],
                "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL"]
            }
        }
    
    async def _generate_category_attributes(
        self,
        client: OllamaClient,
        category: str,
        category_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate category-specific attribute codes."""
        
        prompt = f"""Based on the {category} category analysis, generate 3-5 category-specific attributes in this format:

{{
    "{category.upper()}_USE_CASE": ["primary_use", "secondary_use", "specialized_use"],
    "{category.upper()}_EXPERTISE": ["beginner", "intermediate", "expert"],
    "{category.upper()}_PRIORITY": ["function", "design", "price", "brand"],
    "{category.upper()}_FREQUENCY": ["daily", "weekly", "occasional"]
}}

Make attributes relevant to actual {category} customer behavior."""
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.2,
            max_tokens=300,
            system_prompt="Generate category-specific behavioral attributes as valid JSON."
        )
        
        response = await client.generate(request)
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                pass
        
        # Fallback category attributes
        return {
            f"{category.upper()}_USE_CASE": ["primary_use", "secondary_use"],
            f"{category.upper()}_EXPERTISE": ["beginner", "intermediate", "expert"], 
            f"{category.upper()}_PRIORITY": ["function", "design", "price"]
        }
    
    def _combine_intelligence(
        self,
        category: str,
        brand_context: BrandContext,
        category_analysis: Dict[str, Any],
        competitive_analysis: Dict[str, Any],
        universal_mapping: Dict[str, Any],
        category_attributes: Dict[str, List[str]]
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
                "llm_generated": True
            }
        }
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response that may contain extra text."""
        # Find JSON-like content between braces
        import re
        
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logger.warning("Could not extract JSON from LLM response")
        return {}
    
    def _fallback_to_mock(
        self,
        category: str,
        brand_context: BrandContext,
        customer_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        if hasattr(self, 'mock_extractor') and self.mock_extractor:
            return self.mock_extractor.generate_category_intelligence(
                category, brand_context, customer_narrative
            )
        else:
            # Emergency fallback when mock import fails
            logger.warning("Using emergency fallback - mock extractor not available")
            return {
                "category": category,
                "brand_context": brand_context.brand_name,
                "universal_attributes": {
                    "COREB1": ["QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
                    "MODIFIERE1": ["SMART_SHOPPER", "STATUS_SIGNAL"],
                    "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
                    "COREA2": ["RESEARCH", "QUICK_STORE"],
                    "DEMOD2": ["URBAN_FAST", "SUBURBAN_FAMILY"],
                    "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL"]
                },
                "category_attributes": {
                    f"{category.upper()}_USE_CASE": ["primary_use", "secondary_use"],
                    f"{category.upper()}_EXPERTISE": ["beginner", "expert"],
                    f"{category.upper()}_PRIORITY": ["function", "price"]
                },
                "competitive_landscape": {
                    "primary_competitors": brand_context.competitive_context.primary_competitors,
                    "market_segments": ["premium", "mainstream", "budget"],
                    "positioning_opportunities": ["quality_leader", "value_leader"]
                },
                "price_ranges": {
                    "BUDGET": "< $100",
                    "MIDRANGE": "$100 - $300",
                    "PREMIUM": "> $300"
                },
                "emergency_fallback": True
            }

# Convenience function for backwards compatibility
def create_attribute_extractor(
    use_llm: bool = True,
    ollama_config: Optional[OllamaConfig] = None
) -> LLMAttributeExtractor:
    """Factory function to create attribute extractor."""
    return LLMAttributeExtractor(
        ollama_config=ollama_config,
        fallback_to_mock=True
    )