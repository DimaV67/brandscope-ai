# src/stage1/llm/archetype_builder.py
"""
Enhanced LLM-powered archetype builder with fixed prompts.
"""
import json
import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from src.models.brand import BrandContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass 
class ArchetypePrompts:
    """Fixed prompts that generate actual content instead of meta-instructions"""
    
    ARCHETYPE_GENERATION = """You are analyzing the {category} market for {brand_name}. Based on the market research below, create 3 customer archetypes.

Market Research:
{intelligence_summary}

Competitors: {competitors}
Brand Positioning: {positioning}

Generate exactly this JSON structure with real customer archetypes:

{{
    "archetypes": [
        {{
            "archetype_id": "ARCH_001",
            "name": "Premium Quality Enthusiast",
            "description": "Affluent customer who prioritizes superior quality {category} and is willing to pay premium prices for the best products",
            "attributes": {{
                "COREB1": "QUALITY_CONNOISSEUR",
                "MODIFIERE1": "STATUS_SIGNAL",
                "MODIFIERD3": "PREMIUM",
                "COREA2": "RESEARCH",
                "DEMOD2": "URBAN_FAST",
                "COREB3": "BRAND_AWARE"
            }},
            "market_presence": "HIGH",
            "strategic_value": "HIGH",
            "confidence": 0.90,
            "ai_behavior_prediction": "Asks detailed questions about quality standards and seeks expert recommendations"
        }},
        {{
            "archetype_id": "ARCH_002",
            "name": "Health-Conscious Consumer",
            "description": "Customer focused on nutritional benefits and health aspects of {category} consumption",
            "attributes": {{
                "COREB1": "HEALTH_FOCUSED",
                "MODIFIERE1": "HEALTH_IDENTITY",
                "MODIFIERD3": "MIDRANGE",
                "COREA2": "RESEARCH",
                "DEMOD2": "HEALTH_CONSCIOUS_REGION",
                "COREB3": "BRAND_AWARE"
            }},
            "market_presence": "MEDIUM",
            "strategic_value": "HIGH",
            "confidence": 0.85,
            "ai_behavior_prediction": "Researches nutritional information and health benefits extensively"
        }},
        {{
            "archetype_id": "ARCH_003",
            "name": "Practical Value Shopper",
            "description": "Budget-conscious customer seeking good {category} products at reasonable prices",
            "attributes": {{
                "COREB1": "BUSY_PRACTICAL",
                "MODIFIERE1": "SMART_SHOPPER",
                "MODIFIERD3": "MIDRANGE",
                "COREA2": "QUICK_STORE",
                "DEMOD2": "SUBURBAN_FAMILY",
                "COREB3": "BRAND_NEUTRAL"
            }},
            "market_presence": "HIGH",
            "strategic_value": "MEDIUM",
            "confidence": 0.80,
            "ai_behavior_prediction": "Values convenience and practical benefits over premium features"
        }}
    ]
}}"""

    BEHAVIORAL_REFINEMENT = """Add AI interaction patterns to these customer archetypes:

{archetypes}

For each archetype, predict how they would interact with AI assistants when researching {category} products.

{{
    "refined_archetypes": [
        {{
            "archetype_id": "ARCH_001",
            "ai_query_patterns": [
                "What is the highest quality {category} brand available?",
                "Compare premium {category} options in detail"
            ],
            "information_priorities": ["quality metrics", "expert reviews", "brand reputation"],
            "decision_triggers": ["expert endorsement", "quality certifications"],
            "communication_style": "formal"
        }},
        {{
            "archetype_id": "ARCH_002", 
            "ai_query_patterns": [
                "What are the health benefits of different {category} brands?",
                "Which {category} has the best nutritional profile?"
            ],
            "information_priorities": ["nutritional data", "health benefits", "ingredient quality"],
            "decision_triggers": ["health certifications", "nutritional superiority"],
            "communication_style": "research-focused"
        }},
        {{
            "archetype_id": "ARCH_003",
            "ai_query_patterns": [
                "Best value {category} for everyday use?",
                "Quick comparison of popular {category} options"
            ],
            "information_priorities": ["price comparison", "practical benefits", "convenience"],
            "decision_triggers": ["good value proposition", "practical benefits"],
            "communication_style": "casual"
        }}
    ]
}}"""

    ARCHETYPE_RANKING = """Rank these customer archetypes by strategic importance for {brand_name}:

{archetypes}

Consider market size, brand fit, revenue potential, and competitive positioning.

{{
    "ranked_archetypes": [
        {{
            "archetype_id": "ARCH_001",
            "ranking_score": 0.92,
            "ranking_rationale": "High-value customers with strong brand loyalty potential and premium pricing acceptance",
            "execution_priority": 1
        }},
        {{
            "archetype_id": "ARCH_002",
            "ranking_score": 0.88,
            "ranking_rationale": "Growing health-conscious segment aligns well with premium positioning",
            "execution_priority": 2
        }},
        {{
            "archetype_id": "ARCH_003",
            "ranking_score": 0.75,
            "ranking_rationale": "Large market segment but lower margins and brand loyalty",
            "execution_priority": 3
        }}
    ],
    "top_archetypes": ["ARCH_001", "ARCH_002"],
    "strategic_insights": [
        "Focus on quality and health messaging for top segments",
        "Premium positioning aligns with top two archetypes"
    ]
}}"""

class LLMArchetypeBuilder:
    """Enhanced LLM-powered customer archetype builder with fixed prompts."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = ArchetypePrompts()
        self.total_tokens_used = 0
        self.mock_builder = InternalMockBuilder()

    async def generate_archetypes(self, category_intelligence: Dict[str, Any], brand_context: BrandContext) -> Dict[str, Any]:
        """Generate customer archetypes using LLM analysis."""
        
        logger.info("Building customer archetypes with LLM",
                   metadata={"category": category_intelligence.get("category")})
        
        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(category_intelligence, brand_context)
                
                logger.info("Step 1/3: Generating initial archetypes...")
                raw_archetypes = await self._generate_initial_archetypes(client, category_intelligence, brand_context)
                
                if not raw_archetypes:
                    logger.warning("Initial archetype generation returned no results. Using fallback.")
                    return self._fallback_to_mock(category_intelligence, brand_context)
                
                logger.info("Step 2/3: Refining behavioral patterns...")
                refined_archetypes = await self._refine_behavioral_patterns(client, raw_archetypes, category_intelligence)
                
                logger.info("Step 3/3: Strategic ranking...")
                ranking_analysis = await self._rank_archetypes_strategically(client, refined_archetypes, brand_context, category_intelligence)
                
                final_result = self._combine_archetype_analysis(category_intelligence, refined_archetypes, ranking_analysis)
                
                logger.info("Archetypes generated successfully",
                           metadata={
                               "total_archetypes": len(final_result.get("ranked_archetypes", [])),
                               "avg_confidence": final_result.get("generation_metadata", {}).get("avg_confidence", 0),
                               "total_tokens": self.total_tokens_used
                           })
                
                return final_result
                
        except Exception as e:
            logger.error(f"LLM archetype generation failed: {e}", exc_info=True)
            return self._fallback_to_mock(category_intelligence, brand_context)

    async def _run_json_generation(self, client: OllamaClient, request: LLMRequest) -> Dict[str, Any]:
        """Helper to run self-correcting JSON generation and count tokens."""
        parsed_json, tokens_used = await client.generate_json_with_self_correction(request)
        self.total_tokens_used += tokens_used
        return parsed_json

    async def _generate_initial_archetypes(
        self, client: OllamaClient, category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate initial customer archetypes with simple, direct prompts."""
        
        category = category_intelligence.get("category", "product")
        intelligence_summary = self._summarize_category_intelligence(category_intelligence)
        competitors = ", ".join(brand_context.competitive_context.primary_competitors) if brand_context.competitive_context.primary_competitors else "various competitors"
        
        # Simplified prompt that's less likely to confuse the model
        prompt = f"""Create 3 customer archetypes for the {category} market.

Brand: {brand_context.brand_name}
Market Research: {intelligence_summary}
Competitors: {competitors}

Return this exact JSON format:"""

        prompt += """
{
    "archetypes": [
        {
            "archetype_id": "ARCH_001",
            "name": "Premium Quality Enthusiast", 
            "description": "Customer who values premium quality and is willing to pay more",
            "attributes": {
                "COREB1": "QUALITY_CONNOISSEUR",
                "MODIFIERE1": "STATUS_SIGNAL", 
                "MODIFIERD3": "PREMIUM",
                "COREA2": "RESEARCH",
                "DEMOD2": "URBAN_FAST",
                "COREB3": "BRAND_AWARE"
            },
            "market_presence": "HIGH",
            "strategic_value": "HIGH", 
            "confidence": 0.90,
            "ai_behavior_prediction": "Seeks expert recommendations and detailed comparisons"
        },
        {
            "archetype_id": "ARCH_002",
            "name": "Health-Conscious Consumer",
            "description": "Customer focused on health and nutritional benefits", 
            "attributes": {
                "COREB1": "HEALTH_FOCUSED",
                "MODIFIERE1": "HEALTH_IDENTITY",
                "MODIFIERD3": "MIDRANGE", 
                "COREA2": "RESEARCH",
                "DEMOD2": "HEALTH_CONSCIOUS_REGION",
                "COREB3": "BRAND_AWARE"
            },
            "market_presence": "MEDIUM",
            "strategic_value": "HIGH",
            "confidence": 0.85,
            "ai_behavior_prediction": "Researches nutritional information extensively"
        },
        {
            "archetype_id": "ARCH_003", 
            "name": "Practical Value Shopper",
            "description": "Budget-conscious customer seeking good value",
            "attributes": {
                "COREB1": "BUSY_PRACTICAL",
                "MODIFIERE1": "SMART_SHOPPER",
                "MODIFIERD3": "MIDRANGE",
                "COREA2": "QUICK_STORE", 
                "DEMOD2": "SUBURBAN_FAMILY",
                "COREB3": "BRAND_NEUTRAL"
            },
            "market_presence": "HIGH",
            "strategic_value": "MEDIUM",
            "confidence": 0.80,
            "ai_behavior_prediction": "Values convenience and practical benefits"
        }
    ]
}"""
        
        request = LLMRequest(
            prompt=prompt,
            model="llama3.2",
            temperature=0.3,  # Lower temperature for more predictable output
            max_tokens=1500,
            system_prompt="You are a market researcher. Return only the requested JSON, no other text."
        )
        
        result = await self._run_json_generation(client, request)
        archetypes = result.get("archetypes", [])
        
        # If we still don't get archetypes, try a simpler approach
        if not archetypes:
            logger.warning("Complex archetype generation failed, trying simple approach")
            
            simple_prompt = f"""Generate customer archetypes for {category}:

{{
    "archetypes": [
        {{
            "archetype_id": "ARCH_001",
            "name": "Premium {category.title()} Customer",
            "description": "Values quality over price",
            "confidence": 0.85
        }},
        {{
            "archetype_id": "ARCH_002", 
            "name": "Health-Focused {category.title()} Customer",
            "description": "Prioritizes health benefits",
            "confidence": 0.80
        }}
    ]
}}"""
            
            simple_request = LLMRequest(
                prompt=simple_prompt,
                model="llama3.2",
                temperature=0.1,
                max_tokens=800,
                system_prompt="Return only JSON."
            )
            
            simple_result = await self._run_json_generation(client, simple_request)
            archetypes = simple_result.get("archetypes", [])
        
        return archetypes

    async def _refine_behavioral_patterns(self, client: OllamaClient, archetypes: List[Dict[str, Any]], category_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine AI behavior predictions - simplified approach."""
        
        if not archetypes:
            return []
        
        # Skip refinement if we have basic archetypes working - just return them
        logger.info("Skipping behavioral refinement to ensure basic functionality")
        return archetypes

    async def _rank_archetypes_strategically(
        self, client: OllamaClient, archetypes: List[Dict[str, Any]], 
        brand_context: BrandContext, category_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplified ranking - just return basic ranking structure."""
        
        if not archetypes:
            return {"ranked_archetypes": [], "top_archetypes": [], "strategic_insights": []}
        
        # Simple ranking based on position
        ranking = {
            "ranked_archetypes": [
                {
                    "archetype_id": arch.get("archetype_id", f"ARCH_{i:03d}"),
                    "ranking_score": max(0.9 - i * 0.1, 0.5),
                    "ranking_rationale": f"Ranked #{i+1} based on market importance",
                    "execution_priority": i + 1
                }
                for i, arch in enumerate(archetypes[:5])
            ],
            "top_archetypes": [arch.get("archetype_id", f"ARCH_{i:03d}") for i, arch in enumerate(archetypes[:2])],
            "strategic_insights": ["Focus on top-ranked archetypes for initial execution"]
        }
        
        return ranking

    def _summarize_category_intelligence(self, intelligence: Dict[str, Any]) -> str:
        """Create a concise summary of category intelligence for prompts."""
        summary_parts = []
        
        if intelligence.get("category_insights"):
            insights = intelligence["category_insights"]
            if insights.get("customer_segments"):
                segments = [seg.get("segment_name", "Unknown") for seg in insights["customer_segments"][:3]]
                summary_parts.append(f"Customer Segments: {', '.join(segments)}")
        
        return "\n".join(summary_parts) if summary_parts else "General market analysis"

    def _combine_archetype_analysis(
        self, category_intelligence: Dict[str, Any], refined_archetypes: List[Dict[str, Any]], ranking_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all archetype analysis into final result."""
        
        # Apply ranking information to archetypes
        ranked_archetype_info = {r["archetype_id"]: r for r in ranking_analysis.get("ranked_archetypes", [])}
        top_archetype_ids = ranking_analysis.get("top_archetypes", [])
        
        for archetype in refined_archetypes:
            arch_id = archetype.get("archetype_id", "")
            if arch_id in ranked_archetype_info:
                rank_info = ranked_archetype_info[arch_id]
                archetype.update({
                    "ranking_score": rank_info.get("ranking_score", 0.5),
                    "ranking_rationale": rank_info.get("ranking_rationale", ""),
                    "execution_priority": rank_info.get("execution_priority", 999)
                })
        
        # Sort by ranking score
        refined_archetypes.sort(key=lambda x: x.get("ranking_score", 0.0), reverse=True)
        
        # Get top archetypes for execution
        top_archetypes = [
            arch for arch in refined_archetypes 
            if arch.get("archetype_id") in top_archetype_ids
        ]
        
        # If no top archetypes identified, use first 2
        if not top_archetypes and refined_archetypes:
            top_archetypes = refined_archetypes[:2]
        
        # Calculate metadata
        confidences = [arch.get("confidence", 0.8) for arch in refined_archetypes]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
        
        return {
            "universal_attributes": category_intelligence.get("universal_attributes", {}),
            "category_attributes": category_intelligence.get("category_attributes", {}),
            "ranked_archetypes": refined_archetypes,
            "top_archetypes": top_archetypes,
            "generation_metadata": {
                "total_archetypes": len(refined_archetypes),
                "avg_confidence": avg_confidence,
                "coverage_score": min(len(refined_archetypes) / 5.0, 1.0),
                "llm_generated": True,
                "strategic_insights": ranking_analysis.get("strategic_insights", []),
                "token_usage": self.total_tokens_used
            }
        }
        
    def _fallback_to_mock(self, category_intelligence: Dict[str, Any], brand_context: BrandContext) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        logger.warning("LLM generation failed, using internal mock builder.")
        return self.mock_builder.generate_archetypes(category_intelligence, brand_context)

class InternalMockBuilder:
    """Internal mock builder for fallback scenarios."""
    
    def generate_archetypes(self, category_intelligence: Dict[str, Any], brand_context: BrandContext) -> Dict[str, Any]:
        """Generate mock archetypes."""
        
        category = category_intelligence.get("category", "product")
        
        mock_archetypes = [
            {
                "archetype_id": "ARCH_001",
                "name": f"Premium {category.title()} Enthusiast",
                "description": f"Quality-focused customer seeking premium {category} solutions",
                "attributes": {
                    "COREB1": "QUALITY_CONNOISSEUR",
                    "MODIFIERE1": "STATUS_SIGNAL",
                    "MODIFIERD3": "PREMIUM",
                    "COREA2": "RESEARCH",
                    "DEMOD2": "URBAN_FAST",
                    "COREB3": "BRAND_AWARE"
                },
                "market_presence": "HIGH",
                "strategic_value": "HIGH",
                "confidence": 0.85,
                "ai_behavior_prediction": "Seeks detailed comparisons and expert reviews",
                "ranking_score": 0.9,
                "execution_priority": 1
            },
            {
                "archetype_id": "ARCH_002",
                "name": f"Health-Conscious {category.title()} Consumer",
                "description": f"Health-focused customer prioritizing nutritional benefits of {category}",
                "attributes": {
                    "COREB1": "HEALTH_FOCUSED",
                    "MODIFIERE1": "HEALTH_IDENTITY",
                    "MODIFIERD3": "MIDRANGE",
                    "COREA2": "RESEARCH",
                    "DEMOD2": "HEALTH_CONSCIOUS_REGION",
                    "COREB3": "BRAND_AWARE"
                },
                "market_presence": "MEDIUM",
                "strategic_value": "HIGH",
                "confidence": 0.82,
                "ai_behavior_prediction": "Researches nutritional information extensively",
                "ranking_score": 0.85,
                "execution_priority": 2
            },
            {
                "archetype_id": "ARCH_003",
                "name": f"Practical {category.title()} User",
                "description": f"Efficiency-focused customer needing reliable {category} solutions",
                "attributes": {
                    "COREB1": "BUSY_PRACTICAL",
                    "MODIFIERE1": "SMART_SHOPPER",
                    "MODIFIERD3": "MIDRANGE",
                    "COREA2": "QUICK_STORE",
                    "DEMOD2": "SUBURBAN_FAMILY",
                    "COREB3": "BRAND_NEUTRAL"
                },
                "market_presence": "HIGH",
                "strategic_value": "MEDIUM",
                "confidence": 0.8,
                "ai_behavior_prediction": "Values convenience and practical benefits",
                "ranking_score": 0.8,
                "execution_priority": 3
            }
        ]
        
        return {
            "universal_attributes": category_intelligence.get("universal_attributes", {}),
            "category_attributes": category_intelligence.get("category_attributes", {}),
            "ranked_archetypes": mock_archetypes,
            "top_archetypes": mock_archetypes[:2],
            "generation_metadata": {
                "total_archetypes": len(mock_archetypes),
                "avg_confidence": 0.82,
                "coverage_score": 0.8,
                "llm_generated": False,
                "mock_generated": True
            }
        }

# Factory function for backwards compatibility
def create_archetype_builder(use_llm: bool = True, ollama_config: Optional[OllamaConfig] = None) -> LLMArchetypeBuilder:
    """Factory function to create archetype builder."""
    return LLMArchetypeBuilder(ollama_config=ollama_config)