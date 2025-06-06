# src/stage1/llm/archetype_builder.py
"""
LLM-powered archetype builder with fixed import handling.
"""
import json
import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from models.brand import BrandContext
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass 
class ArchetypePrompts:
    """Structured prompts for archetype generation"""
    
    ARCHETYPE_GENERATION = """You are a customer psychology expert analyzing the {category} market for {brand_name}.

Category Intelligence:
{category_intelligence}

Brand Context:
- Competitors: {competitors}
- Positioning: {positioning}

Generate 3-5 distinct customer archetypes for this brand using this JSON format:
{{
    "archetypes": [
        {{
            "archetype_id": "ARCH_001",
            "name": "Descriptive Customer Name",
            "description": "Detailed description of customer psychology and behavior",
            "attributes": {{
                "COREB1": "primary_behavior_from_universal_set",
                "MODIFIERE1": "primary_identity_from_universal_set", 
                "MODIFIERD3": "price_tier_from_BUDGET_MIDRANGE_PREMIUM",
                "COREA2": "shopping_style_from_universal_set",
                "DEMOD2": "lifestyle_from_universal_set",
                "COREB3": "brand_preference_from_universal_set"
            }},
            "market_presence": "HIGH|MEDIUM|LOW",
            "strategic_value": "HIGH|MEDIUM|LOW", 
            "confidence": 0.0-1.0,
            "ai_behavior_prediction": "How this archetype behaves when researching products with AI"
        }}
    ]
}}

Universal Attributes Available:
{universal_attributes}

Focus on creating distinct, actionable customer personas that would research {category} products differently."""

    ARCHETYPE_RANKING = """Rank these customer archetypes by strategic importance for {brand_name} in the {category} market.

Archetypes to rank:
{archetypes}

Consider:
1. Market size and presence
2. Brand fit and alignment  
3. Revenue potential
4. Competitive vulnerability
5. AI research behavior patterns

Return JSON with ranked list:
{{
    "ranked_archetypes": [
        {{
            "archetype_id": "ARCH_XXX",
            "ranking_score": 0.0-1.0,
            "ranking_rationale": "Why this archetype ranks here",
            "execution_priority": 1-5
        }}
    ],
    "top_archetypes": ["ARCH_001", "ARCH_002"],
    "strategic_insights": [
        "Key insight about archetype prioritization",
        "Strategic recommendation for brand positioning"
    ]
}}"""

    BEHAVIORAL_REFINEMENT = """Refine the AI research behavior predictions for these customer archetypes:

Archetypes:
{archetypes}

For each archetype, predict specific AI interaction patterns:
- What questions they ask AI assistants
- How they phrase queries
- What information they prioritize
- Decision-making patterns

Return JSON:
{{
    "refined_archetypes": [
        {{
            "archetype_id": "ARCH_XXX",
            "ai_query_patterns": [
                "Example query this archetype would ask",
                "Another typical query pattern"
            ],
            "information_priorities": [
                "priority1", "priority2", "priority3"
            ],
            "decision_triggers": [
                "What convinces them to buy",
                "What makes them hesitate"
            ],
            "communication_style": "formal|casual|technical|emotional"
        }}
    ]
}}"""

class LLMArchetypeBuilder:
    """LLM-powered customer archetype builder with robust fallback capabilities."""
    
    def __init__(
        self,
        ollama_config: Optional[OllamaConfig] = None,
        fallback_to_mock: bool = True
    ):
        self.ollama_config = ollama_config or OllamaConfig()
        self.fallback_to_mock = fallback_to_mock
        self.prompts = ArchetypePrompts()
        self.mock_builder = None
        
        # Initialize mock builder for fallback
        if fallback_to_mock:
            self.mock_builder = self._create_mock_builder()
    
    def _create_mock_builder(self):
        """Create mock builder with proper import handling."""
        try:
            mock_builder = self._try_import_mock_builder()
            if mock_builder:
                return mock_builder
            else:
                logger.warning("Could not import mock builder - using internal fallback")
                return InternalMockBuilder()
        except Exception as e:
            logger.warning(f"Mock builder import failed: {e} - using internal fallback")
            return InternalMockBuilder()
    
    def _try_import_mock_builder(self):
        """Try different strategies to import the mock builder."""
        import importlib.util
        
        # Strategy 1: Try direct import from stage1
        try:
            current_dir = os.path.dirname(__file__)
            mock_file_path = os.path.join(current_dir, '..', 'archetype_builder.py')
            mock_file_path = os.path.abspath(mock_file_path)
            
            if os.path.exists(mock_file_path):
                spec = importlib.util.spec_from_file_location("mock_archetype_builder", mock_file_path)
                if spec and spec.loader:
                    mock_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mock_module)
                    
                    if hasattr(mock_module, 'ArchetypeBuilder'):
                        logger.info("Successfully imported mock ArchetypeBuilder")
                        return mock_module.ArchetypeBuilder()
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try sys.path manipulation
        try:
            original_path = sys.path.copy()
            stage1_path = os.path.join(os.path.dirname(__file__), '..')
            if stage1_path not in sys.path:
                sys.path.insert(0, stage1_path)
            
            from archetype_builder import ArchetypeBuilder
            sys.path = original_path
            logger.info("Successfully imported mock ArchetypeBuilder via sys.path")
            return ArchetypeBuilder()
            
        except Exception as e:
            logger.debug(f"Strategy 2 failed: {e}")
            sys.path = original_path
        
        return None
    
    async def generate_archetypes(
        self,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate customer archetypes using LLM analysis."""
        
        logger.info("Building customer archetypes with LLM",
                   metadata={"category": category_intelligence.get("category")})
        
        try:
            async with OllamaClient(self.ollama_config) as client:
                # Check if Ollama is available
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(category_intelligence, brand_context)
                
                # Step 1: Generate initial archetypes
                raw_archetypes = await self._generate_initial_archetypes(
                    client, category_intelligence, brand_context
                )
                
                # Step 2: Refine AI behavior predictions
                refined_archetypes = await self._refine_behavioral_patterns(
                    client, raw_archetypes, category_intelligence
                )
                
                # Step 3: Rank archetypes strategically
                ranking_analysis = await self._rank_archetypes_strategically(
                    client, refined_archetypes, brand_context, category_intelligence
                )
                
                # Step 4: Combine into final result
                final_result = self._combine_archetype_analysis(
                    category_intelligence,
                    refined_archetypes,
                    ranking_analysis
                )
                
                logger.info("Archetypes generated successfully",
                           metadata={
                               "total_archetypes": len(final_result.get("ranked_archetypes", [])),
                               "avg_confidence": final_result.get("generation_metadata", {}).get("avg_confidence", 0)
                           })
                
                return final_result
                
        except Exception as e:
            logger.error(f"LLM archetype generation failed: {e}")
            if self.fallback_to_mock:
                logger.info("Falling back to mock implementation")
                return self._fallback_to_mock(category_intelligence, brand_context)
            raise
    
    async def _generate_initial_archetypes(
        self,
        client: OllamaClient,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate initial customer archetypes."""
        
        # Prepare category intelligence summary
        intelligence_summary = self._summarize_category_intelligence(category_intelligence)
        
        prompt = self.prompts.ARCHETYPE_GENERATION.format(
            category=category_intelligence.get("category", "product"),
            brand_name=brand_context.brand_name,
            category_intelligence=intelligence_summary,
            competitors=", ".join(brand_context.competitive_context.primary_competitors),
            positioning=getattr(brand_context, 'brand_positioning', 'Not specified'),
            universal_attributes=json.dumps(
                category_intelligence.get("universal_attributes", {}), 
                indent=2
            )
        )
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1200,
            system_prompt="You are a customer psychology expert. Create distinct, realistic customer archetypes with valid JSON."
        )
        
        response = await client.generate(request)
        
        if not response.success:
            raise RuntimeError(f"Archetype generation failed: {response.error}")
        
        try:
            result = json.loads(response.content)
            return result.get("archetypes", [])
        except json.JSONDecodeError:
            extracted = self._extract_json_from_response(response.content)
            return extracted.get("archetypes", [])
    
    async def _refine_behavioral_patterns(
        self,
        client: OllamaClient,
        archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Refine AI behavior predictions for each archetype."""
        
        top_archetypes = archetypes[:4]
        
        prompt = self.prompts.BEHAVIORAL_REFINEMENT.format(
            archetypes=json.dumps(top_archetypes, indent=2)
        )
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800,
            system_prompt="You are an AI behavior specialist. Predict how different customer types interact with AI assistants."
        )
        
        response = await client.generate(request)
        
        if response.success:
            try:
                result = json.loads(response.content)
                refined = result.get("refined_archetypes", [])
                return self._merge_behavioral_refinements(archetypes, refined)
            except json.JSONDecodeError:
                logger.warning("Failed to parse behavioral refinements, using original archetypes")
        
        return archetypes
    
    async def _rank_archetypes_strategically(
        self,
        client: OllamaClient,
        archetypes: List[Dict[str, Any]], 
        brand_context: BrandContext,
        category_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rank archetypes by strategic importance."""
        
        prompt = self.prompts.ARCHETYPE_RANKING.format(
            brand_name=brand_context.brand_name,
            category=category_intelligence.get("category", "product"),
            archetypes=json.dumps(archetypes[:5], indent=2)
        )
        
        request = LLMRequest(
            prompt=prompt,
            temperature=0.2,
            max_tokens=600,
            system_prompt="You are a strategic marketing expert. Rank customer archetypes by business value."
        )
        
        response = await client.generate(request)
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._extract_json_from_response(response.content)
        
        # Fallback ranking
        return {
            "ranked_archetypes": [
                {
                    "archetype_id": arch.get("archetype_id", f"ARCH_{i:03d}"),
                    "ranking_score": max(0.9 - i * 0.1, 0.5),
                    "ranking_rationale": "Fallback ranking based on order",
                    "execution_priority": i + 1
                }
                for i, arch in enumerate(archetypes[:5])
            ],
            "top_archetypes": [arch.get("archetype_id", f"ARCH_{i:03d}") for i, arch in enumerate(archetypes[:2])],
            "strategic_insights": ["Focus on top-ranked archetypes for initial execution"]
        }
    
    def _summarize_category_intelligence(self, intelligence: Dict[str, Any]) -> str:
        """Create a concise summary of category intelligence for prompts."""
        summary_parts = []
        
        if intelligence.get("category_insights"):
            insights = intelligence["category_insights"]
            if insights.get("characteristics"):
                summary_parts.append(f"Category: {insights['characteristics']}")
            if insights.get("customer_segments"):
                segments = [seg.get("segment_name", "Unknown") for seg in insights["customer_segments"][:3]]
                summary_parts.append(f"Key Segments: {', '.join(segments)}")
        
        if intelligence.get("competitive_landscape"):
            landscape = intelligence["competitive_landscape"]
            if landscape.get("positioning_opportunities"):
                opps = [opp.get("opportunity", "Unknown") for opp in landscape["positioning_opportunities"][:2]]
                summary_parts.append(f"Opportunities: {', '.join(opps)}")
        
        return "\n".join(summary_parts) if summary_parts else "Limited category intelligence available"
    
    def _merge_behavioral_refinements(
        self,
        original_archetypes: List[Dict[str, Any]],
        refinements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge behavioral refinements into original archetypes."""
        
        refinement_lookup = {
            ref.get("archetype_id"): ref 
            for ref in refinements
        }
        
        merged = []
        for archetype in original_archetypes:
            merged_archetype = archetype.copy()
            arch_id = archetype.get("archetype_id")
            refinement = refinement_lookup.get(arch_id, {})
            
            if refinement:
                merged_archetype.update({
                    "ai_query_patterns": refinement.get("ai_query_patterns", []),
                    "information_priorities": refinement.get("information_priorities", []),
                    "decision_triggers": refinement.get("decision_triggers", []),
                    "communication_style": refinement.get("communication_style", "casual")
                })
                
                if refinement.get("ai_query_patterns"):
                    merged_archetype["ai_behavior_prediction"] = (
                        f"{merged_archetype.get('ai_behavior_prediction', '')} "
                        f"Typical queries: {', '.join(refinement['ai_query_patterns'][:2])}"
                    ).strip()
            
            merged.append(merged_archetype)
        
        return merged
    
    def _combine_archetype_analysis(
        self,
        category_intelligence: Dict[str, Any],
        refined_archetypes: List[Dict[str, Any]],
        ranking_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all archetype analysis into final result."""
        
        ranked_ids = [r.get("archetype_id") for r in ranking_analysis.get("ranked_archetypes", [])]
        top_archetype_ids = ranking_analysis.get("top_archetypes", ranked_ids[:2])
        
        ranked_archetypes = []
        archetype_lookup = {arch.get("archetype_id"): arch for arch in refined_archetypes}
        
        for ranked_id in ranked_ids:
            if ranked_id in archetype_lookup:
                archetype = archetype_lookup[ranked_id].copy()
                
                ranking_info = next(
                    (r for r in ranking_analysis.get("ranked_archetypes", []) 
                     if r.get("archetype_id") == ranked_id),
                    {}
                )
                
                if ranking_info:
                    archetype.update({
                        "ranking_score": ranking_info.get("ranking_score", 0.5),
                        "ranking_rationale": ranking_info.get("ranking_rationale", ""),
                        "execution_priority": ranking_info.get("execution_priority", 999)
                    })
                
                ranked_archetypes.append(archetype)
        
        for archetype in refined_archetypes:
            if archetype.get("archetype_id") not in ranked_ids:
                ranked_archetypes.append(archetype)
        
        top_archetypes = [
            arch for arch in ranked_archetypes 
            if arch.get("archetype_id") in top_archetype_ids
        ]
        
        confidences = [arch.get("confidence", 0.8) for arch in ranked_archetypes]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
        
        return {
            "universal_attributes": category_intelligence.get("universal_attributes", {}),
            "category_attributes": category_intelligence.get("category_attributes", {}),
            "ranked_archetypes": ranked_archetypes,
            "top_archetypes": top_archetypes,
            "generation_metadata": {
                "total_archetypes": len(ranked_archetypes),
                "avg_confidence": avg_confidence,
                "coverage_score": min(len(ranked_archetypes) / 5.0, 1.0),
                "llm_generated": True,
                "strategic_insights": ranking_analysis.get("strategic_insights", [])
            }
        }
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response that may contain extra text."""
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
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        if self.mock_builder:
            try:
                return self.mock_builder.generate_archetypes(category_intelligence, brand_context)
            except Exception as e:
                logger.warning(f"Mock builder failed: {e}")
        
        return self._create_emergency_fallback(category_intelligence, brand_context)
    
    def _create_emergency_fallback(
        self,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Create emergency fallback when all else fails."""
        category = category_intelligence.get("category", "product")
        
        emergency_archetype = {
            "archetype_id": "ARCH_001",
            "name": f"Quality-Focused {category.title()} Customer",
            "description": f"Premium customer seeking high-quality {category} solutions",
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
            "ai_behavior_prediction": "Seeks detailed comparisons and expert reviews"
        }
        
        return {
            "universal_attributes": category_intelligence.get("universal_attributes", {}),
            "category_attributes": category_intelligence.get("category_attributes", {}),
            "ranked_archetypes": [emergency_archetype],
            "top_archetypes": [emergency_archetype],
            "generation_metadata": {
                "total_archetypes": 1,
                "avg_confidence": 0.85,
                "coverage_score": 0.7,
                "emergency_fallback": True
            }
        }


class InternalMockBuilder:
    """Internal mock implementation when external mock is unavailable."""
    
    def generate_archetypes(
        self,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate mock customer archetypes."""
        
        logger.info("Using internal mock builder")
        category = category_intelligence.get("category", "product")
        
        mock_archetypes = [
            {
                "archetype_id": "ARCH_001",
                "name": f"Premium {category.title()} Enthusiast",
                "description": f"High-income customer seeking premium {category} with top features",
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
                "confidence": 0.9,
                "ai_behavior_prediction": "Asks detailed questions about premium features and brand comparisons"
            },
            {
                "archetype_id": "ARCH_002",
                "name": f"Smart {category.title()} Shopper",
                "description": f"Value-conscious customer seeking best {category} for their budget",
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
                "confidence": 0.85,
                "ai_behavior_prediction": "Seeks quick recommendations with clear value propositions"
            },
            {
                "archetype_id": "ARCH_003",
                "name": f"Budget {category.title()} Buyer",
                "description": f"Price-sensitive customer looking for functional {category} at lowest cost",
                "attributes": {
                    "COREB1": "BUSY_PRACTICAL",
                    "MODIFIERE1": "SMART_SHOPPER",
                    "MODIFIERD3": "BUDGET",
                    "COREA2": "ROUTINE_ONLINE",
                    "DEMOD2": "SUBURBAN_FAMILY",
                    "COREB3": "BRAND_BLIND"
                },
                "market_presence": "MEDIUM",
                "strategic_value": "LOW",
                "confidence": 0.8,
                "ai_behavior_prediction": "Focuses on price comparisons and basic functionality"
            }
        ]
        
        return {
            "universal_attributes": category_intelligence.get("universal_attributes", {}),
            "category_attributes": category_intelligence.get("category_attributes", {}),
            "ranked_archetypes": mock_archetypes,
            "top_archetypes": mock_archetypes[:2],
            "generation_metadata": {
                "total_archetypes": len(mock_archetypes),
                "avg_confidence": 0.85,
                "coverage_score": 0.8,
                "mock_generated": True
            }
        }


# Factory function for backwards compatibility
def create_archetype_builder(
    use_llm: bool = True,
    ollama_config: Optional[OllamaConfig] = None
) -> LLMArchetypeBuilder:
    """Factory function to create archetype builder."""
    return LLMArchetypeBuilder(
        ollama_config=ollama_config,
        fallback_to_mock=True
    )