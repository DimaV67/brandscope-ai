# src/stage1/llm/archetype_builder.py
"""
LLM-powered archetype builder with fixed import handling.
"""
import json
import asyncio
import re
# FIXED: Added 'Any' to the typing import
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient, LLMRequest, OllamaConfig
from src.models.brand import BrandContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass 
class ArchetypePrompts:
    # ... (prompts are unchanged from the last version, with strict JSON requirement)
    ARCHETYPE_GENERATION = """CRITICAL CONTEXT: You are a customer psychology expert...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    ARCHETYPE_RANKING = """Rank these customer archetypes...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    BEHAVIORAL_REFINEMENT = """Refine the AI research behavior predictions...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""


class LLMArchetypeBuilder:
    """LLM-powered customer archetype builder with robust fallback capabilities."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = ArchetypePrompts()
        self.total_tokens_used = 0
        self.mock_builder = InternalMockBuilder()

    async def generate_archetypes(self, category_intelligence: Dict[str, Any], brand_context: BrandContext) -> Dict[str, Any]:
        logger.info("Building customer archetypes with LLM")
        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    return self._fallback_to_mock(category_intelligence, brand_context)
                
                raw_archetypes = await self._generate_initial_archetypes(client, category_intelligence, brand_context)
                if not raw_archetypes:
                    logger.warning("Initial archetype generation returned no results. Aborting.")
                    return {"ranked_archetypes": [], "top_archetypes": [], "generation_metadata": {"total_archetypes": 0}}
                
                refined_archetypes = await self._refine_behavioral_patterns(client, raw_archetypes)
                ranking_analysis = await self._rank_archetypes_strategically(client, refined_archetypes, brand_context, category_intelligence)
                
                return self._combine_archetype_analysis(category_intelligence, refined_archetypes, ranking_analysis)
        except Exception as e:
            logger.error(f"LLM archetype generation failed: {e}", exc_info=True)
            return self._fallback_to_mock(category_intelligence, brand_context)

    async def _run_json_generation(self, client: OllamaClient, request: LLMRequest) -> Dict[str, Any]:
        """Helper to run self-correcting JSON generation and count tokens."""
        parsed_json, tokens_used = await client.generate_json_with_self_correction(request)
        self.total_tokens_used += tokens_used
        return parsed_json

    async def _generate_initial_archetypes(self, client: OllamaClient, category_intelligence: Dict[str, Any], brand_context: BrandContext) -> List[Dict[str, Any]]:
        rag_context_summary = self._summarize_category_intelligence(category_intelligence)
        prompt = self.prompts.ARCHETYPE_GENERATION.format(
            category=category_intelligence.get("category", "product"), brand_name=brand_context.brand_name,
            rag_context_summary=rag_context_summary,
            universal_attributes=json.dumps(category_intelligence.get("universal_attributes", {}), indent=2)
        )
        request = LLMRequest(prompt=prompt, model="llama3.2", temperature=0.4, max_tokens=2048, system_prompt="You are a customer psychology expert.")
        result = await self._run_json_generation(client, request)
        return result.get("archetypes", [])

    async def _refine_behavioral_patterns(self, client: OllamaClient, archetypes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not archetypes: return []
        prompt = self.prompts.BEHAVIORAL_REFINEMENT.format(archetypes=json.dumps(archetypes[:4], indent=2))
        request = LLMRequest(prompt=prompt, model="llama3.2", temperature=0.3, max_tokens=1024, system_prompt="You are an AI behavior specialist.")
        result = await self._run_json_generation(client, request)
        if result:
            refined = result.get("refined_archetypes", [])
            return self._merge_behavioral_refinements(archetypes, refined)
        logger.warning("Failed to parse behavioral refinements, using original archetypes")
        return archetypes

    async def _rank_archetypes_strategically(self, client: OllamaClient, archetypes: List[Dict[str, Any]], brand_context: BrandContext, category_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        if not archetypes: return {}
        prompt = self.prompts.ARCHETYPE_RANKING.format(
            brand_name=brand_context.brand_name, category=category_intelligence.get("category", "product"),
            archetypes=json.dumps(archetypes[:5], indent=2)
        )
        request = LLMRequest(prompt=prompt, model="llama3.2", temperature=0.2, max_tokens=800, system_prompt="You are a strategic marketing expert.")
        return await self._run_json_generation(client, request)

    def _summarize_category_intelligence(self, intelligence: Dict[str, Any]) -> str:
        summary_parts = []
        if insights := intelligence.get("category_insights"):
            if characteristics := insights.get("characteristics"): summary_parts.append(f"Category Characteristics: {json.dumps(characteristics)}")
            if segments := insights.get("customer_segments"): summary_parts.append(f"Key Customer Segments: {'; '.join([f'{s.get("segment_name", "Unknown")}' for s in segments[:3]])}")
        if landscape := intelligence.get("competitive_landscape"):
            summary_parts.append(f"Primary Competitors: {', '.join(landscape.get('primary_competitors', []))}")
            if opportunities := landscape.get("positioning_opportunities"): summary_parts.append(f"Positioning Opportunities: {', '.join([o.get('opportunity', 'Unknown') for o in opportunities[:2]])}")
        return "\n".join(summary_parts) if summary_parts else "No specific category intelligence provided."

    def _merge_behavioral_refinements(self, original_archetypes: List[Dict[str, Any]], refinements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refinement_lookup = {ref.get("archetype_id"): ref for ref in refinements}
        for arch in original_archetypes:
            if refinement := refinement_lookup.get(arch.get("archetype_id")):
                arch.update(refinement)
        return original_archetypes

    def _combine_archetype_analysis(self, category_intelligence: Dict[str, Any], refined_archetypes: List[Dict[str, Any]], ranking_analysis: Dict[str, Any]) -> Dict[str, Any]:
        ranked_archetypes_info = {r['archetype_id']: r for r in ranking_analysis.get("ranked_archetypes", [])}
        for arch in refined_archetypes:
            if rank_info := ranked_archetypes_info.get(arch.get('archetype_id', '')):
                arch.update(rank_info)
        
        refined_archetypes.sort(key=lambda x: x.get('ranking_score', 0.0), reverse=True)
        top_archetypes = [arch for arch in refined_archetypes if arch.get('archetype_id') in ranking_analysis.get('top_archetypes', [])]

        return {"universal_attributes": category_intelligence.get("universal_attributes", {}), "category_attributes": category_intelligence.get("category_attributes", {}), "ranked_archetypes": refined_archetypes, "top_archetypes": top_archetypes, "generation_metadata": {"total_archetypes": len(refined_archetypes), "llm_generated": True, "strategic_insights": ranking_analysis.get("strategic_insights", [])}}
        
    def _fallback_to_mock(self, *args, **kwargs) -> Dict[str, Any]:
        logger.warning("LLM generation failed, using internal mock builder.")
        return self.mock_builder.generate_archetypes(*args, **kwargs)

class InternalMockBuilder:
    def generate_archetypes(self, *args, **kwargs) -> Dict[str, Any]:
        return {"ranked_archetypes": [], "top_archetypes": [], "generation_metadata": {"mock_generated": True}}
    
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