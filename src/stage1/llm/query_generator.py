# src/stage1/llm/query_generator.py
"""
LLM-powered query generator with fixed import handling.
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
class QueryGenerationPrompts:
    # ... (prompts are unchanged from the last version, with strict JSON requirement)
    AUTHENTIC_QUERY_GENERATION = """You are a customer behavior expert...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    QUERY_VARIETY_GENERATION = """Generate diverse query types...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""
    QUERY_REFINEMENT = """Refine these queries...
IMPORTANT: Your entire response must be ONLY the valid JSON object described above..."""


class LLMQueryGenerator:
    """LLM-powered query generator with robust fallback handling."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = QueryGenerationPrompts()
        self.total_tokens_used = 0
        self.mock_generator = InternalMockGenerator()

    async def generate_query_package(
        self, top_archetypes: List[Dict[str, Any]], category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> Dict[str, Any]:
        logger.info("Generating query package with LLM")
        
        if not top_archetypes:
            logger.warning("No archetypes provided to query generator. Returning empty package.")
            return {"styled_queries": [], "generation_metadata": {"total_queries": 0}}

        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)
                
                archetype_queries = await self._generate_archetype_queries(client, top_archetypes, category_intelligence, brand_context)
                category_queries = await self._generate_category_coverage_queries(client, top_archetypes, category_intelligence, brand_context)
                combined_queries = self._combine_and_deduplicate_queries(archetype_queries, category_queries)
                refined_queries = await self._refine_queries_for_authenticity(client, combined_queries, category_intelligence, brand_context)
                
                final_package = self._create_final_query_package(refined_queries)
                logger.info("Query package generated successfully")
                return final_package
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}", exc_info=True)
            return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)

    async def _run_json_generation(self, client: OllamaClient, request: LLMRequest) -> Dict[str, Any]:
        """Helper to run self-correcting JSON generation and count tokens."""
        parsed_json, tokens_used = await client.generate_json_with_self_correction(request)
        self.total_tokens_used += tokens_used
        return parsed_json

    async def _generate_archetype_queries(
        self, client: OllamaClient, top_archetypes: List[Dict[str, Any]], 
        category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        all_queries = []
        for archetype in top_archetypes:
            prompt = self.prompts.AUTHENTIC_QUERY_GENERATION.format(
                category=category_intelligence.get("category", "product"), archetype_name=archetype.get("name", "Customer"),
                archetype_description=archetype.get("description", ""), ai_behavior=archetype.get("ai_behavior_prediction", ""),
                communication_style=archetype.get("communication_style", "casual"), brand_name=brand_context.brand_name,
                competitors=", ".join(brand_context.competitive_context.primary_competitors[:3])
            )
            request = LLMRequest(prompt=prompt, model="mistral", temperature=0.7, max_tokens=1024, system_prompt="You are a customer behavior expert.")
            result = await self._run_json_generation(client, request)
            queries = result.get("styled_queries", [])
            for query in queries:
                query["archetype"], query["archetype_id"] = archetype.get("name"), archetype.get("archetype_id")
            all_queries.extend(queries)
        return all_queries

    async def _generate_category_coverage_queries(
        self, client: OllamaClient, top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        archetypes_summary = "\n".join([f"- {a.get('name', '')}: {a.get('description', '')[:100]}..." for a in top_archetypes])
        prompt = self.prompts.QUERY_VARIETY_GENERATION.format(
            category=category_intelligence.get("category", "product"), archetypes_summary=archetypes_summary,
            brand_name=brand_context.brand_name, competitors=", ".join(brand_context.competitive_context.primary_competitors[:3])
        )
        request = LLMRequest(prompt=prompt, model="mistral", temperature=0.6, max_tokens=1200, system_prompt="Generate diverse customer queries.")
        result = await self._run_json_generation(client, request)
        
        all_queries = []
        for cat_type, queries in result.get("query_categories", {}).items():
            for i, query_data in enumerate(queries[:2]):
                all_queries.append({"query_id": f"CAT{cat_type[:2].upper()}{i+1}", "styled_query": query_data.get("query", ""), "category": cat_type, "archetype": query_data.get("archetype", "Multiple")})
        return all_queries

    async def _refine_queries_for_authenticity(
        self, client: OllamaClient, queries: List[Dict[str, Any]], category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        if not queries: return []
        queries_for_refinement = [{"query_id": q.get("query_id", ""), "query": q.get("styled_query", "")} for q in queries[:10]]
        prompt = self.prompts.QUERY_REFINEMENT.format(
            queries=json.dumps(queries_for_refinement, indent=2), brand_name=brand_context.brand_name,
            category=category_intelligence.get("category", "product")
        )
        request = LLMRequest(prompt=prompt, model="mistral", temperature=0.4, max_tokens=1500, system_prompt="Refine customer queries for authenticity.")
        result = await self._run_json_generation(client, request)
        if result:
            return self._apply_refinements(queries, result.get("refined_queries", []))
        return queries

    def _combine_and_deduplicate_queries(self, archetype_queries: List[Dict[str, Any]], category_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        pass
        
    def _apply_refinements(self, original_queries: List[Dict[str, Any]], refinements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        pass

    def _create_final_query_package(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ... (implementation unchanged) ...
        pass
        
    def _fallback_to_mock(self, *args, **kwargs) -> Dict[str, Any]:
        logger.warning("LLM generation failed, using internal mock generator.")
        return self.mock_generator.generate_query_package(*args, **kwargs)

class InternalMockGenerator:
    def generate_query_package(self, *args, **kwargs) -> Dict[str, Any]:
        return {"styled_queries": [], "generation_metadata": {"mock_generated": True}}
    
# Factory function for backwards compatibility
def create_query_generator(
    use_llm: bool = True,
    ollama_config: Optional[OllamaConfig] = None
) -> LLMQueryGenerator:
    """Factory function to create query generator."""
    return LLMQueryGenerator(
        ollama_config=ollama_config,
        fallback_to_mock=True
    )