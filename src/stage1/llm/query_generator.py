# src/stage1/llm/query_generator.py
"""
Enhanced LLM-powered query generator with robust error handling.
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
class QueryGenerationPrompts:
    """Structured prompts for query generation with strict JSON requirements"""
    
    AUTHENTIC_QUERY_GENERATION = """Generate authentic customer queries for {category} research.

Customer Profile:
- Archetype: {archetype_name}
- Description: {archetype_description}  
- Communication Style: {communication_style}
- AI Behavior: {ai_behavior}

Brand Context:
- Target Brand: {brand_name}
- Competitors: {competitors}

Create 4-6 natural queries this customer would ask when researching {category}.

Respond with ONLY this JSON structure:
{{
    "styled_queries": [
        {{
            "query_id": "Q001",
            "styled_query": "Looking for the best {category} for quality and reliability - what do experts recommend?",
            "original_query": "Which {category} brand is recommended for quality?",
            "category": "direct_recommendation",
            "authenticity_score": 8.5,
            "reasoning": "This archetype values expert opinions and quality assurance"
        }},
        {{
            "query_id": "Q002", 
            "styled_query": "Need a {category} that won't break the bank but still works well - any suggestions?",
            "original_query": "What's the best value {category}?",
            "category": "indirect_recommendation",
            "authenticity_score": 8.2,
            "reasoning": "Balances budget concerns with quality expectations"
        }},
        {{
            "query_id": "Q003",
            "styled_query": "{brand_name} vs {main_competitor} - which is better for everyday use?",
            "original_query": "Compare {brand_name} and {main_competitor}",
            "category": "comparative_analysis",
            "authenticity_score": 8.8,
            "reasoning": "Direct brand comparison reflects decision-making process"
        }}
    ]
}}"""

    CATEGORY_COVERAGE_QUERIES = """Generate diverse query types for comprehensive {category} research coverage.

Target Archetypes:
{archetypes_summary}

Brand: {brand_name}
Competitors: {competitors}

Create queries covering different customer intentions and research stages.

Respond with ONLY this JSON structure:
{{
    "query_categories": {{
        "direct_recommendation": [
            {{
                "query": "What's the best {category} for my needs?",
                "archetype": "Quality Seeker", 
                "authenticity": 8.0
            }}
        ],
        "problem_solving": [
            {{
                "query": "I'm having trouble choosing between {category} options - help?",
                "archetype": "Overwhelmed Shopper",
                "authenticity": 7.5
            }}
        ],
        "feature_inquiry": [
            {{
                "query": "Does {brand_name} {category} have the features I need?", 
                "archetype": "Feature-Focused",
                "authenticity": 8.2
            }}
        ],
        "price_value": [
            {{
                "query": "Is {brand_name} {category} worth the price?",
                "archetype": "Value Conscious",
                "authenticity": 8.3
            }}
        ]
    }}
}}"""

    QUERY_REFINEMENT = """Refine these customer queries for maximum authenticity and brand research effectiveness.

Original Queries:
{queries}

Brand: {brand_name}
Category: {category}

Improve each query for:
1. Natural conversational language
2. Specificity and relevance  
3. Brand mention likelihood
4. Customer psychology alignment

Respond with ONLY this JSON structure:
{{
    "refined_queries": [
        {{
            "query_id": "Q001",
            "original": "original query text",
            "refined": "improved query with better natural language",
            "improvements": ["more conversational", "added context", "brand-relevant"],
            "authenticity_score": 8.5,
            "brand_mention_likelihood": 8.0
        }}
    ]
}}"""

class LLMQueryGenerator:
    """Enhanced LLM-powered query generator with robust fallback handling."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.ollama_config = ollama_config or OllamaConfig()
        self.prompts = QueryGenerationPrompts()
        self.total_tokens_used = 0
        self.mock_generator = InternalMockGenerator()

    async def generate_query_package(
        self, top_archetypes: List[Dict[str, Any]], category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate comprehensive query package using LLM analysis."""
        
        logger.info("Generating query package with LLM",
                   metadata={"archetypes_count": len(top_archetypes)})
        
        if not top_archetypes:
            logger.warning("No archetypes provided to query generator. Using fallback.")
            return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)

        try:
            async with OllamaClient(self.ollama_config) as client:
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)
                
                logger.info("Step 1/4: Generating archetype-specific queries...")
                archetype_queries = await self._generate_archetype_queries(client, top_archetypes, category_intelligence, brand_context)
                
                logger.info("Step 2/4: Generating category coverage queries...")
                category_queries = await self._generate_category_coverage_queries(client, top_archetypes, category_intelligence, brand_context)
                
                logger.info("Step 3/4: Combining and deduplicating...")
                combined_queries = self._combine_and_deduplicate_queries(archetype_queries, category_queries)
                
                logger.info("Step 4/4: Refining for authenticity...")
                refined_queries = await self._refine_queries_for_authenticity(client, combined_queries, category_intelligence, brand_context)
                
                final_package = self._create_final_query_package(refined_queries, top_archetypes, category_intelligence, brand_context)
                
                logger.info("Query package generated successfully",
                           metadata={
                               "total_queries": len(final_package.get("styled_queries", [])),
                               "avg_authenticity": final_package.get("generation_metadata", {}).get("avg_authenticity", 0),
                               "total_tokens": self.total_tokens_used
                           })
                
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
        """Generate queries for each top archetype."""
        
        all_queries = []
        category = category_intelligence.get("category", "product")
        competitors = brand_context.competitive_context.primary_competitors
        main_competitor = competitors[0] if competitors else "leading competitor"
        competitors_str = ", ".join(competitors[:3]) if competitors else "various competitors"
        
        for archetype in top_archetypes[:3]:  # Focus on top 3 archetypes
            archetype_name = archetype.get("name", "Customer")
            archetype_desc = archetype.get("description", "Generic customer")
            ai_behavior = archetype.get("ai_behavior_prediction", "Seeks general information")
            comm_style = archetype.get("communication_style", "casual")
            
            prompt = self.prompts.AUTHENTIC_QUERY_GENERATION.format(
                category=category,
                archetype_name=archetype_name,
                archetype_description=archetype_desc,
                communication_style=comm_style,
                ai_behavior=ai_behavior,
                brand_name=brand_context.brand_name,
                competitors=competitors_str,
                main_competitor=main_competitor
            )
            
            request = LLMRequest(
                prompt=prompt,
                model="mistral",  # Use creative model for natural language
                temperature=0.7,
                max_tokens=1024,
                system_prompt="Generate authentic customer queries. Respond only with valid JSON."
            )
            
            result = await self._run_json_generation(client, request)
            queries = result.get("styled_queries", [])
            
            # Add archetype info to each query
            for i, query in enumerate(queries):
                query["archetype"] = archetype_name
                query["archetype_id"] = archetype.get("archetype_id", "UNKNOWN")
                if not query.get("query_id"):
                    query["query_id"] = f"A{archetype.get('archetype_id', 'XXX')[-3:]}_Q{i+1:02d}"
            
            all_queries.extend(queries)
            
            # Add fallback queries if LLM didn't generate enough
            if len(queries) < 3:
                fallback_queries = self._generate_fallback_archetype_queries(archetype, category, brand_context)
                all_queries.extend(fallback_queries)
        
        return all_queries

    async def _generate_category_coverage_queries(
        self, client: OllamaClient, top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate queries to ensure comprehensive category coverage."""
        
        category = category_intelligence.get("category", "product")
        competitors = ", ".join(brand_context.competitive_context.primary_competitors[:3]) if brand_context.competitive_context.primary_competitors else "various competitors"
        
        archetypes_summary = "\n".join([
            f"- {arch.get('name', 'Unknown')}: {arch.get('description', 'No description')[:80]}..."
            for arch in top_archetypes[:3]
        ])
        
        prompt = self.prompts.CATEGORY_COVERAGE_QUERIES.format(
            category=category,
            archetypes_summary=archetypes_summary,
            brand_name=brand_context.brand_name,
            competitors=competitors
        )
        
        request = LLMRequest(
            prompt=prompt,
            model="mistral",
            temperature=0.6,
            max_tokens=1200,
            system_prompt="Generate diverse customer queries covering all research intents. Respond only with valid JSON."
        )
        
        result = await self._run_json_generation(client, request)
        
        all_queries = []
        query_categories = result.get("query_categories", {})
        
        query_id = 1
        for category_type, queries in query_categories.items():
            for query_data in queries[:2]:  # Limit per category
                all_queries.append({
                    "query_id": f"CAT_{category_type[:3].upper()}_{query_id:02d}",
                    "styled_query": query_data.get("query", ""),
                    "original_query": query_data.get("query", ""),
                    "category": category_type,
                    "archetype": query_data.get("archetype", "Multiple"),
                    "authenticity_score": query_data.get("authenticity", 7.0),
                    "execution_priority": query_id + 100,  # Lower priority than archetype queries
                    "source": "category_coverage"
                })
                query_id += 1
        
        return all_queries

    def _combine_and_deduplicate_queries(
        self, archetype_queries: List[Dict[str, Any]], category_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine queries and remove duplicates."""
        
        all_queries = archetype_queries + category_queries
        
        # Deduplicate by query text similarity
        unique_queries = []
        seen_queries = set()
        
        for query in all_queries:
            query_text = query.get("styled_query", "").lower().strip()
            
            # Skip empty queries
            if not query_text:
                continue
            
            # Simple similarity check
            is_duplicate = False
            for seen_query in seen_queries:
                if self._calculate_similarity(query_text, seen_query) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_queries.append(query)
                seen_queries.add(query_text)
        
        return unique_queries

    async def _refine_queries_for_authenticity(
        self, client: OllamaClient, queries: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Refine queries for maximum authenticity and effectiveness."""
        
        if not queries:
            return []
        
        # Focus on top queries by authenticity score
        top_queries = sorted(
            queries, 
            key=lambda q: q.get("authenticity_score", 5.0), 
            reverse=True
        )[:10]
        
        queries_for_refinement = [
            {
                "query_id": q.get("query_id", ""),
                "query": q.get("styled_query", ""),
                "category": q.get("category", ""),
                "archetype": q.get("archetype", "")
            }
            for q in top_queries
        ]
        
        prompt = self.prompts.QUERY_REFINEMENT.format(
            queries=json.dumps(queries_for_refinement, indent=2),
            brand_name=brand_context.brand_name,
            category=category_intelligence.get("category", "product")
        )
        
        request = LLMRequest(
            prompt=prompt,
            model="mistral",
            temperature=0.4,
            max_tokens=1500,
            system_prompt="Refine customer queries for maximum authenticity. Respond only with valid JSON."
        )
        
        result = await self._run_json_generation(client, request)
        
        if result and result.get("refined_queries"):
            refined_queries = result.get("refined_queries", [])
            return self._apply_refinements(queries, refined_queries)
        
        return queries

    def _apply_refinements(
        self, original_queries: List[Dict[str, Any]], refinements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply refinements back to original queries."""
        
        refinement_lookup = {ref.get("query_id"): ref for ref in refinements}
        
        refined_queries = []
        for query in original_queries:
            query_id = query.get("query_id", "")
            refinement = refinement_lookup.get(query_id)
            
            if refinement and refinement.get("refined"):
                # Apply refinement
                refined_query = query.copy()
                refined_query.update({
                    "styled_query": refinement["refined"],
                    "authenticity_score": refinement.get("authenticity_score", query.get("authenticity_score", 7.0)),
                    "brand_mention_likelihood": refinement.get("brand_mention_likelihood", 7.0),
                    "improvements": refinement.get("improvements", []),
                    "refined": True
                })
                refined_queries.append(refined_query)
            else:
                # Keep original
                refined_queries.append(query)
        
        return refined_queries

    def _create_final_query_package(
        self, queries: List[Dict[str, Any]], top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Create final structured query package."""
        
        # Assign execution priorities
        prioritized_queries = []
        priority = 1
        
        # Sort by authenticity score, then by whether it's archetype-specific
        sorted_queries = sorted(queries, key=lambda q: (
            q.get("authenticity_score", 0),
            1 if q.get("source") != "category_coverage" else 0
        ), reverse=True)
        
        for query in sorted_queries[:20]:  # Limit to top 20 queries
            query_copy = query.copy()
            query_copy["execution_priority"] = priority
            
            # Ensure required fields
            if not query_copy.get("query_id"):
                query_copy["query_id"] = f"Q{priority:03d}"
            
            # Ensure category field
            if not query_copy.get("category"):
                query_copy["category"] = "general_inquiry"
            
            prioritized_queries.append(query_copy)
            priority += 1
        
        # Calculate metadata
        authenticity_scores = [q.get("authenticity_score", 7.0) for q in prioritized_queries]
        avg_authenticity = sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 7.0
        
        categories_covered = list(set(q.get("category", "unknown") for q in prioritized_queries))
        
        return {
            "styled_queries": prioritized_queries,
            "generation_metadata": {
                "total_queries": len(prioritized_queries),
                "avg_authenticity": avg_authenticity,
                "categories_covered": categories_covered,
                "archetypes_covered": len(top_archetypes),
                "llm_generated": True,
                "refinement_applied": any(q.get("refined", False) for q in prioritized_queries),
                "token_usage": self.total_tokens_used
            },
            "framework_compliance": True
        }

    def _generate_fallback_archetype_queries(
        self, archetype: Dict[str, Any], category: str, brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate fallback queries for an archetype when LLM fails."""
        
        archetype_name = archetype.get("name", "Customer")
        attributes = archetype.get("attributes", {})
        arch_id = archetype.get("archetype_id", "XXX")[-3:]
        
        queries = []
        
        if attributes.get("MODIFIERD3") == "PREMIUM":
            queries.append({
                "query_id": f"FB_{arch_id}_01",
                "styled_query": f"What's the highest quality {category} available? Money isn't an issue.",
                "original_query": f"Best quality {category}",
                "category": "direct_recommendation",
                "authenticity_score": 7.5,
                "archetype": archetype_name,
                "archetype_id": archetype.get("archetype_id", "UNKNOWN")
            })
        
        if attributes.get("COREA2") == "RESEARCH":
            queries.append({
                "query_id": f"FB_{arch_id}_02",
                "styled_query": f"I've been researching {category} options - what do experts actually recommend?",
                "original_query": f"Expert recommendations {category}",
                "category": "indirect_recommendation",
                "authenticity_score": 8.0,
                "archetype": archetype_name,
                "archetype_id": archetype.get("archetype_id", "UNKNOWN")
            })
        
        # Always add a comparison query if competitors exist
        if brand_context.competitive_context.primary_competitors:
            competitor = brand_context.competitive_context.primary_competitors[0]
            queries.append({
                "query_id": f"FB_{arch_id}_03",
                "styled_query": f"{brand_context.brand_name} vs {competitor} - which should I choose?",
                "original_query": f"Compare {brand_context.brand_name} {competitor}",
                "category": "comparative_analysis",
                "authenticity_score": 8.5,
                "archetype": archetype_name,
                "archetype_id": archetype.get("archetype_id", "UNKNOWN")
            })
        
        return queries

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _fallback_to_mock(
        self, top_archetypes: List[Dict[str, Any]], category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        logger.warning("LLM generation failed, using internal mock generator.")
        return self.mock_generator.generate_query_package(top_archetypes, category_intelligence, brand_context)

class InternalMockGenerator:
    """Internal mock generator for fallback scenarios."""
    
    def generate_query_package(
        self, top_archetypes: List[Dict[str, Any]], category_intelligence: Dict[str, Any], brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate mock queries."""
        
        category = category_intelligence.get("category", "product")
        competitors = brand_context.competitive_context.primary_competitors
        main_competitor = competitors[0] if competitors else "leading competitor"
        
        mock_queries = [
            {
                "query_id": "MOCK_001",
                "styled_query": f"Looking for the best {category} for quality and reliability - what do experts recommend?",
                "original_query": f"Which {category} brand is recommended for quality?",
                "archetype": "Quality-Focused Customer",
                "category": "direct_recommendation",
                "execution_priority": 1,
                "authenticity_score": 8.5
            },
            {
                "query_id": "MOCK_002",
                "styled_query": f"Need a {category} that won't break the bank but still works well - any suggestions?",
                "original_query": f"What's the best value {category}?",
                "archetype": "Value-Conscious Customer",
                "category": "indirect_recommendation",
                "execution_priority": 2,
                "authenticity_score": 8.2
            }
        ]
        
        if competitors:
            mock_queries.append({
                "query_id": "MOCK_003",
                "styled_query": f"{brand_context.brand_name} vs {main_competitor} - which is better for everyday use?",
                "original_query": f"Compare {brand_context.brand_name} and {main_competitor}",
                "archetype": "Comparison Shopper",
                "category": "comparative_analysis",
                "execution_priority": 3,
                "authenticity_score": 8.8
            })
        
        return {
            "styled_queries": mock_queries,
            "generation_metadata": {
                "total_queries": len(mock_queries),
                "avg_authenticity": 8.5,
                "categories_covered": ["direct_recommendation", "indirect_recommendation", "comparative_analysis"],
                "llm_generated": False,
                "mock_generated": True
            },
            "framework_compliance": True
        }

# Factory function for backwards compatibility
def create_query_generator(use_llm: bool = True, ollama_config: Optional[OllamaConfig] = None) -> LLMQueryGenerator:
    """Factory function to create query generator."""
    return LLMQueryGenerator(ollama_config=ollama_config)
