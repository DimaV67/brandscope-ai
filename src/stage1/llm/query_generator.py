# src/stage1/llm/query_generator.py
"""
LLM-powered query generator with fixed import handling.
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
class QueryGenerationPrompts:
    """Structured prompts for query generation"""
    
    AUTHENTIC_QUERY_GENERATION = """You are a customer behavior expert generating authentic search queries for {category} research.

Customer Archetype: {archetype_name}
Description: {archetype_description}
AI Behavior: {ai_behavior}
Communication Style: {communication_style}

Category: {category}
Brand Context: {brand_name}
Competitors: {competitors}

Generate 5-7 authentic queries this customer archetype would ask when researching {category} products. 

Use this JSON format:
{{
    "styled_queries": [
        {{
            "query_id": "Q001",
            "styled_query": "Exact query the customer would type/ask",
            "original_query": "Base query without styling",
            "category": "direct_recommendation|indirect_recommendation|comparative_analysis|problem_solving|feature_inquiry",
            "authenticity_score": 0.0-10.0,
            "reasoning": "Why this archetype would ask this query"
        }}
    ]
}}

Make queries feel natural and authentic to how this specific customer type actually communicates."""

    QUERY_VARIETY_GENERATION = """Generate diverse query types for {category} research covering different customer intents.

Archetypes to cover:
{archetypes_summary}

Brand: {brand_name}
Competitors: {competitors}

Create queries that cover these categories:
1. Direct recommendations ("What's the best...")
2. Indirect recommendations ("I need help choosing...")  
3. Comparative analysis ("X vs Y...")
4. Problem-solving ("I have this issue...")
5. Feature inquiries ("Does X have...")
6. Price/value questions ("Is X worth...")

Return JSON:
{{
    "query_categories": {{
        "direct_recommendation": [
            {{"query": "query text", "archetype": "target_archetype", "authenticity": 0-10}}
        ],
        "comparative_analysis": [
            {{"query": "query text", "archetype": "target_archetype", "authenticity": 0-10}}
        ],
        "problem_solving": [
            {{"query": "query text", "archetype": "target_archetype", "authenticity": 0-10}}
        ]
    }}
}}"""

    QUERY_REFINEMENT = """Refine these queries to maximize authenticity and research value:

Original Queries:
{queries}

Brand: {brand_name}
Category: {category}

For each query, improve:
1. Natural language authenticity
2. Specificity and relevance
3. Likelihood to generate brand mentions
4. Customer psychology alignment

Return refined queries in JSON format:
{{
    "refined_queries": [
        {{
            "query_id": "Q001",
            "original": "original query",
            "refined": "improved query",
            "improvements": ["improvement1", "improvement2"],
            "authenticity_score": 0.0-10.0,
            "brand_mention_likelihood": 0.0-10.0
        }}
    ]
}}"""

class LLMQueryGenerator:
    """LLM-powered query generator with robust fallback handling."""
    
    def __init__(
        self,
        ollama_config: Optional[OllamaConfig] = None,
        fallback_to_mock: bool = True
    ):
        self.ollama_config = ollama_config or OllamaConfig()
        self.fallback_to_mock = fallback_to_mock
        self.prompts = QueryGenerationPrompts()
        self.mock_generator = None
        self.total_tokens_used =0 # Token counter
        
        # Initialize mock generator for fallback
        if fallback_to_mock:
            self.mock_generator = self._create_mock_generator()
    
    def _create_mock_generator(self):
        """Create mock generator with proper import handling."""
        try:
            mock_generator = self._try_import_mock_generator()
            if mock_generator:
                return mock_generator
            else:
                logger.warning("Could not import mock generator - using internal fallback")
                return InternalMockGenerator()
        except Exception as e:
            logger.warning(f"Mock generator import failed: {e} - using internal fallback")
            return InternalMockGenerator()
    
    def _try_import_mock_generator(self):
        """Try different strategies to import the mock generator."""
        import importlib.util
        
        # Strategy 1: Try direct import from stage1
        try:
            current_dir = os.path.dirname(__file__)
            mock_file_path = os.path.join(current_dir, '..', 'query_generator.py')
            mock_file_path = os.path.abspath(mock_file_path)
            
            if os.path.exists(mock_file_path):
                spec = importlib.util.spec_from_file_location("mock_query_generator", mock_file_path)
                if spec and spec.loader:
                    mock_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mock_module)
                    
                    if hasattr(mock_module, 'QueryGenerator'):
                        logger.info("Successfully imported mock QueryGenerator")
                        return mock_module.QueryGenerator()
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try sys.path manipulation
        try:
            original_path = sys.path.copy()
            stage1_path = os.path.join(os.path.dirname(__file__), '..')
            if stage1_path not in sys.path:
                sys.path.insert(0, stage1_path)
            
            from query_generator import QueryGenerator
            sys.path = original_path
            logger.info("Successfully imported mock QueryGenerator via sys.path")
            return QueryGenerator()
            
        except Exception as e:
            logger.debug(f"Strategy 2 failed: {e}")
            sys.path = original_path
        
        return None
    
    async def generate_query_package(
        self,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate comprehensive query package using LLM analysis."""
        
        logger.info("Generating query package with LLM",
                   metadata={"archetypes_count": len(top_archetypes)})
        
        try:
            async with OllamaClient(self.ollama_config) as client:
                # Check if Ollama is available
                if not await client.health_check():
                    logger.warning("Ollama not available, falling back to mock")
                    return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)
                
                # Step 1: Generate archetype-specific queries
                archetype_queries = await self._generate_archetype_queries(
                    client, top_archetypes, category_intelligence, brand_context
                )
                
                # Step 2: Generate diverse query categories
                category_queries = await self._generate_category_coverage_queries(
                    client, top_archetypes, category_intelligence, brand_context
                )
                
                # Step 3: Combine and deduplicate queries
                combined_queries = self._combine_and_deduplicate_queries(
                    archetype_queries, category_queries
                )
                
                # Step 4: Refine for authenticity and effectiveness
                refined_queries = await self._refine_queries_for_authenticity(
                    client, combined_queries, category_intelligence, brand_context
                )
                
                # Step 5: Create final package
                final_package = self._create_final_query_package(
                    refined_queries, top_archetypes, category_intelligence, brand_context
                )
                
                logger.info("Query package generated successfully",
                           metadata={
                               "total_queries": len(final_package.get("styled_queries", [])),
                               "avg_authenticity": final_package.get("generation_metadata", {}).get("avg_authenticity", 0)
                           })
                
                return final_package
                
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            if self.fallback_to_mock:
                logger.info("Falling back to mock implementation")
                return self._fallback_to_mock(top_archetypes, category_intelligence, brand_context)
            raise
    
    async def _generate_archetype_queries(
        self,
        client: OllamaClient,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate queries for each top archetype."""
        
        all_queries = []
        category = category_intelligence.get("category", "product")
        
        for archetype in top_archetypes[:3]:
            archetype_name = archetype.get("name", "Customer")
            archetype_desc = archetype.get("description", "Generic customer")
            ai_behavior = archetype.get("ai_behavior_prediction", "Seeks general information")
            comm_style = archetype.get("communication_style", "casual")
            
            prompt = self.prompts.AUTHENTIC_QUERY_GENERATION.format(
                category=category,
                archetype_name=archetype_name,
                archetype_description=archetype_desc,
                ai_behavior=ai_behavior,
                communication_style=comm_style,
                brand_name=brand_context.brand_name,
                competitors=", ".join(brand_context.competitive_context.primary_competitors[:3])
            )
            
            request = LLMRequest(
                prompt=prompt,
                model="mistral",
                temperature=0.7,
                max_tokens=800,
                system_prompt="You are a customer behavior expert. Generate authentic, natural customer queries."
            )
            
            response = await client.generate(request)
            self.total_tokens_used += response.tokens_used  #Accumulate tokens
            
            if response.success:
                try:
                    result = json.loads(response.content)
                    queries = result.get("styled_queries", [])
                    
                    for query in queries:
                        query["archetype"] = archetype_name
                        query["archetype_id"] = archetype.get("archetype_id", "UNKNOWN")
                    
                    all_queries.extend(queries)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse queries for archetype {archetype_name}")
                    all_queries.extend(self._generate_fallback_archetype_queries(
                        archetype, category, brand_context
                    ))
            else:
                logger.warning(f"Query generation failed for archetype {archetype_name}: {response.error}")
                all_queries.extend(self._generate_fallback_archetype_queries(
                    archetype, category, brand_context
                ))
        
        return all_queries
    
    async def _generate_category_coverage_queries(
        self,
        client: OllamaClient,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate queries to ensure comprehensive category coverage."""
        
        category = category_intelligence.get("category", "product")
        
        archetypes_summary = "\n".join([
            f"- {arch.get('name', 'Unknown')}: {arch.get('description', 'No description')[:100]}..."
            for arch in top_archetypes[:3]
        ])
        
        prompt = self.prompts.QUERY_VARIETY_GENERATION.format(
            category=category,
            archetypes_summary=archetypes_summary,
            brand_name=brand_context.brand_name,
            competitors=", ".join(brand_context.competitive_context.primary_competitors[:3])
        )
        
        request = LLMRequest(
            prompt=prompt,
            model="mistral",
            temperature=0.6,
            max_tokens=1000,
            system_prompt="Generate diverse, authentic customer queries covering all research intents."
        )
        
        response = await client.generate(request)
        self.total_tokens_used += response.tokens_used #Accumulate tokens
        
        all_queries = []
        
        if response.success:
            try:
                result = json.loads(response.content)
                query_categories = result.get("query_categories", {})
                
                query_id = 1
                for category_type, queries in query_categories.items():
                    for query_data in queries[:2]:
                        all_queries.append({
                            "query_id": f"CAT{query_id:03d}",
                            "styled_query": query_data.get("query", ""),
                            "original_query": query_data.get("query", ""),
                            "category": category_type,
                            "archetype": query_data.get("archetype", "Multiple"),
                            "authenticity_score": query_data.get("authenticity", 7.0),
                            "execution_priority": query_id + 100,
                            "source": "category_coverage"
                        })
                        query_id += 1
                        
            except json.JSONDecodeError:
                logger.warning("Failed to parse category coverage queries")
        
        return all_queries
    
    def _combine_and_deduplicate_queries(
        self,
        archetype_queries: List[Dict[str, Any]],
        category_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine queries and remove duplicates."""
        
        all_queries = archetype_queries + category_queries
        
        unique_queries = []
        seen_queries = set()
        
        for query in all_queries:
            query_text = query.get("styled_query", "").lower().strip()
            
            is_duplicate = False
            for seen_query in seen_queries:
                if self._calculate_similarity(query_text, seen_query) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate and query_text:
                unique_queries.append(query)
                seen_queries.add(query_text)
        
        return unique_queries
    
    async def _refine_queries_for_authenticity(
        self,
        client: OllamaClient,
        queries: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Refine queries for maximum authenticity and effectiveness."""
        
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
            max_tokens=1200,
            system_prompt="Refine customer queries for maximum authenticity and brand research effectiveness."
        )
        
        response = await client.generate(request)
        self.total_tokens_used += response.tokens_used #Accumulate tokens
        
        if response.success:
            try:
                result = json.loads(response.content)
                refined_queries = result.get("refined_queries", [])
                return self._apply_refinements(queries, refined_queries)
            except json.JSONDecodeError:
                logger.warning("Failed to parse query refinements")
        
        return queries
    
    def _apply_refinements(
        self,
        original_queries: List[Dict[str, Any]],
        refinements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply refinements back to original queries."""
        
        refinement_lookup = {
            ref.get("query_id"): ref
            for ref in refinements
        }
        
        refined_queries = []
        
        for query in original_queries:
            query_id = query.get("query_id", "")
            refinement = refinement_lookup.get(query_id)
            
            if refinement and refinement.get("refined"):
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
                refined_queries.append(query)
        
        return refined_queries
    
    def _create_final_query_package(
        self,
        queries: List[Dict[str, Any]],
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Create final structured query package."""
        
        prioritized_queries = []
        priority = 1
        
        for query in sorted(queries, key=lambda q: q.get("authenticity_score", 0), reverse=True):
            query_copy = query.copy()
            query_copy["execution_priority"] = priority
            
            if not query_copy.get("query_id"):
                query_copy["query_id"] = f"Q{priority:03d}"
            
            prioritized_queries.append(query_copy)
            priority += 1
        
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
                "refinement_applied": any(q.get("refined", False) for q in prioritized_queries)
            },
            "framework_compliance": True
        }
    
    def _generate_fallback_archetype_queries(
        self,
        archetype: Dict[str, Any],
        category: str,
        brand_context: BrandContext
    ) -> List[Dict[str, Any]]:
        """Generate fallback queries for an archetype when LLM fails."""
        
        archetype_name = archetype.get("name", "Customer")
        attributes = archetype.get("attributes", {})
        
        queries = []
        
        if attributes.get("MODIFIERD3") == "PREMIUM":
            queries.append({
                "query_id": "FB01",
                "styled_query": f"What's the highest quality {category} available? Money isn't an issue.",
                "original_query": f"Best quality {category}",
                "category": "direct_recommendation",
                "authenticity_score": 7.5
            })
        
        if attributes.get("COREA2") == "RESEARCH":
            queries.append({
                "query_id": "FB02", 
                "styled_query": f"I've been researching {category} options - what do experts actually recommend?",
                "original_query": f"Expert recommendations {category}",
                "category": "indirect_recommendation",
                "authenticity_score": 8.0
            })
        
        if brand_context.competitive_context.primary_competitors:
            competitor = brand_context.competitive_context.primary_competitors[0]
            queries.append({
                "query_id": "FB03",
                "styled_query": f"{brand_context.brand_name} vs {competitor} - which should I choose?",
                "original_query": f"Compare {brand_context.brand_name} {competitor}",
                "category": "comparative_analysis",
                "authenticity_score": 8.5
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
        self,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Fallback to mock implementation."""
        if self.mock_generator:
            try:
                return self.mock_generator.generate_query_package(
                    top_archetypes, category_intelligence, brand_context
                )
            except Exception as e:
                logger.warning(f"Mock generator failed: {e}")
        
        return self._create_emergency_fallback(category_intelligence, brand_context)
    
    def _create_emergency_fallback(
        self,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Create emergency fallback when all else fails."""
        category = category_intelligence.get("category", "product")
        
        return {
            "styled_queries": [
                {
                    "query_id": "EMERGENCY01",
                    "styled_query": f"What's the best {category} for the money?",
                    "original_query": f"Best {category}",
                    "archetype": "Generic Customer",
                    "category": "direct_recommendation",
                    "execution_priority": 1,
                    "authenticity_score": 7.0
                }
            ],
            "generation_metadata": {
                "total_queries": 1,
                "avg_authenticity": 7.0,
                "categories_covered": ["direct_recommendation"],
                "emergency_fallback": True
            },
            "framework_compliance": True
        }


class InternalMockGenerator:
    """Internal mock implementation when external mock is unavailable."""
    
    def generate_query_package(
        self,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate mock query package."""
        
        logger.info("Using internal mock generator")
        category = category_intelligence.get("category", "product")
        brand_name = brand_context.brand_name
        competitors = brand_context.competitive_context.primary_competitors
        
        mock_queries = [
            {
                "query_id": "Q001",
                "styled_query": f"What's the best {category} for someone who wants quality?",
                "original_query": f"Best quality {category}",
                "archetype": "Quality Seeker",
                "category": "direct_recommendation",
                "execution_priority": 1,
                "authenticity_score": 8.0
            },
            {
                "query_id": "Q002", 
                "styled_query": f"I'm trying to decide between {brand_name} and {competitors[0] if competitors else 'other brands'} - any thoughts?",
                "original_query": f"Compare {brand_name} vs competitors",
                "archetype": "Comparison Shopper",
                "category": "comparative_analysis",
                "execution_priority": 2,
                "authenticity_score": 8.5
            },
            {
                "query_id": "Q003",
                "styled_query": f"Is {brand_name} worth the extra cost for {category}?",
                "original_query": f"Is {brand_name} worth it",
                "archetype": "Value Conscious",
                "category": "feature_inquiry",
                "execution_priority": 3,
                "authenticity_score": 7.5
            },
            {
                "query_id": "Q004",
                "styled_query": f"I need help choosing a {category} for daily use - what should I look for?",
                "original_query": f"Help choosing {category}",
                "archetype": "Practical User",
                "category": "indirect_recommendation",
                "execution_priority": 4,
                "authenticity_score": 7.8
            },
            {
                "query_id": "Q005",
                "styled_query": f"What are the main differences between budget and premium {category}?",
                "original_query": f"Budget vs premium {category}",
                "archetype": "Research Oriented",
                "category": "comparative_analysis",
                "execution_priority": 5,
                "authenticity_score": 8.2
            }
        ]
        
        return {
            "styled_queries": mock_queries,
            "generation_metadata": {
                "total_queries": len(mock_queries),
                "avg_authenticity": 8.0,
                "categories_covered": ["direct_recommendation", "comparative_analysis", "feature_inquiry", "indirect_recommendation"],
                "archetypes_covered": len(top_archetypes),
                "mock_generated": True
            },
            "framework_compliance": True
        }


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