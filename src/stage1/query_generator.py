"""
Mock query generator for testing Stage 1 orchestration.
"""
from typing import Dict, Any, List
from ..models.brand import BrandContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class QueryGenerator:
    """Mock query generator for testing."""
    
    def generate_query_package(
        self,
        top_archetypes: List[Dict[str, Any]],
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate mock styled queries."""
        
        logger.info("Generating query package (mock)",
                   metadata={"archetypes_count": len(top_archetypes)})
        
        category = category_intelligence.get("category", "product")
        brand_name = brand_context.brand_name
        
        # Generate mock queries for each archetype
        styled_queries = []
        query_id = 1
        
        for archetype in top_archetypes:
            archetype_name = archetype.get("name", "Customer")
            
            # Direct recommendation queries
            styled_queries.extend([
                {
                    "query_id": f"Q{query_id:03d}",
                    "styled_query": f"Looking for the best {category} for quality and reliability - what do experts recommend?",
                    "original_query": f"Which {category} brand is recommended for quality?",
                    "archetype": archetype_name,
                    "category": "direct_recommendation",
                    "execution_priority": query_id,
                    "authenticity_score": 8.5
                },
                {
                    "query_id": f"Q{query_id+1:03d}",
                    "styled_query": f"Need a {category} that won't break the bank but still works well - any suggestions?",
                    "original_query": f"What's the best value {category}?", 
                    "archetype": archetype_name,
                    "category": "indirect_recommendation",
                    "execution_priority": query_id + 1,
                    "authenticity_score": 8.2
                }
            ])
            query_id += 2
        
        # Add comparative queries
        if brand_context.competitive_context.primary_competitors:
            competitor = brand_context.competitive_context.primary_competitors[0]
            styled_queries.append({
                "query_id": f"Q{query_id:03d}",
                "styled_query": f"{brand_name} vs {competitor} - which is better for everyday use?",
                "original_query": f"Compare {brand_name} and {competitor}",
                "archetype": "Multiple",
                "category": "comparative_analysis", 
                "execution_priority": query_id,
                "authenticity_score": 8.8
            })
        
        return {
            "styled_queries": styled_queries,
            "generation_metadata": {
                "total_queries": len(styled_queries),
                "avg_authenticity": sum(q["authenticity_score"] for q in styled_queries) / len(styled_queries),
                "categories_covered": list(set(q["category"] for q in styled_queries))
            },
            "framework_compliance": True
        }