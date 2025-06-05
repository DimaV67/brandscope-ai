"""
Mock archetype builder for testing Stage 1 orchestration.
"""
from typing import Dict, Any, List
from ..models.brand import BrandContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ArchetypeBuilder:
    """Mock archetype builder for testing."""
    
    def generate_archetypes(
        self,
        category_intelligence: Dict[str, Any],
        brand_context: BrandContext
    ) -> Dict[str, Any]:
        """Generate mock customer archetypes."""
        
        logger.info("Building customer archetypes (mock)",
                   metadata={"category": category_intelligence.get("category")})
        
        # Mock archetypes based on category
        category = category_intelligence.get("category", "generic")
        
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
                "confidence": 0.92,
                "ai_behavior_prediction": "Seeks expert endorsements and technical specifications"
            },
            {
                "archetype_id": "ARCH_002", 
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
                "confidence": 0.88,
                "ai_behavior_prediction": "Values convenience and practical benefits"
            },
            {
                "archetype_id": "ARCH_003",
                "name": f"Budget-Conscious {category.title()} Shopper", 
                "description": f"Value-focused customer seeking affordable {category} options",
                "attributes": {
                    "COREB1": "BUSY_PRACTICAL",
                    "MODIFIERE1": "SMART_SHOPPER",
                    "MODIFIERD3": "BUDGET",
                    "COREA2": "RESEARCH",
                    "DEMOD2": "PRACTICAL_HEARTLAND",
                    "COREB3": "BRAND_BLIND"
                },
                "market_presence": "MEDIUM",
                "strategic_value": "MEDIUM",
                "confidence": 0.85,
                "ai_behavior_prediction": "Focuses on price comparisons and value propositions"
            }
        ]
        
        return {
            "universal_attributes": category_intelligence["universal_attributes"],
            "category_attributes": category_intelligence["category_attributes"],
            "ranked_archetypes": mock_archetypes,
            "top_archetypes": mock_archetypes[:2],  # Top 2 for execution
            "generation_metadata": {
                "total_archetypes": len(mock_archetypes),
                "avg_confidence": sum(a["confidence"] for a in mock_archetypes) / len(mock_archetypes),
                "coverage_score": 0.87
            }
        }