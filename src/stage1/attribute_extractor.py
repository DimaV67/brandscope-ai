"""
Mock attribute extractor for testing Stage 1 orchestration.
"""
from typing import Dict, Any, Optional
from ..models.brand import BrandContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AttributeExtractor:
    """Mock attribute extractor for testing."""
    
    def generate_category_intelligence(
        self,
        category: str,
        brand_context: BrandContext,
        customer_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mock category intelligence."""
        
        logger.info("Generating category intelligence (mock)",
                   metadata={"category": category, "brand": brand_context.brand_name})
        
        # Mock category intelligence structure
        return {
            "category": category,
            "brand_context": brand_context.brand_name,
            "universal_attributes": {
                "COREB1": ["HEALTH_FOCUSED", "QUALITY_CONNOISSEUR", "BUSY_PRACTICAL"],
                "MODIFIERE1": ["STATUS_SIGNAL", "HEALTH_IDENTITY", "SMART_SHOPPER"],
                "MODIFIERD3": ["BUDGET", "MIDRANGE", "PREMIUM"],
                "COREA2": ["RESEARCH", "QUICK_STORE", "ROUTINE_ONLINE"],
                "DEMOD2": ["URBAN_FAST", "SUBURBAN_FAMILY", "HEALTH_CONSCIOUS_REGION"],
                "COREB3": ["BRAND_AWARE", "BRAND_NEUTRAL", "BRAND_BLIND"]
            },
            "category_attributes": {
                f"{category.upper()}_USE_CASE": ["primary_use", "secondary_use"],
                f"{category.upper()}_EXPERTISE": ["beginner", "enthusiast", "expert"],
                f"{category.upper()}_PRIORITY": ["function", "design", "price"]
            },
            "competitive_landscape": {
                "primary_competitors": brand_context.competitive_context.primary_competitors,
                "market_segments": ["premium", "mainstream", "budget"],
                "positioning_opportunities": ["quality_leader", "value_leader", "innovation_leader"]
            },
            "price_ranges": {
                "BUDGET": f"< $100",
                "MIDRANGE": f"$100 - $300", 
                "PREMIUM": f"> $300"
            }
        }