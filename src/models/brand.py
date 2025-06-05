"""
Brand and product context models.
"""
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PriceTierEnum(str, Enum):
    """Product pricing tier enumeration."""
    BUDGET = "budget"
    MIDRANGE = "midrange" 
    PREMIUM = "premium"
    LUXURY = "luxury"


class ProductInfo(BaseModel):
    """Product information with validation."""
    
    name: str = Field(..., min_length=1, max_length=100)
    product_type: str = Field(..., min_length=1, max_length=50)
    price_tier: PriceTierEnum
    price_range: Optional[str] = None
    key_features: List[str] = Field(default_factory=list, max_items=10)
    
    @field_validator('name', 'product_type')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Validate text fields for security."""
        if any(char in v for char in ['<', '>', '&', '"', "'"]):
            raise ValueError("Field cannot contain HTML/script characters")
        return v.strip()
    
    @field_validator('key_features')
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """Validate feature list."""
        return [feature.strip() for feature in v if feature.strip()]


class CompetitiveContext(BaseModel):
    """Competitive landscape context."""
    
    primary_competitors: List[str] = Field(default_factory=list, max_items=10)
    competitive_positioning: Optional[str] = None
    market_segment: Optional[str] = None
    
    @field_validator('primary_competitors')
    @classmethod
    def validate_competitors(cls, v: List[str]) -> List[str]:
        """Validate competitor names."""
        validated = []
        for competitor in v:
            if competitor.strip() and len(competitor) <= 100:
                validated.append(competitor.strip())
        return validated[:10]  # Limit to 10 competitors


class BrandContext(BaseModel):
    """Complete brand context with products and competitive landscape."""
    
    brand_name: str = Field(..., min_length=1, max_length=100)
    products: List[ProductInfo] = Field(..., min_items=1, max_items=20)
    competitive_context: CompetitiveContext = Field(default_factory=CompetitiveContext)
    brand_positioning: Optional[str] = None
    target_markets: List[str] = Field(default_factory=list, max_items=10)
    
    @field_validator('brand_name')
    @classmethod
    def validate_brand_name(cls, v: str) -> str:
        """Validate brand name."""
        if any(char in v for char in ['<', '>', '&', '"', "'"]):
            raise ValueError("Brand name cannot contain HTML/script characters")
        return v.strip()
    
    @field_validator('products')
    @classmethod
    def validate_products_list(cls, v: List[ProductInfo]) -> List[ProductInfo]:
        """Validate products list."""
        if not v:
            raise ValueError("At least one product is required")
        return v