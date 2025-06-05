"""
Project configuration and metadata models with security validation.
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
import validators


class StageStatusEnum(str, Enum):
    """Stage execution status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class ProjectMetadata(BaseModel):
    """Project metadata with validation and security checks."""
    
    project_id: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    brand: str = Field(..., min_length=1, max_length=100)
    category: str = Field(..., min_length=1, max_length=50)
    created_date: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")
    
    @validator('project_id')
    def validate_project_id(cls, v: str) -> str:
        """Validate project ID for security (no path traversal)."""
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Project ID cannot contain path traversal characters")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Project ID must be alphanumeric with underscores/hyphens")
        return v.lower()
    
    @validator('brand', 'category')
    def validate_string_fields(cls, v: str) -> str:
        """Validate string fields for basic security."""
        if any(char in v for char in ['<', '>', '&', '"', "'"]):
            raise ValueError("Field cannot contain HTML/script characters")
        return v.strip()


class StageStatus(BaseModel):
    """Stage execution status with timing and outputs tracking."""
    
    status: StageStatusEnum = StageStatusEnum.PENDING
    started_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    outputs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0, le=5)
    
    @validator('outputs')
    def validate_outputs(cls, v: List[str]) -> List[str]:
        """Validate output file paths for security."""
        validated = []
        for output in v:
            if '..' in output or output.startswith('/'):
                continue  # Skip potentially dangerous paths
            validated.append(output)
        return validated


class ProjectConfig(BaseModel):
    """Complete project configuration with security validation."""
    
    project_metadata: ProjectMetadata
    stage_status: Dict[str, StageStatus] = Field(
        default_factory=lambda: {
            "stage1": StageStatus(),
            "stage2": StageStatus(), 
            "stage3": StageStatus()
        }
    )
    settings: Dict[str, Union[str, int, bool, List[str]]] = Field(
        default_factory=dict
    )
    
    @validator('stage_status')
    def validate_stages(cls, v: Dict[str, StageStatus]) -> Dict[str, StageStatus]:
        """Validate stage configuration."""
        required_stages = {"stage1", "stage2", "stage3"}
        if not required_stages.issubset(set(v.keys())):
            missing = required_stages - set(v.keys())
            raise ValueError(f"Missing required stages: {missing}")
        return v
    
    @root_validator
    def validate_config_consistency(cls, values: Dict) -> Dict:
        """Validate overall configuration consistency."""
        metadata = values.get('project_metadata')
        if metadata:
            # Update last_modified when config changes
            values['project_metadata'].last_modified = datetime.now()
        return values
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }