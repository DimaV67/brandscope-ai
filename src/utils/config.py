"""
Configuration management with environment variable support.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .security import SecurityConfig


class AppConfig(BaseModel):
    """Application configuration."""
    
    # Core settings
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    
    # Paths
    projects_root: Path = Field(default=Path("./projects"))
    cache_root: Path = Field(default=Path("./cache"))
    logs_root: Path = Field(default=Path("./logs"))
    
    # API settings
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    cache_ttl: int = Field(default=3600, ge=60)
    
    # Security
    security: SecurityConfig = Field(default_factory=SecurityConfig.from_env)
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = {'development', 'staging', 'production', 'testing'}
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @classmethod
    def load_config(cls, config_file: Optional[Path] = None) -> 'AppConfig':
        """Load configuration from file and environment."""
        config_data = {}
        
        # Load from config file if provided
        if config_file and config_file.exists():
            with open(config_file) as f:
                config_data = yaml.safe_load(f) or {}
        
        # Override with environment variables
        env_mappings = {
            'DEBUG': 'debug',
            'ENVIRONMENT': 'environment', 
            'LOG_LEVEL': 'log_level',
            'PROJECTS_ROOT': 'projects_root',
            'TIMEOUT_SECONDS': 'timeout_seconds',
            'MAX_RETRIES': 'max_retries',
            'CACHE_TTL': 'cache_ttl'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion based on field
                if config_key in ['debug']:
                    config_data[config_key] = env_value.lower() in ('true', '1', 'yes')
                elif config_key in ['timeout_seconds', 'max_retries', 'cache_ttl']:
                    config_data[config_key] = int(env_value)
                elif config_key in ['projects_root']:
                    config_data[config_key] = Path(env_value)
                else:
                    config_data[config_key] = env_value
        
        return cls(**config_data)
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for path in [self.projects_root, self.cache_root, self.logs_root]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Create .gitkeep files
        for path in [self.projects_root, self.cache_root, self.logs_root]:
            gitkeep = path / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        config_file = Path("config/settings.yaml")
        _config = AppConfig.load_config(config_file)
        _config.ensure_directories()
    return _config


def reload_config(config_file: Optional[Path] = None) -> AppConfig:
    """Reload configuration."""
    global _config
    _config = AppConfig.load_config(config_file)
    _config.ensure_directories()
    return _config