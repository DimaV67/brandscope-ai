"""
Security utilities and validation functions.
"""
import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """Security configuration with validation."""
    
    secret_key: str = Field(..., min_length=32)
    api_rate_limit: int = Field(default=100, ge=1, le=1000)
    max_file_size: int = Field(default=10485760, ge=1024)  # 10MB default
    allowed_file_extensions: set = Field(default_factory=lambda: {'.json', '.txt', '.md', '.yaml'})
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Load security config from environment variables."""
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
            
        return cls(
            secret_key=secret_key,
            api_rate_limit=int(os.getenv('API_RATE_LIMIT', '100')),
            max_file_size=int(os.getenv('MAX_FILE_SIZE', '10485760'))
        )


class SecurityValidator:
    """Security validation utilities."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._cipher = Fernet(self._derive_key(config.secret_key))
    
    @staticmethod
    def _derive_key(secret: str) -> bytes:
        """Derive encryption key from secret."""
        return Fernet.generate_key()
    
    def validate_file_path(self, file_path: Path, base_path: Path) -> bool:
        """Validate file path to prevent directory traversal."""
        try:
            resolved_path = (base_path / file_path).resolve()
            base_resolved = base_path.resolve()
            
            # Check if resolved path is within base directory
            return str(resolved_path).startswith(str(base_resolved))
        except (OSError, ValueError):
            return False
    
    def validate_file_extension(self, file_path: Path) -> bool:
        """Validate file extension against allowed list."""
        return file_path.suffix.lower() in self.config.allowed_file_extensions
    
    def validate_file_size(self, file_path: Path) -> bool:
        """Validate file size against maximum allowed."""
        try:
            return file_path.stat().st_size <= self.config.max_file_size
        except OSError:
            return False
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input for basic security."""
        # Remove potential HTML/script tags and dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '`']
        sanitized = user_input
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized.strip()
    
    def generate_secure_filename(self, original_name: str) -> str:
        """Generate secure filename."""
        # Remove path traversal patterns first
        safe_name = original_name.replace('../', '').replace('..\\', '')
        
        # Remove HTML/script tags completely
        safe_name = re.sub(r'<script[^>]*>.*?</script>', '', safe_name, flags=re.IGNORECASE)
        safe_name = re.sub(r'<[^>]*>', '', safe_name)
        
        # Remove dangerous characters but keep letters, numbers, dots, hyphens, underscores
        # Note: We keep dots here, so file.txt stays file.txt
        safe_name = re.sub(r'[^a-zA-Z0-9.\-_]', '', safe_name)
        
        # Remove leading dots to prevent hidden files
        safe_name = safe_name.lstrip('.')
        
        # Clean up multiple consecutive dots AFTER other cleaning
        safe_name = re.sub(r'\.{2,}', '.', safe_name)
        
        # Handle empty result
        if not safe_name:
            safe_name = 'safe_file'
        
        return safe_name[:100]
    
    def create_hmac_signature(self, data: str) -> str:
        """Create HMAC signature for data integrity."""
        return hmac.new(
            self.config.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.create_hmac_signature(data)
        return hmac.compare_digest(expected, signature)