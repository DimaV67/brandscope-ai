"""
Custom exception hierarchy with detailed error information.
"""
from typing import Any, Dict, Optional
from uuid import uuid4


class BrandscopeError(Exception):
    """Base exception for all Brandscope errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid4())
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "exception_type": self.__class__.__name__
        }


class ValidationError(BrandscopeError):
    """Raised when data validation fails."""
    pass


class SecurityError(BrandscopeError):
    """Raised when security validation fails."""
    pass


class ProjectError(BrandscopeError):
    """Base class for project-related errors."""
    pass


class ProjectNotFoundError(ProjectError):
    """Raised when project is not found."""
    pass


class ProjectExistsError(ProjectError):
    """Raised when trying to create duplicate project."""
    pass


class ProjectCorruptedError(ProjectError):
    """Raised when project data is corrupted."""
    pass


class StageExecutionError(BrandscopeError):
    """Raised when stage execution fails."""
    
    def __init__(
        self,
        message: str,
        stage: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.stage = stage
        self.operation = operation
        self.details.update({
            "stage": stage,
            "operation": operation
        })


class LLMError(BrandscopeError):
    """Base class for LLM-related errors."""
    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code
        self.details.update({
            "provider": provider,
            "status_code": status_code
        })


class LLMRateLimitError(LLMAPIError):
    """Raised when LLM API rate limit is exceeded."""
    pass


class LLMTimeoutError(LLMAPIError):
    """Raised when LLM API request times out."""
    pass


class ConfigurationError(BrandscopeError):
    """Raised when configuration is invalid."""
    pass


class FileOperationError(BrandscopeError):
    """Raised when file operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation
        self.details.update({
            "file_path": file_path,
            "operation": operation
        })