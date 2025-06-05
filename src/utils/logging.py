"""
Comprehensive logging system with security and performance optimization.
"""
import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import get_config


class LogEntry(BaseModel):
    """Structured log entry model."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str
    message: str
    logger_name: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    stage: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SecurityAwareFormatter(logging.Formatter):
    """Logging formatter with security filtering."""
    
    SENSITIVE_KEYS = {
        'password', 'secret', 'token', 'key', 'api_key', 
        'authorization', 'credential', 'private'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with security filtering."""
        # Create log entry
        log_entry = LogEntry(
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            correlation_id=getattr(record, 'correlation_id', None),
            user_id=getattr(record, 'user_id', None),
            project_id=getattr(record, 'project_id', None),
            stage=getattr(record, 'stage', None),
            operation=getattr(record, 'operation', None),
            duration_ms=getattr(record, 'duration_ms', None),
            metadata=self._filter_sensitive_data(getattr(record, 'metadata', {})),
            error_details=self._extract_error_details(record)
        )
        
        return json.dumps(log_entry.dict(), default=str, separators=(',', ':'))
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive information from log data."""
        if not isinstance(data, dict):
            return {}
            
        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings to prevent log bloat
                filtered[key] = value[:97] + "..."
            else:
                filtered[key] = value
        return filtered
    
    def _extract_error_details(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract error details from log record."""
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            return {
                "exception_type": exc_type.__name__ if exc_type else None,
                "exception_message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_tb(exc_traceback) if exc_traceback else None,
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        return None


class ContextAwareLogger:
    """Logger with context management and correlation tracking."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs: Any) -> None:
        """Set logging context."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context."""
        self._context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs: Any) -> None:
        """Log message with context."""
        extra = {**self._context, **kwargs}
        
        # Handle exc_info separately to avoid conflicts
        exc_info = extra.pop('exc_info', False)
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, exc_info=exc_info, **kwargs)


class LoggingManager:
    """Central logging management."""
    
    def __init__(self):
        self.config = get_config()
        self._initialized = False
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        if self._initialized:
            return
        
        # Create logs directory
        self.config.logs_root.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler for development
        if self.config.debug:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler for structured logging
        log_file = self.config.logs_root / "application.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(SecurityAwareFormatter())
        root_logger.addHandler(file_handler)
        
        # Error-only file handler
        error_file = self.config.logs_root / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(SecurityAwareFormatter())
        root_logger.addHandler(error_handler)
        
        # Operation-specific handler
        operation_file = self.config.logs_root / "operations.log"
        operation_handler = logging.handlers.RotatingFileHandler(
            operation_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
        operation_handler.setLevel(logging.INFO)
        operation_handler.setFormatter(SecurityAwareFormatter())
        operation_handler.addFilter(lambda record: hasattr(record, 'operation'))
        root_logger.addHandler(operation_handler)
        
        self._initialized = True
    
    def get_logger(self, name: str) -> ContextAwareLogger:
        """Get context-aware logger instance."""
        if not self._initialized:
            self.setup_logging()
        return ContextAwareLogger(name)


# Global logging manager
_logging_manager = LoggingManager()


def get_logger(name: str) -> ContextAwareLogger:
    """Get logger instance."""
    return _logging_manager.get_logger(name)


def setup_logging() -> None:
    """Setup global logging."""
    _logging_manager.setup_logging()