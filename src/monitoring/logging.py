"""Enhanced logging configuration for the autofill pipeline."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from ..config.settings import get_config
from ..core.exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class AutoFillLogger:
    """Enhanced logger for the autofill pipeline."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._configured = False
    
    def configure(self) -> None:
        """Configure the logger based on settings."""
        if self._configured:
            return
        
        try:
            config = get_config()
            log_config = config.logging
        except Exception:
            # Fallback to basic configuration if config not available
            log_config = None
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        if log_config:
            level = getattr(logging, log_config.level.upper(), logging.INFO)
        else:
            level = logging.INFO
        
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if log_config and log_config.format:
            console_formatter = logging.Formatter(log_config.format)
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if configured)
        if log_config and log_config.file_path:
            try:
                log_config.file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_config.file_path,
                    maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                    backupCount=log_config.backup_count
                )
                
                # Use structured format for file logs
                file_formatter = StructuredFormatter()
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                
            except Exception as e:
                self.logger.warning(f"Failed to configure file logging: {e}")
        
        self._configured = True
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with extra fields."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with extra fields."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with extra fields."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with extra fields."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message with extra fields."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def _log(self, level: int, message: str, exc_info: bool = False, **kwargs) -> None:
        """Internal logging method."""
        if not self._configured:
            self.configure()
        
        # Create log record with extra fields
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
    
    def log_performance(self, operation: str, duration_seconds: float, **kwargs) -> None:
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_seconds=duration_seconds,
            **kwargs
        )
    
    def log_api_call(self, provider: str, model: str, tokens_used: Optional[int] = None, 
                     cost: Optional[float] = None, **kwargs) -> None:
        """Log API call information."""
        self.info(
            f"API call: {provider}/{model}",
            api_provider=provider,
            model=model,
            tokens_used=tokens_used,
            cost=cost,
            **kwargs
        )
    
    def log_imputation(self, column: str, method: str, count: int, 
                      confidence_avg: float, **kwargs) -> None:
        """Log imputation results."""
        self.info(
            f"Imputation completed: {column}",
            column=column,
            method=method,
            imputed_count=count,
            average_confidence=confidence_avg,
            **kwargs
        )


# Global logger instances
_loggers = {}


def get_logger(name: str) -> AutoFillLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = AutoFillLogger(name)
    return _loggers[name]


def configure_all_loggers() -> None:
    """Configure all existing logger instances."""
    for logger in _loggers.values():
        logger.configure()


# Convenience functions
def log_performance(operation: str, duration_seconds: float, **kwargs) -> None:
    """Log performance metrics using the main logger."""
    logger = get_logger("autofill.performance")
    logger.log_performance(operation, duration_seconds, **kwargs)


def log_api_call(provider: str, model: str, tokens_used: Optional[int] = None,
                cost: Optional[float] = None, **kwargs) -> None:
    """Log API call using the main logger."""
    logger = get_logger("autofill.api")
    logger.log_api_call(provider, model, tokens_used, cost, **kwargs)


def log_imputation(column: str, method: str, count: int, 
                  confidence_avg: float, **kwargs) -> None:
    """Log imputation results using the main logger."""
    logger = get_logger("autofill.imputation")
    logger.log_imputation(column, method, count, confidence_avg, **kwargs)