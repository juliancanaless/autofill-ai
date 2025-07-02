"""Configuration management for the autofill pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path

from ..core.types import LLMProvider
from ..core.exceptions import ConfigurationError


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    rate_limit_requests_per_minute: int = 60
    max_tokens: int = 500
    temperature: float = 0.3


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    backend: str = "memory"  # "memory", "redis", "file"
    ttl_seconds: int = 3600
    max_size: int = 1000
    redis_url: Optional[str] = None
    file_path: Optional[Path] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    max_rows: int = 100000
    max_columns: int = 1000
    max_missing_percentage: float = 90.0
    min_non_null_values: int = 2
    allowed_file_types: List[str] = field(default_factory=lambda: ['.csv', '.xlsx', '.parquet'])


@dataclass
class ImputationConfig:
    """Configuration for imputation strategies."""
    statistical_method: str = "knn"  # "knn", "median", "mean"
    knn_neighbors: int = 3
    categorical_threshold: int = 20
    confidence_threshold: float = 0.5
    max_llm_calls_per_column: int = 100


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enabled: bool = True
    metrics_backend: str = "prometheus"  # "prometheus", "statsd", "none"
    metrics_port: int = 8000
    health_check_enabled: bool = True
    performance_tracking: bool = True


@dataclass
class AutoFillConfig:
    """Main configuration class."""
    llm: LLMConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_env(cls, llm_provider: str = "groq") -> "AutoFillConfig":
        """Create configuration from environment variables."""
        
        # Validate provider
        try:
            provider = LLMProvider(llm_provider.lower())
        except ValueError:
            raise ConfigurationError(f"Invalid LLM provider: {llm_provider}")
        
        # Get API key from environment
        api_key_map = {
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.OLLAMA: None  # No API key needed for Ollama
        }
        
        api_key = None
        if api_key_map[provider]:
            api_key = os.getenv(api_key_map[provider])
            if not api_key and provider != LLMProvider.OLLAMA:
                raise ConfigurationError(f"Missing {api_key_map[provider]} environment variable")
        
        # Default model names
        default_models = {
            LLMProvider.GROQ: "llama3-8b-8192",
            LLMProvider.OPENAI: "gpt-3.5-turbo",
            LLMProvider.OLLAMA: "llama3.1:8b"
        }
        
        llm_config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model_name=os.getenv("MODEL_NAME", default_models[provider]),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT", "30")),
            rate_limit_requests_per_minute=int(os.getenv("LLM_RATE_LIMIT", "60"))
        )
        
        # Cache configuration
        cache_config = CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            backend=os.getenv("CACHE_BACKEND", "memory"),
            ttl_seconds=int(os.getenv("CACHE_TTL", "3600")),
            max_size=int(os.getenv("CACHE_MAX_SIZE", "1000")),
            redis_url=os.getenv("REDIS_URL"),
            file_path=Path(os.getenv("CACHE_FILE_PATH", "./cache")) if os.getenv("CACHE_FILE_PATH") else None
        )
        
        # Logging configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_file = os.getenv("LOG_FILE")
        logging_config = LoggingConfig(
            level=log_level,
            file_path=Path(log_file) if log_file else None,
            max_file_size_mb=int(os.getenv("LOG_MAX_SIZE_MB", "10")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
        
        # Validation configuration
        validation_config = ValidationConfig(
            max_rows=int(os.getenv("MAX_ROWS", "100000")),
            max_columns=int(os.getenv("MAX_COLUMNS", "1000")),
            max_missing_percentage=float(os.getenv("MAX_MISSING_PCT", "90.0")),
            min_non_null_values=int(os.getenv("MIN_NON_NULL", "2"))
        )
        
        # Imputation configuration
        imputation_config = ImputationConfig(
            statistical_method=os.getenv("STATISTICAL_METHOD", "knn"),
            knn_neighbors=int(os.getenv("KNN_NEIGHBORS", "3")),
            categorical_threshold=int(os.getenv("CATEGORICAL_THRESHOLD", "20")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            max_llm_calls_per_column=int(os.getenv("MAX_LLM_CALLS", "100"))
        )
        
        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            metrics_backend=os.getenv("METRICS_BACKEND", "prometheus"),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
            health_check_enabled=os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true",
            performance_tracking=os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true"
        )
        
        return cls(
            llm=llm_config,
            cache=cache_config,
            logging=logging_config,
            validation=validation_config,
            imputation=imputation_config,
            monitoring=monitoring_config
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        if self.llm.provider != LLMProvider.OLLAMA and not self.llm.api_key:
            raise ConfigurationError(f"API key required for {self.llm.provider.value}")
        
        if self.validation.max_rows <= 0:
            raise ConfigurationError("max_rows must be positive")
        
        if self.validation.max_columns <= 0:
            raise ConfigurationError("max_columns must be positive")
        
        if not 0 <= self.validation.max_missing_percentage <= 100:
            raise ConfigurationError("max_missing_percentage must be between 0 and 100")
        
        if self.cache.enabled and self.cache.backend == "redis" and not self.cache.redis_url:
            raise ConfigurationError("redis_url required when using redis cache backend")


# Global configuration instance
_config: Optional[AutoFillConfig] = None


def get_config() -> AutoFillConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise ConfigurationError("Configuration not initialized. Call init_config() first.")
    return _config


def init_config(llm_provider: str = "groq", config: Optional[AutoFillConfig] = None) -> AutoFillConfig:
    """Initialize the global configuration."""
    global _config
    if config is None:
        config = AutoFillConfig.from_env(llm_provider)
    
    config.validate()
    _config = config
    return config


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None