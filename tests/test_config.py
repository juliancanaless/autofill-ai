"""Tests for configuration management."""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path if not already there
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from config.settings import (
        AutoFillConfig, LLMConfig, CacheConfig, LoggingConfig,
        ValidationConfig, ImputationConfig, MonitoringConfig,
        init_config, get_config, reset_config
    )
    from core.types import LLMProvider
    from core.exceptions import ConfigurationError
except ImportError:
    # Fallback import method
    from src.config.settings import (
        AutoFillConfig, LLMConfig, CacheConfig, LoggingConfig,
        ValidationConfig, ImputationConfig, MonitoringConfig,
        init_config, get_config, reset_config
    )
    from src.core.types import LLMProvider
    from src.core.exceptions import ConfigurationError


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_llm_config_creation(self):
        config = LLMConfig(
            provider=LLMProvider.GROQ,
            api_key="test-key",
            model_name="test-model"
        )
        assert config.provider == LLMProvider.GROQ
        assert config.api_key == "test-key"
        assert config.model_name == "test-model"
        assert config.max_retries == 3
    
    def test_llm_config_defaults(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert config.temperature == 0.3


class TestAutoFillConfig:
    """Test main configuration class."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
    
    def teardown_method(self):
        """Reset config after each test."""
        reset_config()
    
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"})
    def test_from_env_groq(self):
        config = AutoFillConfig.from_env("groq")
        assert config.llm.provider == LLMProvider.GROQ
        assert config.llm.api_key == "test-groq-key"
        assert config.llm.model_name == "llama3-8b-8192"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_from_env_openai(self):
        config = AutoFillConfig.from_env("openai")
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.llm.api_key == "test-openai-key"
        assert config.llm.model_name == "gpt-3.5-turbo"
    
    def test_from_env_ollama(self):
        config = AutoFillConfig.from_env("ollama")
        assert config.llm.provider == LLMProvider.OLLAMA
        assert config.llm.api_key is None
        assert config.llm.model_name == "llama3.1:8b"
    
    def test_from_env_invalid_provider(self):
        with pytest.raises(ConfigurationError, match="Invalid LLM provider"):
            AutoFillConfig.from_env("invalid")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="Missing GROQ_API_KEY"):
            AutoFillConfig.from_env("groq")
    
    @patch.dict(os.environ, {
        "GROQ_API_KEY": "test-key",
        "CACHE_ENABLED": "false",
        "LOG_LEVEL": "DEBUG",
        "MAX_ROWS": "50000"
    })
    def test_from_env_custom_settings(self):
        config = AutoFillConfig.from_env("groq")
        assert not config.cache.enabled
        assert config.logging.level == "DEBUG"
        assert config.validation.max_rows == 50000
    
    def test_validate_valid_config(self):
        config = AutoFillConfig(
            llm=LLMConfig(provider=LLMProvider.OLLAMA)
        )
        config.validate()  # Should not raise
    
    def test_validate_missing_api_key(self):
        config = AutoFillConfig(
            llm=LLMConfig(provider=LLMProvider.GROQ, api_key=None)
        )
        with pytest.raises(ConfigurationError, match="API key required"):
            config.validate()
    
    def test_validate_invalid_max_rows(self):
        config = AutoFillConfig(
            llm=LLMConfig(provider=LLMProvider.OLLAMA),
            validation=ValidationConfig(max_rows=0)
        )
        with pytest.raises(ConfigurationError, match="max_rows must be positive"):
            config.validate()
    
    def test_validate_invalid_missing_percentage(self):
        config = AutoFillConfig(
            llm=LLMConfig(provider=LLMProvider.OLLAMA),
            validation=ValidationConfig(max_missing_percentage=150.0)
        )
        with pytest.raises(ConfigurationError, match="must be between 0 and 100"):
            config.validate()


class TestConfigGlobalState:
    """Test global configuration state management."""
    
    def setup_method(self):
        reset_config()
    
    def teardown_method(self):
        reset_config()
    
    def test_init_config(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            config = init_config("groq")
            assert isinstance(config, AutoFillConfig)
            assert config.llm.provider == LLMProvider.GROQ
    
    def test_get_config_before_init(self):
        with pytest.raises(ConfigurationError, match="Configuration not initialized"):
            get_config()
    
    def test_get_config_after_init(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            init_config("groq")
            config = get_config()
            assert isinstance(config, AutoFillConfig)
    
    def test_init_config_with_custom_config(self):
        custom_config = AutoFillConfig(
            llm=LLMConfig(provider=LLMProvider.OLLAMA)
        )
        result = init_config(config=custom_config)
        assert result is custom_config
        assert get_config() is custom_config


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_cache_config_defaults(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000
    
    def test_cache_config_custom(self):
        config = CacheConfig(
            enabled=False,
            backend="redis",
            redis_url="redis://localhost:6379"
        )
        assert not config.enabled
        assert config.backend == "redis"
        assert config.redis_url == "redis://localhost:6379"


class TestValidationConfig:
    """Test validation configuration."""
    
    def test_validation_config_defaults(self):
        config = ValidationConfig()
        assert config.max_rows == 100000
        assert config.max_columns == 1000
        assert config.max_missing_percentage == 90.0
        assert config.min_non_null_values == 2
        assert '.csv' in config.allowed_file_types
    
    def test_validation_config_custom(self):
        config = ValidationConfig(
            max_rows=50000,
            allowed_file_types=['.csv', '.json']
        )
        assert config.max_rows == 50000
        assert config.allowed_file_types == ['.csv', '.json']


class TestImputationConfig:
    """Test imputation configuration."""
    
    def test_imputation_config_defaults(self):
        config = ImputationConfig()
        assert config.statistical_method == "knn"
        assert config.knn_neighbors == 3
        assert config.categorical_threshold == 20
        assert config.confidence_threshold == 0.5
    
    def test_imputation_config_custom(self):
        config = ImputationConfig(
            statistical_method="median",
            confidence_threshold=0.7
        )
        assert config.statistical_method == "median"
        assert config.confidence_threshold == 0.7


class TestMonitoringConfig:
    """Test monitoring configuration."""
    
    def test_monitoring_config_defaults(self):
        config = MonitoringConfig()
        assert config.enabled is True
        assert config.metrics_backend == "prometheus"
        assert config.metrics_port == 8000
        assert config.health_check_enabled is True
    
    def test_monitoring_config_custom(self):
        config = MonitoringConfig(
            enabled=False,
            metrics_backend="statsd"
        )
        assert not config.enabled
        assert config.metrics_backend == "statsd"