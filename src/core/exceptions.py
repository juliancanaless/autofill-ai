"""Custom exceptions for the autofill pipeline."""


class AutoFillError(Exception):
    """Base exception for autofill pipeline errors."""
    pass


class ConfigurationError(AutoFillError):
    """Raised when there's a configuration issue."""
    pass


class APIKeyError(ConfigurationError):
    """Raised when API key is missing or invalid."""
    pass


class LLMProviderError(AutoFillError):
    """Raised when there's an issue with the LLM provider."""
    pass


class ImputationError(AutoFillError):
    """Raised when imputation fails."""
    pass


class ValidationError(AutoFillError):
    """Raised when input/output validation fails."""
    pass


class DataQualityError(AutoFillError):
    """Raised when data quality issues are detected."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when API rate limits are exceeded."""
    pass


class ModelNotAvailableError(LLMProviderError):
    """Raised when the specified model is not available."""
    pass