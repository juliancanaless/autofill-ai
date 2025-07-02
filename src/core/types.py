"""Type definitions for the autofill pipeline."""

from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class ImputationMethod(Enum):
    """Available imputation methods."""
    STATISTICAL = "statistical"
    LLM_CATEGORICAL = "llm_categorical"
    LLM_CONTEXTUAL = "llm_contextual"


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Priority(Enum):
    """Task priority levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column."""
    data_type: str
    unique_values: int
    sample_values: List[Any]
    is_numeric: bool
    is_categorical: bool
    has_patterns: bool
    domain_hints: List[str]
    categories: Optional[List[str]] = None


@dataclass
class ImputationStrategy:
    """Strategy for imputing missing values in a column."""
    method: ImputationMethod
    approach: str
    confidence_base: float
    explanation: str
    categories: Optional[List[str]] = None
    domain_hints: Optional[List[str]] = None


@dataclass
class ImputationResult:
    """Result of a single imputation operation."""
    index: int
    column: str
    original_value: Any
    suggested_value: Any
    confidence: float
    method: str
    reasoning: str
    timestamp: str


@dataclass
class MissingDataSummary:
    """Summary of missing data for a column."""
    missing_count: int
    missing_percentage: float
    priority: Priority
    estimated_time_minutes: float


@dataclass
class DatasetProfile:
    """Complete profile of a dataset."""
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    analysis_timestamp: str


@dataclass
class QualityAssessment:
    """Quality assessment of imputation results."""
    overall_quality_score: float
    high_confidence_percentage: float
    low_confidence_count: int
    error_count: int
    status: str


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers."""
    
    def query(self, prompt: str) -> Tuple[str, float, str]:
        """Query the LLM with a prompt and return value, confidence, reasoning."""
        ...
    
    def get_usage_info(self) -> str:
        """Get information about API usage and costs."""
        ...


class ImputationStrategyProtocol(Protocol):
    """Protocol for imputation strategies."""
    
    def can_handle(self, column_analysis: ColumnAnalysis) -> bool:
        """Check if this strategy can handle the given column."""
        ...
    
    def impute(self, df: pd.DataFrame, column: str) -> List[ImputationResult]:
        """Perform imputation on the specified column."""
        ...