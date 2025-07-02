"""Base classes and interfaces for the autofill pipeline."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from .types import ColumnAnalysis, ImputationStrategy, ImputationResult, LLMProvider
from .exceptions import ImputationError, LLMProviderError


class DataAnalyzer(ABC):
    """Abstract base class for data analyzers."""
    
    @abstractmethod
    def analyze_column(self, series: pd.Series, column_name: str) -> ColumnAnalysis:
        """Analyze a single column and return its characteristics."""
        pass
    
    @abstractmethod
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze the entire dataset and return comprehensive analysis."""
        pass
    
    @abstractmethod
    def determine_strategy(self, column_analysis: ColumnAnalysis, series: pd.Series) -> ImputationStrategy:
        """Determine the best imputation strategy for a column."""
        pass


class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def query(self, prompt: str, **kwargs) -> Tuple[str, float, str]:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        
        Returns:
            Tuple of (value, confidence, reasoning)
        
        Raises:
            LLMProviderError: If the query fails
        """
        pass
    
    @abstractmethod
    def get_usage_info(self) -> str:
        """Get information about API usage and costs."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass
    
    def validate_response(self, response: str) -> bool:
        """Validate the LLM response format. Override if needed."""
        return True
    
    def estimate_cost(self, prompt: str, response: str) -> Optional[float]:
        """Estimate the cost of the API call. Override if available."""
        return None


class ImputationStrategyInterface(ABC):
    """Abstract base class for imputation strategies."""
    
    @abstractmethod
    def can_handle(self, column_analysis: ColumnAnalysis) -> bool:
        """Check if this strategy can handle the given column."""
        pass
    
    @abstractmethod
    def impute(self, df: pd.DataFrame, column: str, 
              column_analysis: ColumnAnalysis) -> List[ImputationResult]:
        """
        Perform imputation on the specified column.
        
        Args:
            df: The DataFrame to impute
            column: The column name to impute
            column_analysis: Analysis results for the column
        
        Returns:
            List of imputation results
        
        Raises:
            ImputationError: If imputation fails
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass
    
    def estimate_time(self, missing_count: int, column_analysis: ColumnAnalysis) -> float:
        """Estimate time to complete imputation. Override if needed."""
        return missing_count * 0.1  # Default: 0.1 seconds per missing value
    
    def get_confidence_base(self) -> float:
        """Get the base confidence level for this strategy."""
        return 0.5  # Default confidence


class CacheInterface(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of items in the cache."""
        pass


class MetricsCollectorInterface(ABC):
    """Abstract base class for metrics collectors."""
    
    @abstractmethod
    def record_operation(self, operation: str, duration: float, 
                        metadata: Optional[Dict] = None) -> None:
        """Record an operation metric."""
        pass
    
    @abstractmethod
    def record_api_call(self, provider: str, model: str, duration: float,
                       success: bool = True, **kwargs) -> None:
        """Record an API call metric."""
        pass
    
    @abstractmethod
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get a summary of metrics for the specified time period."""
        pass


class ValidatorInterface(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the data and return True if valid."""
        pass
    
    def get_validation_errors(self, data: Any) -> List[str]:
        """Get detailed validation errors. Override if needed."""
        return []


class PipelineComponent(ABC):
    """Base class for pipeline components."""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    def initialize(self, config: Optional[Dict] = None) -> None:
        """Initialize the component with configuration."""
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process the input data and return the result."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources. Override if needed."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if the component is initialized."""
        return self._initialized
    
    def get_name(self) -> str:
        """Get the component name."""
        return self.name


class PipelineOrchestrator(ABC):
    """Abstract base class for pipeline orchestrators."""
    
    def __init__(self):
        self.components: Dict[str, PipelineComponent] = {}
        self.pipeline_order: List[str] = []
    
    def add_component(self, component: PipelineComponent) -> None:
        """Add a component to the pipeline."""
        self.components[component.get_name()] = component
    
    def remove_component(self, name: str) -> None:
        """Remove a component from the pipeline."""
        if name in self.components:
            self.components[name].cleanup()
            del self.components[name]
        
        if name in self.pipeline_order:
            self.pipeline_order.remove(name)
    
    def set_pipeline_order(self, order: List[str]) -> None:
        """Set the order of pipeline execution."""
        # Validate that all components exist
        for name in order:
            if name not in self.components:
                raise ValueError(f"Component '{name}' not found in pipeline")
        
        self.pipeline_order = order
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute the pipeline."""
        pass
    
    def initialize_all(self, config: Optional[Dict] = None) -> None:
        """Initialize all components."""
        for component in self.components.values():
            if not component.is_initialized():
                component_config = config.get(component.get_name(), {}) if config else {}
                component.initialize(component_config)
    
    def cleanup_all(self) -> None:
        """Clean up all components."""
        for component in self.components.values():
            component.cleanup()


class ConfigurableComponent(PipelineComponent):
    """Base class for components that can be configured."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.config: Dict = {}
    
    def set_config(self, config: Dict) -> None:
        """Set the configuration for this component."""
        self.config = config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration with new values."""
        self.config.update(updates)


class RetryableComponent(ConfigurableComponent):
    """Base class for components that support retry logic."""
    
    def __init__(self, name: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def execute_with_retry(self, operation: callable, *args, **kwargs) -> Any:
        """Execute an operation with retry logic."""
        import time
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise last_exception


class AsyncComponent(ConfigurableComponent):
    """Base class for components that support async operations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._executor = None
    
    def initialize(self, config: Optional[Dict] = None) -> None:
        """Initialize the async component."""
        from concurrent.futures import ThreadPoolExecutor
        
        if config:
            self.set_config(config)
        
        max_workers = self.get_config('max_workers', 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up the executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def submit_async(self, func: callable, *args, **kwargs):
        """Submit a function for async execution."""
        if not self._executor:
            raise RuntimeError("Component not initialized")
        
        return self._executor.submit(func, *args, **kwargs)