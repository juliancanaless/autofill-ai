"""Metrics and monitoring for the autofill pipeline."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

from ..core.types import ImputationResult
from ..config.settings import get_config
from ..monitoring.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    operation: str
    duration_seconds: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APICallMetric:
    """API call metric data point."""
    provider: str
    model: str
    timestamp: datetime
    duration_seconds: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ImputationMetric:
    """Imputation operation metric."""
    column: str
    method: str
    count: int
    average_confidence: float
    timestamp: datetime
    duration_seconds: float


class MetricsCollector:
    """Collects and aggregates metrics for the autofill pipeline."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Metric storage
        self.performance_metrics: deque = deque(maxlen=max_history)
        self.api_call_metrics: deque = deque(maxlen=max_history)
        self.imputation_metrics: deque = deque(maxlen=max_history)
        
        # Counters
        self.counters = defaultdict(int)
        
        # Active operations (for tracking ongoing operations)
        self._active_operations = {}
        
        try:
            self.config = get_config().monitoring
        except Exception:
            logger.warning("Using fallback monitoring configuration")
            from ..config.settings import MonitoringConfig
            self.config = MonitoringConfig()
        
        logger.info("Metrics collector initialized", max_history=max_history)
    
    def start_operation(self, operation_id: str, operation_name: str, 
                       metadata: Optional[Dict] = None) -> None:
        """Start tracking an operation."""
        with self._lock:
            self._active_operations[operation_id] = {
                "name": operation_name,
                "start_time": time.time(),
                "metadata": metadata or {}
            }
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error: Optional[str] = None) -> Optional[float]:
        """End tracking an operation and record the metric."""
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning("Attempted to end unknown operation", operation_id=operation_id)
                return None
            
            operation = self._active_operations.pop(operation_id)
            duration = time.time() - operation["start_time"]
            
            # Record performance metric
            metric = PerformanceMetric(
                operation=operation["name"],
                duration_seconds=duration,
                timestamp=datetime.now(),
                metadata={
                    **operation["metadata"],
                    "success": success,
                    "error": error
                }
            )
            
            self.performance_metrics.append(metric)
            self.counters[f"operations.{operation['name']}.total"] += 1
            
            if success:
                self.counters[f"operations.{operation['name']}.success"] += 1
            else:
                self.counters[f"operations.{operation['name']}.error"] += 1
            
            logger.debug(
                "Operation completed",
                operation=operation["name"],
                duration=duration,
                success=success
            )
            
            return duration
    
    def record_api_call(self, provider: str, model: str, duration_seconds: float,
                       tokens_used: Optional[int] = None, cost: Optional[float] = None,
                       success: bool = True, error: Optional[str] = None) -> None:
        """Record an API call metric."""
        with self._lock:
            metric = APICallMetric(
                provider=provider,
                model=model,
                timestamp=datetime.now(),
                duration_seconds=duration_seconds,
                tokens_used=tokens_used,
                cost=cost,
                success=success,
                error=error
            )
            
            self.api_call_metrics.append(metric)
            
            # Update counters
            self.counters[f"api.{provider}.calls.total"] += 1
            self.counters[f"api.{provider}.calls.{model}"] += 1
            
            if success:
                self.counters[f"api.{provider}.calls.success"] += 1
                if tokens_used:
                    self.counters[f"api.{provider}.tokens.total"] += tokens_used
                if cost:
                    self.counters[f"api.{provider}.cost.total"] += cost
            else:
                self.counters[f"api.{provider}.calls.error"] += 1
    
    def record_imputation(self, column: str, method: str, count: int,
                         average_confidence: float, duration_seconds: float) -> None:
        """Record an imputation operation metric."""
        with self._lock:
            metric = ImputationMetric(
                column=column,
                method=method,
                count=count,
                average_confidence=average_confidence,
                timestamp=datetime.now(),
                duration_seconds=duration_seconds
            )
            
            self.imputation_metrics.append(metric)
            
            # Update counters
            self.counters[f"imputation.{method}.total"] += 1
            self.counters[f"imputation.values_filled.total"] += count
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"period_hours": hours, "total_operations": 0}
        
        # Aggregate by operation type
        by_operation = defaultdict(list)
        for metric in recent_metrics:
            by_operation[metric.operation].append(metric.duration_seconds)
        
        summary = {
            "period_hours": hours,
            "total_operations": len(recent_metrics),
            "operations": {}
        }
        
        for operation, durations in by_operation.items():
            summary["operations"][operation] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            }
        
        return summary
    
    def get_api_usage_summary(self, hours: int = 24) -> Dict:
        """Get API usage summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                m for m in self.api_call_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"period_hours": hours, "total_calls": 0}
        
        # Aggregate by provider
        by_provider = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "errors": 0})
        
        for metric in recent_metrics:
            provider_stats = by_provider[metric.provider]
            provider_stats["calls"] += 1
            
            if metric.success:
                if metric.tokens_used:
                    provider_stats["tokens"] += metric.tokens_used
                if metric.cost:
                    provider_stats["cost"] += metric.cost
            else:
                provider_stats["errors"] += 1
        
        summary = {
            "period_hours": hours,
            "total_calls": len(recent_metrics),
            "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics) * 100,
            "providers": dict(by_provider)
        }
        
        return summary
    
    def get_imputation_summary(self, hours: int = 24) -> Dict:
        """Get imputation summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                m for m in self.imputation_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"period_hours": hours, "total_operations": 0}
        
        # Aggregate by method
        by_method = defaultdict(lambda: {"operations": 0, "values_filled": 0, "avg_confidence": []})
        
        for metric in recent_metrics:
            method_stats = by_method[metric.method]
            method_stats["operations"] += 1
            method_stats["values_filled"] += metric.count
            method_stats["avg_confidence"].append(metric.average_confidence)
        
        # Calculate average confidence for each method
        for method_stats in by_method.values():
            confidences = method_stats["avg_confidence"]
            method_stats["avg_confidence"] = sum(confidences) / len(confidences)
        
        summary = {
            "period_hours": hours,
            "total_operations": len(recent_metrics),
            "total_values_filled": sum(m.count for m in recent_metrics),
            "methods": dict(by_method)
        }
        
        return summary
    
    def get_current_counters(self) -> Dict:
        """Get current counter values."""
        with self._lock:
            return dict(self.counters)
    
    def reset_counters(self) -> None:
        """Reset all counters."""
        with self._lock:
            self.counters.clear()
        logger.info("Metrics counters reset")
    
    def export_metrics(self) -> Dict:
        """Export all metrics for external monitoring systems."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": [
                    {
                        "operation": m.operation,
                        "duration_seconds": m.duration_seconds,
                        "timestamp": m.timestamp.isoformat(),
                        "metadata": m.metadata
                    }
                    for m in list(self.performance_metrics)
                ],
                "api_call_metrics": [
                    {
                        "provider": m.provider,
                        "model": m.model,
                        "timestamp": m.timestamp.isoformat(),
                        "duration_seconds": m.duration_seconds,
                        "tokens_used": m.tokens_used,
                        "cost": m.cost,
                        "success": m.success,
                        "error": m.error
                    }
                    for m in list(self.api_call_metrics)
                ],
                "imputation_metrics": [
                    {
                        "column": m.column,
                        "method": m.method,
                        "count": m.count,
                        "average_confidence": m.average_confidence,
                        "timestamp": m.timestamp.isoformat(),
                        "duration_seconds": m.duration_seconds
                    }
                    for m in list(self.imputation_metrics)
                ],
                "counters": dict(self.counters)
            }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, operation_name: str, metadata: Optional[Dict] = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.operation_id = f"{operation_name}_{time.time()}"
        self.collector = get_metrics_collector()
        self.duration = None
    
    def __enter__(self):
        self.collector.start_operation(self.operation_id, self.operation_name, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.duration = self.collector.end_operation(self.operation_id, success, error)


# Convenience functions
def track_performance(operation_name: str, metadata: Optional[Dict] = None) -> PerformanceTracker:
    """Create a performance tracker context manager."""
    return PerformanceTracker(operation_name, metadata)


def record_api_call(provider: str, model: str, duration_seconds: float,
                   tokens_used: Optional[int] = None, cost: Optional[float] = None,
                   success: bool = True, error: Optional[str] = None) -> None:
    """Record an API call metric."""
    collector = get_metrics_collector()
    collector.record_api_call(provider, model, duration_seconds, tokens_used, cost, success, error)


def record_imputation(column: str, method: str, count: int,
                     average_confidence: float, duration_seconds: float) -> None:
    """Record an imputation operation metric."""
    collector = get_metrics_collector()
    collector.record_imputation(column, method, count, average_confidence, duration_seconds)