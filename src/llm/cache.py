"""Caching system for LLM responses to reduce costs and improve performance."""

import hashlib
import json
import time
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import pickle
import threading
from abc import ABC, abstractmethod

from ..config.settings import get_config
from ..core.exceptions import ConfigurationError
from ..monitoring.logging import get_logger


logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached LLM response."""
    value: str
    confidence: float
    reasoning: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache entry is expired."""
        return time.time() - self.timestamp > ttl_seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            value=data["value"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a cache entry."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of cached entries."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = []  # For LRU eviction
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            if key in self._cache:
                # Update access order for LRU
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    def set(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            # If key already exists, update access order
            if key in self._cache:
                self._access_order.remove(key)
            
            # If at capacity, remove least recently used
            elif len(self._cache) >= self.max_size:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[key] = entry
            self._access_order.append(key)
    
    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: Path, max_files: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.max_files = max_files
        self._lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("File cache initialized", cache_dir=str(self.cache_dir))
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                return CacheEntry.from_dict(data)
        except Exception as e:
            logger.warning("Failed to read cache file", file_path=str(file_path), error=str(e))
            # Clean up corrupted file
            try:
                file_path.unlink()
            except:
                pass
        
        return None
    
    def set(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            # Check if we need to clean up old files
            if self.size() >= self.max_files:
                self._cleanup_old_files()
            
            file_path = self._get_file_path(key)
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.to_dict(), f)
            except Exception as e:
                logger.error("Failed to write cache file", file_path=str(file_path), error=str(e))
    
    def delete(self, key: str) -> None:
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning("Failed to delete cache file", file_path=str(file_path), error=str(e))
    
    def clear(self) -> None:
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
        except Exception as e:
            logger.error("Failed to clear cache directory", error=str(e))
    
    def size(self) -> int:
        try:
            return len(list(self.cache_dir.glob("*.cache")))
        except Exception:
            return 0
    
    def _cleanup_old_files(self) -> None:
        """Remove oldest cache files to make room."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            if len(cache_files) >= self.max_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda p: p.stat().st_mtime)
                files_to_remove = len(cache_files) - self.max_files + 100  # Remove extra for buffer
                
                for file_path in cache_files[:files_to_remove]:
                    file_path.unlink()
                
                logger.info("Cleaned up old cache files", removed_count=files_to_remove)
        except Exception as e:
            logger.error("Failed to cleanup old cache files", error=str(e))


class LLMCache:
    """Main LLM cache interface with TTL support."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        try:
            self.config = get_config().cache
        except Exception:
            logger.warning("Using fallback cache configuration")
            from ..config.settings import CacheConfig
            self.config = CacheConfig()
        
        if not self.config.enabled:
            logger.info("LLM caching is disabled")
            self.backend = None
            return
        
        # Initialize backend
        if backend:
            self.backend = backend
        elif self.config.backend == "memory":
            self.backend = MemoryCache(max_size=self.config.max_size)
        elif self.config.backend == "file":
            if self.config.file_path:
                self.backend = FileCache(self.config.file_path, max_files=self.config.max_size)
            else:
                raise ConfigurationError("file_path required for file cache backend")
        elif self.config.backend == "redis":
            # Redis backend would be implemented here
            raise ConfigurationError("Redis cache backend not implemented yet")
        else:
            raise ConfigurationError(f"Unknown cache backend: {self.config.backend}")
        
        logger.info(
            "LLM cache initialized",
            backend=self.config.backend,
            ttl_seconds=self.config.ttl_seconds,
            max_size=self.config.max_size
        )
    
    def _generate_key(self, prompt: str, provider: str, model: str, 
                     temperature: float = 0.3) -> str:
        """Generate cache key from prompt and parameters."""
        # Create a stable hash of the prompt and parameters
        cache_input = {
            "prompt": prompt.strip(),
            "provider": provider,
            "model": model,
            "temperature": temperature
        }
        
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, provider: str, model: str, 
           temperature: float = 0.3) -> Optional[Tuple[str, float, str]]:
        """Get cached LLM response."""
        if not self.backend:
            return None
        
        key = self._generate_key(prompt, provider, model, temperature)
        
        try:
            entry = self.backend.get(key)
            if entry is None:
                return None
            
            # Check if expired
            if entry.is_expired(self.config.ttl_seconds):
                self.backend.delete(key)
                logger.debug("Cache entry expired", key=key[:12])
                return None
            
            logger.debug("Cache hit", key=key[:12], provider=provider, model=model)
            return (entry.value, entry.confidence, entry.reasoning)
        
        except Exception as e:
            logger.error("Cache get error", error=str(e), key=key[:12])
            return None
    
    def set(self, prompt: str, provider: str, model: str, 
           value: str, confidence: float, reasoning: str,
           temperature: float = 0.3, metadata: Optional[Dict] = None) -> None:
        """Cache LLM response."""
        if not self.backend:
            return
        
        key = self._generate_key(prompt, provider, model, temperature)
        
        try:
            entry = CacheEntry(
                value=value,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            self.backend.set(key, entry)
            logger.debug("Cache set", key=key[:12], provider=provider, model=model)
        
        except Exception as e:
            logger.error("Cache set error", error=str(e), key=key[:12])
    
    def clear(self) -> None:
        """Clear all cached entries."""
        if self.backend:
            self.backend.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.backend:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "backend": self.config.backend,
            "size": self.backend.size(),
            "max_size": self.config.max_size,
            "ttl_seconds": self.config.ttl_seconds
        }


# Global cache instance
_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """Get the global LLM cache instance."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMCache()
    return _llm_cache


def clear_cache() -> None:
    """Clear the global LLM cache."""
    cache = get_llm_cache()
    cache.clear()


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache = get_llm_cache()
    return cache.get_stats()