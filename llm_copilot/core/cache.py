"""
Caching Layer for Performance Optimization

Caches expensive operations: curve fitting, optimization, embeddings.
"""

from typing import Any, Optional, Dict, Callable
import json
import hashlib
import pickle
import logging
from pathlib import Path
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CacheBackend:
    """Base class for cache backends"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        raise NotImplementedError
    
    def clear(self) -> None:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache.
    
    Fast but not persistent. Lost on restart.
    """
    
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        logger.info("Initialized InMemoryCache")
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check if expired
        if expiry and time.time() > expiry:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()
        logger.info("Cleared in-memory cache")
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class FileCache(CacheBackend):
    """
    File-based cache for persistence.
    
    Slower than memory but survives restarts.
    """
    
    def __init__(self, cache_dir: Path = Path(".cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FileCache: {self.cache_dir}")
    
    def _get_filepath(self, key: str) -> Path:
        """Convert key to filepath"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        filepath = self._get_filepath(key)
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check expiry
            if data['expiry'] and time.time() > data['expiry']:
                filepath.unlink()
                return None
            
            return data['value']
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        filepath = self._get_filepath(key)
        
        data = {
            'value': value,
            'expiry': time.time() + ttl if ttl else None
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def delete(self, key: str) -> None:
        filepath = self._get_filepath(key)
        if filepath.exists():
            filepath.unlink()
    
    def clear(self) -> None:
        for filepath in self.cache_dir.glob("*.pkl"):
            filepath.unlink()
        logger.info(f"Cleared file cache: {self.cache_dir}")
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisCache(CacheBackend):
    """
    Redis-based cache for production.
    
    Fast, persistent, distributed. Requires Redis server.
    """
    
    def __init__(self, redis_url: str):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis not installed. Install with: pip install redis"
            )
        
        self.client = redis.from_url(redis_url)
        logger.info(f"Initialized RedisCache: {redis_url}")
    
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            data = pickle.dumps(value)
            if ttl:
                self.client.setex(key, ttl, data)
            else:
                self.client.set(key, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self) -> None:
        try:
            self.client.flushdb()
            logger.info("Cleared Redis cache")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False


class CacheManager:
    """
    Unified cache manager with automatic backend selection.
    
    Features:
    - Multiple backends (memory, file, Redis)
    - Automatic cache key generation
    - TTL support
    - Statistics tracking
    
    Parameters
    ----------
    backend : str, default="memory"
        Cache backend: "memory", "file", or "redis"
    redis_url : str, optional
        Redis connection URL (required if backend="redis")
    cache_dir : Path, optional
        Cache directory for file backend
    default_ttl : int, optional
        Default TTL in seconds
        
    Examples
    --------
    >>> cache = CacheManager(backend="memory", default_ttl=3600)
    >>> 
    >>> # Manual caching
    >>> cache.set("curves_tv", curve_params)
    >>> params = cache.get("curves_tv")
    >>> 
    >>> # Decorator caching
    >>> @cache.cached(ttl=3600)
    >>> def expensive_calculation(x):
    >>>     return x ** 2
    """
    
    def __init__(
        self,
        backend: str = "memory",
        *,
        redis_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        default_ttl: Optional[int] = None
    ):
        self.backend_name = backend
        self.default_ttl = default_ttl
        
        # Initialize backend
        if backend == "memory":
            self.backend = InMemoryCache()
        elif backend == "file":
            self.backend = FileCache(cache_dir or Path(".cache"))
        elif backend == "redis":
            if not redis_url:
                raise ValueError("redis_url required for Redis backend")
            self.backend = RedisCache(redis_url)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
        
        logger.info(f"Initialized CacheManager: backend={backend}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.backend.get(key)
        
        if value is not None:
            self.stats['hits'] += 1
            logger.debug(f"Cache HIT: {key}")
        else:
            self.stats['misses'] += 1
            logger.debug(f"Cache MISS: {key}")
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        self.backend.set(key, value, ttl)
        self.stats['sets'] += 1
        logger.debug(f"Cache SET: {key}")
    
    def delete(self, key: str) -> None:
        """Delete key from cache"""
        self.backend.delete(key)
        logger.debug(f"Cache DELETE: {key}")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.backend.clear()
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0}
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.backend.exists(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'backend': self.backend_name
        }
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator for caching function results.
        
        Parameters
        ----------
        ttl : int, optional
            Cache TTL in seconds
        key_prefix : str, default=""
            Prefix for cache keys
            
        Examples
        --------
        >>> @cache.cached(ttl=3600, key_prefix="curves")
        >>> def fit_curve(channel, data):
        >>>     # Expensive curve fitting
        >>>     return result
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [key_prefix, func.__name__]
                
                # Hash arguments
                arg_str = json.dumps({
                    'args': [str(a) for a in args],
                    'kwargs': {k: str(v) for k, v in kwargs.items()}
                }, sort_keys=True)
                arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
                key_parts.append(arg_hash)
                
                cache_key = ":".join(filter(None, key_parts))
                
                # Try cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.info(f"Returning cached result for {func.__name__}")
                    return cached_value
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache instance (configured from environment)
def get_cache() -> CacheManager:
    """
    Get global cache instance configured from environment.
    
    Returns
    -------
    CacheManager
        Configured cache manager
    """
    from llm_copilot.config import config
    
    if config.cache_enabled:
        return CacheManager(
            backend="redis",
            redis_url=config.redis_url,
            default_ttl=config.cache_ttl_seconds
        )
    else:
        # Fallback to file cache
        return CacheManager(
            backend="file",
            default_ttl=3600
        )

