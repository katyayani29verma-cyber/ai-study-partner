"""Caching layer for performance optimization"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """In-memory cache manager for development"""
    
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set cache value with TTL"""
        self.cache[key] = value
        self.expiry[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        logger.debug(f"Cache set: {key} (TTL: {ttl_seconds}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired"""
        if key not in self.cache:
            return None
        
        if datetime.utcnow() > self.expiry.get(key, datetime.utcnow()):
            del self.cache[key]
            del self.expiry[key]
            logger.debug(f"Cache expired: {key}")
            return None
        
        logger.debug(f"Cache hit: {key}")
        return self.cache[key]
    
    def delete(self, key: str) -> None:
        """Delete cache entry"""
        if key in self.cache:
            del self.cache[key]
            del self.expiry[key]
            logger.debug(f"Cache deleted: {key}")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.expiry.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "total_entries": len(self.cache),
            "memory_usage": sum(len(str(v)) for v in self.cache.values()),
        }


class RedisCache:
    """Redis cache manager for production"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis cache"""
        try:
            import redis
            self.redis = redis.from_url(redis_url)
            self.redis.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using in-memory cache.")
            self.redis = None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set cache value in Redis"""
        if not self.redis:
            return
        
        try:
            self.redis.setex(
                key,
                ttl_seconds,
                json.dumps(value, default=str)
            )
            logger.debug(f"Redis cache set: {key}")
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value from Redis"""
        if not self.redis:
            return None
        
        try:
            value = self.redis.get(key)
            if value:
                logger.debug(f"Redis cache hit: {key}")
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> None:
        """Delete cache entry from Redis"""
        if not self.redis:
            return
        
        try:
            self.redis.delete(key)
            logger.debug(f"Redis cache deleted: {key}")
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self) -> None:
        """Clear all cache in Redis"""
        if not self.redis:
            return
        
        try:
            self.redis.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


# Global cache instance
cache = CacheManager()


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    return ":".join(key_parts)
