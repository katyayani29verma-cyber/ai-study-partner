"""Performance, scaling, and maintenance module"""
from .caching import CacheManager, RedisCache, cache, cache_key
from .background_tasks import BackgroundTask, TaskQueue, task_queue, TaskStatus
from .monitoring import PerformanceMonitor, RequestTimer, monitor_performance, Logger, monitor
from .scaling import (
    LoadBalancer,
    LoadBalancingStrategy,
    Server,
    AutoScaler,
    ConnectionPool,
)

__all__ = [
    # Caching
    "CacheManager",
    "RedisCache",
    "cache",
    "cache_key",
    # Background Tasks
    "BackgroundTask",
    "TaskQueue",
    "task_queue",
    "TaskStatus",
    # Monitoring
    "PerformanceMonitor",
    "RequestTimer",
    "monitor_performance",
    "Logger",
    "monitor",
    # Scaling
    "LoadBalancer",
    "LoadBalancingStrategy",
    "Server",
    "AutoScaler",
    "ConnectionPool",
]
