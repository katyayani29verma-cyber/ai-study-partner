"""Monitoring and logging for performance tracking"""
import logging
import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_time": 0,
            "start_time": datetime.utcnow(),
        }
        self.endpoint_metrics = {}
    
    def record_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """Record request metrics"""
        self.metrics["requests"] += 1
        self.metrics["total_time"] += duration
        
        if status_code >= 400:
            self.metrics["errors"] += 1
        
        key = f"{method} {endpoint}"
        if key not in self.endpoint_metrics:
            self.endpoint_metrics[key] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "errors": 0,
            }
        
        metrics = self.endpoint_metrics[key]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        
        if status_code >= 400:
            metrics["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = datetime.utcnow() - self.metrics["start_time"]
        avg_time = (self.metrics["total_time"] / self.metrics["requests"] 
                   if self.metrics["requests"] > 0 else 0)
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": self.metrics["requests"],
            "total_errors": self.metrics["errors"],
            "error_rate": (self.metrics["errors"] / self.metrics["requests"] 
                          if self.metrics["requests"] > 0 else 0),
            "average_response_time": avg_time,
            "endpoints": self.endpoint_metrics,
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        process = psutil.Process(os.getpid())
        
        return {
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
        }


class RequestTimer:
    """Context manager for timing requests"""
    
    def __init__(self, endpoint: str, method: str = "GET"):
        self.endpoint = endpoint
        self.method = method
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        logger.debug(f"{self.method} {self.endpoint}: {self.duration:.3f}s")


def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__}: {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__}: {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class Logger:
    """Structured logging"""
    
    @staticmethod
    def setup_logging(level=logging.INFO):
        """Setup logging configuration"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler(),
            ]
        )
    
    @staticmethod
    def log_event(event_type: str, user_id: int = None, details: dict = None):
        """Log application event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details or {},
        }
        logger.info(f"Event: {log_entry}")
    
    @staticmethod
    def log_error(error_type: str, error_message: str, user_id: int = None):
        """Log error event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "user_id": user_id,
        }
        logger.error(f"Error: {log_entry}")


# Global monitor instance
monitor = PerformanceMonitor()
