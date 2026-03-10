"""Rate limiting to prevent abuse"""
from datetime import datetime, timedelta
from typing import Dict, Optional
import asyncio


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.requests: Dict[str, list] = {}
    
    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]
        
        # Check if limit exceeded
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_remaining(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> int:
        """Get remaining requests in window"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)
        
        if key not in self.requests:
            return max_requests
        
        # Count requests in window
        requests_in_window = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]
        
        return max(0, max_requests - len(requests_in_window))


class RateLimitConfig:
    """Rate limit configurations"""
    
    # Authentication limits
    LOGIN_ATTEMPTS = (5, 60)  # 5 attempts per minute
    PASSWORD_RESET = (3, 3600)  # 3 attempts per hour
    
    # API limits
    API_GENERAL = (100, 3600)  # 100 requests per hour
    API_UPLOAD = (10, 3600)  # 10 uploads per hour
    API_EXPORT = (5, 3600)  # 5 exports per hour
    
    # Document limits
    DOCUMENT_CREATE = (20, 3600)  # 20 documents per hour
    DOCUMENT_DELETE = (10, 3600)  # 10 deletes per hour
    
    @staticmethod
    def get_limit_key(endpoint: str, identifier: str) -> str:
        """Generate rate limit key"""
        return f"{endpoint}:{identifier}"
