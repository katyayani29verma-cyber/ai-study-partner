"""CSRF token generation and validation"""
import secrets
import hmac
import hashlib
from typing import Tuple


class CSRFManager:
    """Manage CSRF token generation and validation"""
    
    def __init__(self, secret_key: str):
        """Initialize CSRF manager"""
        self.secret_key = secret_key
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(32)
        signature = hmac.new(
            self.secret_key.encode(),
            f"{session_id}:{token}".encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{token}.{signature}"
    
    def validate_token(self, token: str, session_id: str) -> bool:
        """Validate CSRF token"""
        try:
            token_part, signature = token.split('.')
            expected_signature = hmac.new(
                self.secret_key.encode(),
                f"{session_id}:{token_part}".encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
