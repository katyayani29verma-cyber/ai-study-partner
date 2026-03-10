"""Input validation module - Clean implementation"""
import re
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password(password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'[0-9]', password):
            return False
        return True
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize input to prevent XSS"""
        if not text:
            return text
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    @staticmethod
    def check_sql_injection(text: str) -> bool:
        """Check for SQL injection patterns"""
        sql_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b',
            r'\bINSERT\b',
            r'\bUPDATE\b',
            r'\bUNION\b',
            r'\bSELECT\b',
            r';\s*(DROP|DELETE|INSERT|UPDATE)',
            r"'\s*(OR|AND)\s*'",
        ]
        text_upper = text.upper()
        
        for pattern in sql_patterns:
            if re.search(pattern, text_upper):
                return True
        
        return False
