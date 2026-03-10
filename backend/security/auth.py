"""Authentication module - Production-ready implementation"""
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import logging
from security.config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class AuthManager:
    """Authentication manager - uses SecuritySettings for secrets"""
    
    def __init__(self, secret_key: str = None):
        """Initialize with secret key from settings or parameter"""
        if secret_key is None:
            if settings is None:
                raise RuntimeError("SecuritySettings not initialized. Ensure .env is configured.")
            secret_key = settings.SECRET_KEY
        self.secret_key = secret_key
        self.algorithm = settings.ALGORITHM if settings else "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES if settings else 30
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            raise
    
    def decode_token(self, token: str) -> dict:
        """Decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"Error decoding token: {e}")
            return None
