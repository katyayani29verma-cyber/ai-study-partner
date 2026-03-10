"""Security configuration"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import os


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # JWT Configuration
    SECRET_KEY: str = Field(..., validation_alias="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Encryption
    MASTER_KEY: str = Field(..., validation_alias="MASTER_KEY")
    
    # Database
    DATABASE_URL: str = Field(..., validation_alias="DATABASE_URL")
    
    # CORS
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        validation_alias="ALLOWED_ORIGINS"
    )
    
    # Rate Limiting
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        validation_alias="REDIS_URL"
    )
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: list = [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
    ]
    
    # Security Headers
    HSTS_MAX_AGE: int = 31536000  # 1 year
    
    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v):
        """Ensure SECRET_KEY is secure"""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v
    
    @field_validator('MASTER_KEY')
    @classmethod
    def validate_master_key(cls, v):
        """Ensure MASTER_KEY is secure"""
        if len(v) < 32:
            raise ValueError("MASTER_KEY must be at least 32 characters")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Create settings instance
try:
    settings = SecuritySettings()
    # Convert ALLOWED_ORIGINS string to list
    if isinstance(settings.ALLOWED_ORIGINS, str):
        settings.ALLOWED_ORIGINS = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
except Exception as e:
    # If .env not found, create with defaults
    import logging
    logging.warning(f"Could not load settings from .env: {e}")
    settings = None
