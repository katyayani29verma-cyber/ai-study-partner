"""Database configuration - Production-ready"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
import logging
from security.config import settings

logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()


def get_database_url():
    """Get database URL from SecuritySettings or environment"""
    if settings:
        return settings.DATABASE_URL
    # Fallback for development
    import os
    return os.getenv("DATABASE_URL", "sqlite:///./study_partner.db")


def create_db_engine():
    """Create SQLAlchemy engine with production settings"""
    database_url = get_database_url()
    
    # Use SQLite for development
    if "sqlite" in database_url:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        logger.info("Using SQLite database (development mode)")
    else:
        # PostgreSQL or other production databases
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "application_name": "ai_study_partner"
            }
        )
        logger.info("Using production database (PostgreSQL or equivalent)")
    
    logger.info(f"Database engine created: {database_url.split('@')[0] if '@' in database_url else database_url}")
    return engine
