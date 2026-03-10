"""Database session management - Clean implementation"""
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

# Global session factory
SessionLocal = None


def init_db_session(engine):
    """Initialize database session with engine"""
    global SessionLocal
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
    return SessionLocal


def get_session_factory():
    """Get the session factory (lazy initialization)"""
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db_session first.")
    return SessionLocal


def get_db():
    """Get database session for FastAPI dependency"""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
