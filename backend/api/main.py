"""Main FastAPI application - Production-ready"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Import database setup
from database.config import create_db_engine, Base
from database.session import init_db_session
from security.config import settings
from security.headers import SecurityHeadersMiddleware

# Import all routers
from api.routes import auth, cognitive_load, content_chunking, revision, learning_path

logger = logging.getLogger(__name__)

# Create database engine
engine = create_db_engine()

# Initialize session
init_db_session(engine)

# Create tables only in development mode
# In production, use Alembic migrations: alembic upgrade head
if os.getenv("ENVIRONMENT", "development") == "development":
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created (development mode)")
else:
    logger.info("Production mode: Using Alembic migrations for schema management")

# Create FastAPI app
app = FastAPI(
    title="AI Study Partner API",
    description="Adaptive learning platform",
    version="1.0.0"
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware with production settings
cors_origins = settings.ALLOWED_ORIGINS if settings else ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(cognitive_load.router, prefix="/cognitive-load", tags=["cognitive-load"])
app.include_router(content_chunking.router, prefix="/content", tags=["content-chunking"])
app.include_router(revision.router, prefix="/revision", tags=["revision"])
app.include_router(learning_path.router, prefix="/learning-path", tags=["learning-path"])


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Study Partner API",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Study Partner API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
