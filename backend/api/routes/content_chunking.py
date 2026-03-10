"""Content chunking routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging
from typing import List

from database.session import get_session_factory
from database.operations import ContentChunkingOperations

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class ChunkData(BaseModel):
    title: str
    content: str
    estimated_cognitive_load: float = 50
    estimated_duration: int = 15
    difficulty_level: str = "medium"
    learning_objectives: List[str] = None
    key_concepts: List[str] = None

class ChunkInteractionRequest(BaseModel):
    user_id: int
    chunk_id: int
    time_spent: int
    completion_percentage: float
    comprehension_score: float = None
    cognitive_load: float = None

# Dependencies
def get_db():
    """Get database session"""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes
@router.post("/create/{material_id}")
async def create_chunks(
    material_id: int,
    chunks: List[ChunkData],
    db: Session = Depends(get_db)
):
    """Create chunks for material"""
    try:
        chunks_data = [chunk.dict() for chunk in chunks]
        created_chunks = ContentChunkingOperations.create_chunks_from_material(
            session=db,
            material_id=material_id,
            chunks_data=chunks_data
        )
        
        db.commit()
        
        return {
            "material_id": material_id,
            "chunks_created": len(created_chunks),
            "chunks": [
                {
                    "id": c.id,
                    "chunk_number": c.chunk_number,
                    "title": c.title,
                    "difficulty": c.difficulty_level
                }
                for c in created_chunks
            ]
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chunks"
        )

@router.post("/interaction")
async def record_chunk_interaction(
    request: ChunkInteractionRequest,
    db: Session = Depends(get_db)
):
    """Record user interaction with chunk"""
    try:
        interaction = ContentChunkingOperations.record_chunk_interaction(
            session=db,
            user_id=request.user_id,
            chunk_id=request.chunk_id,
            time_spent=request.time_spent,
            completion_percentage=request.completion_percentage,
            comprehension_score=request.comprehension_score,
            cognitive_load=request.cognitive_load
        )
        
        db.commit()
        
        return {
            "id": interaction.id,
            "user_id": interaction.user_id,
            "chunk_id": interaction.chunk_id,
            "comprehension_score": interaction.comprehension_score,
            "timestamp": interaction.timestamp.isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record interaction"
        )

@router.get("/analytics/{chunk_id}")
async def get_chunk_analytics(
    chunk_id: int,
    db: Session = Depends(get_db)
):
    """Get analytics for chunk"""
    try:
        analytics = ContentChunkingOperations.get_chunk_analytics(db, chunk_id)
        
        return {
            "chunk_id": chunk_id,
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics"
        )
