"""Cognitive load management routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from database.session import SessionLocal
from database.operations import CognitiveLoadOperations

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class CognitiveLoadRequest(BaseModel):
    user_id: int
    mental_effort: float
    working_memory_load: float
    attention_level: float
    stress_level: float
    session_id: int = None

class CognitiveLoadResponse(BaseModel):
    id: int
    user_id: int
    overall_cognitive_load: float
    is_overloaded: bool
    recommended_break: bool
    recommended_pace: str

# Dependencies
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes
@router.post("/record", response_model=CognitiveLoadResponse)
async def record_cognitive_load(
    request: CognitiveLoadRequest,
    db: Session = Depends(get_db)
):
    """Record cognitive load metrics"""
    try:
        metric = CognitiveLoadOperations.record_cognitive_load(
            session=db,
            user_id=request.user_id,
            mental_effort=request.mental_effort,
            working_memory_load=request.working_memory_load,
            attention_level=request.attention_level,
            stress_level=request.stress_level,
            session_id=request.session_id
        )
        
        db.commit()
        
        return CognitiveLoadResponse(
            id=metric.id,
            user_id=metric.user_id,
            overall_cognitive_load=metric.overall_cognitive_load,
            is_overloaded=metric.is_overloaded,
            recommended_break=metric.recommended_break,
            recommended_pace=metric.recommended_pace
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording cognitive load: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record cognitive load"
        )

@router.get("/current/{user_id}", response_model=CognitiveLoadResponse)
async def get_current_cognitive_load(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get current cognitive load for user"""
    try:
        metric = CognitiveLoadOperations.get_current_cognitive_load(db, user_id)
        
        if not metric:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No cognitive load data found"
            )
        
        return CognitiveLoadResponse(
            id=metric.id,
            user_id=metric.user_id,
            overall_cognitive_load=metric.overall_cognitive_load,
            is_overloaded=metric.is_overloaded,
            recommended_break=metric.recommended_break,
            recommended_pace=metric.recommended_pace
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cognitive load: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cognitive load"
        )

@router.get("/history/{user_id}")
async def get_cognitive_load_history(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get cognitive load history"""
    try:
        metrics = CognitiveLoadOperations.get_cognitive_load_history(db, user_id, days)
        
        return {
            "user_id": user_id,
            "days": days,
            "count": len(metrics),
            "metrics": [
                {
                    "id": m.id,
                    "overall_load": m.overall_cognitive_load,
                    "is_overloaded": m.is_overloaded,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ]
        }
    except Exception as e:
        logger.error(f"Error getting cognitive load history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cognitive load history"
        )
