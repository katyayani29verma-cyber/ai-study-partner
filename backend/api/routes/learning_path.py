"""Adaptive learning path routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging
from typing import List

from database.session import SessionLocal
from database.operations import AdaptiveLearningOperations

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class ModuleData(BaseModel):
    title: str
    description: str = None
    difficulty_level: str = "medium"
    estimated_duration: int = 60
    learning_objectives: List[str] = None
    content_chunks: List[int] = None

class LearningPathRequest(BaseModel):
    user_id: int
    name: str
    subject: str
    goal: str
    modules: List[ModuleData]

class RecommendationRequest(BaseModel):
    user_id: int
    recommendation_type: str
    subject: str
    current_value: str
    recommended_value: str
    reason: str
    confidence: float

# Dependencies
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes
@router.post("/create")
async def create_learning_path(
    request: LearningPathRequest,
    db: Session = Depends(get_db)
):
    """Create learning path"""
    try:
        modules_data = [module.dict() for module in request.modules]
        
        path = AdaptiveLearningOperations.create_learning_path(
            session=db,
            user_id=request.user_id,
            name=request.name,
            subject=request.subject,
            goal=request.goal,
            modules_data=modules_data
        )
        
        db.commit()
        
        return {
            "id": path.id,
            "user_id": path.user_id,
            "name": path.name,
            "subject": path.subject,
            "total_modules": path.total_modules,
            "progress": path.progress_percentage,
            "created_at": path.created_at.isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating learning path: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create learning path"
        )

@router.get("/metrics/{user_id}/{subject}")
async def get_performance_metrics(
    user_id: int,
    subject: str,
    db: Session = Depends(get_db)
):
    """Get performance metrics"""
    try:
        metrics = AdaptiveLearningOperations.get_performance_metrics(
            session=db,
            user_id=user_id,
            subject=subject
        )
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No metrics found"
            )
        
        return {
            "user_id": user_id,
            "subject": subject,
            "accuracy": metrics.accuracy,
            "speed": metrics.speed,
            "consistency": metrics.consistency,
            "retention_rate": metrics.retention_rate,
            "mastery_level": metrics.mastery_level,
            "engagement_score": metrics.engagement_score,
            "trend": metrics.trend,
            "calculated_at": metrics.calculated_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics"
        )

@router.post("/recommendation")
async def make_recommendation(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Make adaptive recommendation"""
    try:
        recommendation = AdaptiveLearningOperations.make_adaptive_recommendation(
            session=db,
            user_id=request.user_id,
            recommendation_type=request.recommendation_type,
            subject=request.subject,
            current_value=request.current_value,
            recommended_value=request.recommended_value,
            reason=request.reason,
            confidence=request.confidence
        )
        
        db.commit()
        
        return {
            "id": recommendation.id,
            "user_id": recommendation.user_id,
            "type": recommendation.recommendation_type,
            "subject": recommendation.subject,
            "current": recommendation.current_value,
            "recommended": recommendation.recommended_value,
            "confidence": recommendation.confidence_score,
            "created_at": recommendation.created_at.isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error making recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to make recommendation"
        )

@router.get("/{user_id}")
async def get_learning_paths(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user's learning paths"""
    try:
        from database.models import LearningPath
        
        paths = db.query(LearningPath).filter(
            LearningPath.user_id == user_id,
            LearningPath.is_active == True
        ).all()
        
        return {
            "user_id": user_id,
            "paths_count": len(paths),
            "paths": [
                {
                    "id": p.id,
                    "name": p.name,
                    "subject": p.subject,
                    "progress": p.progress_percentage,
                    "current_module": p.current_module,
                    "total_modules": p.total_modules
                }
                for p in paths
            ]
        }
    except Exception as e:
        logger.error(f"Error getting learning paths: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning paths"
        )
