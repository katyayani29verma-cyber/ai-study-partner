"""Revision engine routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from database.session import SessionLocal
from database.operations import RevisionEngineOperations

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class RevisionItemRequest(BaseModel):
    user_id: int
    item_type: str
    item_id: int
    subject: str
    difficulty: str = "medium"

class RevisionReviewRequest(BaseModel):
    revision_item_id: int
    user_id: int
    quality: int
    time_taken: int
    confidence: float = None

# Dependencies
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Routes
@router.post("/item/create")
async def create_revision_item(
    request: RevisionItemRequest,
    db: Session = Depends(get_db)
):
    """Create revision item"""
    try:
        item = RevisionEngineOperations.create_revision_item(
            session=db,
            user_id=request.user_id,
            item_type=request.item_type,
            item_id=request.item_id,
            subject=request.subject,
            difficulty=request.difficulty
        )
        
        db.commit()
        
        return {
            "id": item.id,
            "user_id": item.user_id,
            "item_type": item.item_type,
            "next_review": item.next_review.isoformat(),
            "ease_factor": item.ease_factor
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating revision item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create revision item"
        )

@router.get("/due/{user_id}")
async def get_due_items(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get items due for review"""
    try:
        due_items = RevisionEngineOperations.get_due_items(db, user_id)
        
        return {
            "user_id": user_id,
            "due_count": len(due_items),
            "items": [
                {
                    "id": item.id,
                    "item_type": item.item_type,
                    "subject": item.subject,
                    "difficulty": item.difficulty,
                    "next_review": item.next_review.isoformat(),
                    "ease_factor": item.ease_factor,
                    "repetitions": item.repetitions
                }
                for item in due_items
            ]
        }
    except Exception as e:
        logger.error(f"Error getting due items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get due items"
        )

@router.post("/review")
async def record_review(
    request: RevisionReviewRequest,
    db: Session = Depends(get_db)
):
    """Record review session"""
    try:
        review = RevisionEngineOperations.record_review(
            session=db,
            revision_item_id=request.revision_item_id,
            user_id=request.user_id,
            quality=request.quality,
            time_taken=request.time_taken,
            confidence=request.confidence
        )
        
        db.commit()
        
        return {
            "id": review.id,
            "revision_item_id": review.revision_item_id,
            "quality": review.quality,
            "was_correct": review.was_correct,
            "timestamp": review.timestamp.isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording review: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record review"
        )

@router.get("/schedule/{user_id}")
async def get_revision_schedule(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get revision schedule"""
    try:
        from database.models import RevisionSchedule
        
        schedule = db.query(RevisionSchedule).filter_by(user_id=user_id).first()
        
        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )
        
        return {
            "user_id": user_id,
            "daily_target": schedule.daily_target_items,
            "items_due_today": schedule.items_due_today,
            "items_completed_today": schedule.items_completed_today,
            "preferred_time": schedule.preferred_study_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get schedule"
        )
