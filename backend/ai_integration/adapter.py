"""AI module adapter for database and security integration"""
import logging
from typing import Dict, List, Any
from sqlalchemy.orm import Session

from database.operations import (
    CognitiveLoadOperations,
    ContentChunkingOperations,
    RevisionEngineOperations,
    AdaptiveLearningOperations
)
from database.models import (
    CognitiveLoadMetric,
    ContentChunk,
    RevisionItem,
    LearningPath,
    PerformanceMetric
)

logger = logging.getLogger(__name__)

class AIModuleAdapter:
    """Adapter for AI modules to interact with database and security"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ========== Cognitive Load Integration ==========
    
    def get_cognitive_load_for_adaptation(self, user_id: int) -> Dict[str, Any]:
        """Get cognitive load data for adaptive planning"""
        try:
            current_load = CognitiveLoadOperations.get_current_cognitive_load(
                self.db, user_id
            )
            
            if not current_load:
                return {"status": "no_data"}
            
            return {
                "status": "success",
                "overall_load": current_load.overall_cognitive_load,
                "is_overloaded": current_load.is_overloaded,
                "recommended_pace": current_load.recommended_pace,
                "recommended_break": current_load.recommended_break,
                "mental_effort": current_load.mental_effort,
                "working_memory": current_load.working_memory_load,
                "attention": current_load.attention_level,
                "stress": current_load.stress_level
            }
        except Exception as e:
            logger.error(f"Error getting cognitive load: {e}")
            return {"status": "error", "message": str(e)}
    
    def record_cognitive_load_from_ai(
        self,
        user_id: int,
        mental_effort: float,
        working_memory_load: float,
        attention_level: float,
        stress_level: float,
        session_id: int = None
    ) -> Dict[str, Any]:
        """Record cognitive load from AI module"""
        try:
            metric = CognitiveLoadOperations.record_cognitive_load(
                self.db, user_id, mental_effort, working_memory_load,
                attention_level, stress_level, session_id
            )
            self.db.commit()
            
            return {
                "status": "success",
                "metric_id": metric.id,
                "overall_load": metric.overall_cognitive_load
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording cognitive load: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========== Content Chunking Integration ==========
    
    def get_chunk_difficulty_analysis(self, chunk_id: int) -> Dict[str, Any]:
        """Get chunk difficulty analysis for content adaptation"""
        try:
            analytics = ContentChunkingOperations.get_chunk_analytics(
                self.db, chunk_id
            )
            
            return {
                "status": "success",
                "chunk_id": chunk_id,
                "analytics": analytics
            }
        except Exception as e:
            logger.error(f"Error getting chunk analytics: {e}")
            return {"status": "error", "message": str(e)}
    
    def record_chunk_comprehension(
        self,
        user_id: int,
        chunk_id: int,
        comprehension_score: float,
        time_spent: int
    ) -> Dict[str, Any]:
        """Record chunk comprehension from AI analysis"""
        try:
            interaction = ContentChunkingOperations.record_chunk_interaction(
                self.db, user_id, chunk_id, time_spent,
                completion_percentage=100,
                comprehension_score=comprehension_score
            )
            self.db.commit()
            
            return {
                "status": "success",
                "interaction_id": interaction.id,
                "comprehension": comprehension_score
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording comprehension: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========== Revision Engine Integration ==========
    
    def get_due_items_for_revision(self, user_id: int) -> Dict[str, Any]:
        """Get items due for revision"""
        try:
            due_items = RevisionEngineOperations.get_due_items(self.db, user_id)
            
            return {
                "status": "success",
                "user_id": user_id,
                "due_count": len(due_items),
                "items": [
                    {
                        "id": item.id,
                        "type": item.item_type,
                        "subject": item.subject,
                        "difficulty": item.difficulty,
                        "ease_factor": item.ease_factor,
                        "repetitions": item.repetitions
                    }
                    for item in due_items
                ]
            }
        except Exception as e:
            logger.error(f"Error getting due items: {e}")
            return {"status": "error", "message": str(e)}
    
    def record_revision_result(
        self,
        revision_item_id: int,
        user_id: int,
        quality: int,
        time_taken: int,
        confidence: float = None
    ) -> Dict[str, Any]:
        """Record revision result from AI module"""
        try:
            review = RevisionEngineOperations.record_review(
                self.db, revision_item_id, user_id,
                quality, time_taken, confidence
            )
            self.db.commit()
            
            return {
                "status": "success",
                "review_id": review.id,
                "quality": quality,
                "was_correct": review.was_correct
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording revision: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========== Adaptive Learning Integration ==========
    
    def get_performance_for_adaptation(
        self,
        user_id: int,
        subject: str
    ) -> Dict[str, Any]:
        """Get performance metrics for adaptive decisions"""
        try:
            metrics = AdaptiveLearningOperations.get_performance_metrics(
                self.db, user_id, subject
            )
            
            if not metrics:
                return {"status": "no_data"}
            
            return {
                "status": "success",
                "user_id": user_id,
                "subject": subject,
                "accuracy": metrics.accuracy,
                "speed": metrics.speed,
                "consistency": metrics.consistency,
                "retention_rate": metrics.retention_rate,
                "mastery_level": metrics.mastery_level,
                "engagement": metrics.engagement_score,
                "trend": metrics.trend
            }
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {"status": "error", "message": str(e)}
    
    def make_adaptive_recommendation(
        self,
        user_id: int,
        recommendation_type: str,
        subject: str,
        current_value: str,
        recommended_value: str,
        reason: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Make adaptive recommendation from AI analysis"""
        try:
            recommendation = AdaptiveLearningOperations.make_adaptive_recommendation(
                self.db, user_id, recommendation_type, subject,
                current_value, recommended_value, reason, confidence
            )
            self.db.commit()
            
            return {
                "status": "success",
                "recommendation_id": recommendation.id,
                "type": recommendation_type,
                "confidence": confidence
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error making recommendation: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========== Utility Methods ==========
    
    def get_user_learning_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive learning summary for user"""
        try:
            # Get cognitive load
            current_load = CognitiveLoadOperations.get_current_cognitive_load(
                self.db, user_id
            )
            
            # Get due items
            due_items = RevisionEngineOperations.get_due_items(self.db, user_id)
            
            # Get active paths
            active_paths = self.db.query(LearningPath).filter(
                LearningPath.user_id == user_id,
                LearningPath.is_active == True
            ).all()
            
            return {
                "status": "success",
                "user_id": user_id,
                "cognitive_load": {
                    "current": current_load.overall_cognitive_load if current_load else None,
                    "is_overloaded": current_load.is_overloaded if current_load else False
                },
                "revision": {
                    "due_count": len(due_items),
                    "items": [item.id for item in due_items[:5]]
                },
                "learning_paths": {
                    "active_count": len(active_paths),
                    "paths": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "progress": p.progress_percentage
                        }
                        for p in active_paths
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {"status": "error", "message": str(e)}
