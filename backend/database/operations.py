"""Common database operations"""
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, text
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Common database operations"""
    
    @staticmethod
    def create_record(session: Session, model, **kwargs) -> Any:
        """Create a new record"""
        try:
            record = model(**kwargs)
            session.add(record)
            session.flush()
            logger.debug(f"Created {model.__name__} record: {record.id}")
            return record
        except Exception as e:
            logger.error(f"Error creating {model.__name__}: {e}")
            raise
    
    @staticmethod
    def get_record_by_id(session: Session, model, record_id: str) -> Optional[Any]:
        """Get record by ID"""
        try:
            record = session.query(model).filter(model.id == record_id).first()
            return record
        except Exception as e:
            logger.error(f"Error fetching {model.__name__}: {e}")
            raise
    
    @staticmethod
    def update_record(session: Session, record: Any, **kwargs) -> Any:
        """Update a record"""
        try:
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.flush()
            logger.debug(f"Updated {record.__class__.__name__} record: {record.id}")
            return record
        except Exception as e:
            logger.error(f"Error updating record: {e}")
            raise
    
    @staticmethod
    def delete_record(session: Session, record: Any) -> bool:
        """Delete a record"""
        try:
            session.delete(record)
            session.flush()
            logger.debug(f"Deleted {record.__class__.__name__} record: {record.id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting record: {e}")
            raise
    
    @staticmethod
    def bulk_create(session: Session, model, records: List[Dict]) -> List[Any]:
        """Create multiple records in a single transaction"""
        try:
            created_records = []
            for record_data in records:
                record = model(**record_data)
                session.add(record)
                created_records.append(record)
            session.flush()
            logger.debug(f"Created {len(created_records)} {model.__name__} records")
            return created_records
        except Exception as e:
            logger.error(f"Error bulk creating {model.__name__}: {e}")
            raise
    
    @staticmethod
    def bulk_update(session: Session, model, updates: Dict[str, Dict]) -> int:
        """Update multiple records"""
        try:
            count = 0
            for record_id, update_data in updates.items():
                record = session.query(model).filter(model.id == record_id).first()
                if record:
                    for key, value in update_data.items():
                        if hasattr(record, key):
                            setattr(record, key, value)
                    count += 1
            session.flush()
            logger.debug(f"Updated {count} {model.__name__} records")
            return count
        except Exception as e:
            logger.error(f"Error bulk updating {model.__name__}: {e}")
            raise
    
    @staticmethod
    def count_records(session: Session, model, **filters) -> int:
        """Count records matching filters"""
        try:
            query = session.query(model)
            for key, value in filters.items():
                if hasattr(model, key):
                    query = query.filter(getattr(model, key) == value)
            return query.count()
        except Exception as e:
            logger.error(f"Error counting {model.__name__}: {e}")
            raise
    
    @staticmethod
    def get_records_paginated(
        session: Session,
        model,
        page: int = 1,
        page_size: int = 20,
        **filters
    ) -> tuple:
        """Get paginated records"""
        try:
            query = session.query(model)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(model, key):
                    query = query.filter(getattr(model, key) == value)
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            records = query.offset(offset).limit(page_size).all()
            
            return records, total
        except Exception as e:
            logger.error(f"Error fetching paginated {model.__name__}: {e}")
            raise
    
    @staticmethod
    def execute_raw_query(session: Session, query_str: str, params: Dict = None) -> List:
        """Execute raw SQL query"""
        try:
            result = session.execute(text(query_str), params or {})
            return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            raise


class DataConsistencyChecker:
    """Check and fix data consistency issues"""
    
    @staticmethod
    def check_orphaned_records(session: Session, model, foreign_key_model) -> int:
        """Find records with missing foreign key references"""
        try:
            # This is a generic check - implement specific logic per model
            logger.info(f"Checking for orphaned {model.__name__} records")
            return 0
        except Exception as e:
            logger.error(f"Error checking orphaned records: {e}")
            raise
    
    @staticmethod
    def fix_missing_timestamps(session: Session, model) -> int:
        """Fix records with missing timestamps"""
        try:
            records = session.query(model).filter(
                model.updated_at.is_(None)
            ).all()
            
            for record in records:
                record.updated_at = datetime.utcnow()
            
            session.flush()
            logger.info(f"Fixed {len(records)} records with missing timestamps")
            return len(records)
        except Exception as e:
            logger.error(f"Error fixing timestamps: {e}")
            raise
    
    @staticmethod
    def validate_data_integrity(session: Session) -> Dict[str, Any]:
        """Validate overall data integrity"""
        try:
            results = {
                "status": "healthy",
                "issues": []
            }
            
            # Add specific integrity checks here
            
            return results
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            raise



# ============================================================================
# FEATURE 1: COGNITIVE LOAD OPERATIONS
# ============================================================================

class CognitiveLoadOperations:
    """Operations for cognitive load management"""
    
    @staticmethod
    def record_cognitive_load(
        session: Session,
        user_id: int,
        mental_effort: float,
        working_memory_load: float,
        attention_level: float,
        stress_level: float,
        session_id: Optional[int] = None
    ) -> Any:
        """Record cognitive load metrics"""
        from database.models import CognitiveLoadMetric
        
        try:
            # Calculate overall load
            overall_load = (mental_effort + working_memory_load + attention_level + stress_level) / 4
            
            # Get user thresholds
            thresholds = session.query(CognitiveLoadMetric).filter_by(user_id=user_id).first()
            
            # Determine if overloaded
            is_overloaded = overall_load > 75
            recommended_break = overall_load > 80
            
            # Determine pace
            if overall_load < 40:
                pace = "fast"
            elif overall_load < 70:
                pace = "normal"
            else:
                pace = "slow"
            
            metric = CognitiveLoadMetric(
                user_id=user_id,
                session_id=session_id,
                mental_effort=mental_effort,
                working_memory_load=working_memory_load,
                attention_level=attention_level,
                stress_level=stress_level,
                overall_cognitive_load=overall_load,
                is_overloaded=is_overloaded,
                recommended_break=recommended_break,
                recommended_pace=pace
            )
            
            session.add(metric)
            session.flush()
            logger.info(f"Recorded cognitive load for user {user_id}: {overall_load:.1f}")
            return metric
        except Exception as e:
            logger.error(f"Error recording cognitive load: {e}")
            raise
    
    @staticmethod
    def get_current_cognitive_load(session: Session, user_id: int) -> Optional[Any]:
        """Get current cognitive load for user"""
        from database.models import CognitiveLoadMetric
        
        try:
            return session.query(CognitiveLoadMetric)\
                .filter_by(user_id=user_id)\
                .order_by(CognitiveLoadMetric.timestamp.desc())\
                .first()
        except Exception as e:
            logger.error(f"Error getting cognitive load: {e}")
            raise
    
    @staticmethod
    def get_cognitive_load_history(
        session: Session,
        user_id: int,
        days: int = 7
    ) -> List[Any]:
        """Get cognitive load history"""
        from database.models import CognitiveLoadMetric
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return session.query(CognitiveLoadMetric)\
                .filter(
                    CognitiveLoadMetric.user_id == user_id,
                    CognitiveLoadMetric.timestamp >= cutoff_date
                )\
                .order_by(CognitiveLoadMetric.timestamp.desc())\
                .all()
        except Exception as e:
            logger.error(f"Error getting cognitive load history: {e}")
            raise


# ============================================================================
# FEATURE 2: CONTENT CHUNKING OPERATIONS
# ============================================================================

class ContentChunkingOperations:
    """Operations for content chunking"""
    
    @staticmethod
    def create_chunks_from_material(
        session: Session,
        material_id: int,
        chunks_data: List[Dict]
    ) -> List[Any]:
        """Create multiple chunks for a material"""
        from database.models import ContentChunk
        
        try:
            created_chunks = []
            for idx, chunk_data in enumerate(chunks_data, 1):
                chunk = ContentChunk(
                    material_id=material_id,
                    chunk_number=idx,
                    title=chunk_data.get("title"),
                    content=chunk_data.get("content"),
                    estimated_cognitive_load=chunk_data.get("estimated_cognitive_load", 50),
                    estimated_duration=chunk_data.get("estimated_duration", 15),
                    difficulty_level=chunk_data.get("difficulty_level", "medium"),
                    learning_objectives=chunk_data.get("learning_objectives"),
                    key_concepts=chunk_data.get("key_concepts")
                )
                session.add(chunk)
                created_chunks.append(chunk)
            
            session.flush()
            logger.info(f"Created {len(created_chunks)} chunks for material {material_id}")
            return created_chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise
    
    @staticmethod
    def record_chunk_interaction(
        session: Session,
        user_id: int,
        chunk_id: int,
        time_spent: int,
        completion_percentage: float,
        comprehension_score: Optional[float] = None,
        cognitive_load: Optional[float] = None
    ) -> Any:
        """Record user interaction with chunk"""
        from database.models import ChunkInteraction
        
        try:
            interaction = ChunkInteraction(
                user_id=user_id,
                chunk_id=chunk_id,
                time_spent=time_spent,
                completion_percentage=completion_percentage,
                comprehension_score=comprehension_score,
                cognitive_load_during=cognitive_load
            )
            
            session.add(interaction)
            session.flush()
            logger.info(f"Recorded interaction for user {user_id} on chunk {chunk_id}")
            return interaction
        except Exception as e:
            logger.error(f"Error recording chunk interaction: {e}")
            raise
    
    @staticmethod
    def get_chunk_analytics(session: Session, chunk_id: int) -> Dict[str, Any]:
        """Get analytics for a chunk"""
        from database.models import ChunkInteraction
        
        try:
            interactions = session.query(ChunkInteraction)\
                .filter_by(chunk_id=chunk_id)\
                .all()
            
            if not interactions:
                return {"total_interactions": 0}
            
            avg_comprehension = sum(i.comprehension_score for i in interactions if i.comprehension_score) / len([i for i in interactions if i.comprehension_score])
            avg_time = sum(i.time_spent for i in interactions) / len(interactions)
            
            return {
                "total_interactions": len(interactions),
                "avg_comprehension": avg_comprehension,
                "avg_time_spent": avg_time,
                "completion_rate": sum(i.completion_percentage for i in interactions) / len(interactions)
            }
        except Exception as e:
            logger.error(f"Error getting chunk analytics: {e}")
            raise


# ============================================================================
# FEATURE 3: REVISION ENGINE OPERATIONS
# ============================================================================

class RevisionEngineOperations:
    """Operations for spaced repetition"""
    
    @staticmethod
    def create_revision_item(
        session: Session,
        user_id: int,
        item_type: str,
        item_id: int,
        subject: str,
        difficulty: str = "medium"
    ) -> Any:
        """Create a revision item. difficulty: 'easy'->1, 'medium'->3, 'hard'->5."""
        from database.models import RevisionItem
        
        try:
            diff_map = {"easy": 1, "medium": 3, "hard": 5}
            difficulty_int = diff_map.get(str(difficulty).lower(), 3)
            item = RevisionItem(
                user_id=user_id,
                item_type=item_type,
                item_id=item_id,
                subject=subject,
                question="",  # required; set when linking to flashcard/chunk
                answer="",
                difficulty=difficulty_int,
                ease_factor=2.5,
                interval=1,
                next_review=datetime.utcnow()
            )
            
            session.add(item)
            session.flush()
            logger.info(f"Created revision item for user {user_id}")
            return item
        except Exception as e:
            logger.error(f"Error creating revision item: {e}")
            raise
    
    @staticmethod
    def get_due_items(session: Session, user_id: int) -> List[Any]:
        """Get items due for review"""
        from database.models import RevisionItem
        
        try:
            return session.query(RevisionItem)\
                .filter(
                    RevisionItem.user_id == user_id,
                    RevisionItem.next_review <= datetime.utcnow()
                )\
                .order_by(RevisionItem.next_review)\
                .all()
        except Exception as e:
            logger.error(f"Error getting due items: {e}")
            raise
    
    @staticmethod
    def record_review(
        session: Session,
        revision_item_id: int,
        user_id: int,
        quality: int,
        time_taken: int,
        confidence: Optional[float] = None
    ) -> Any:
        """Record a review session"""
        from database.models import RevisionReview, RevisionItem
        
        try:
            # Record review
            review = RevisionReview(
                revision_item_id=revision_item_id,
                user_id=user_id,
                quality=quality,
                was_correct=quality >= 3,
                time_taken=time_taken,
                confidence=confidence
            )
            
            session.add(review)
            
            # Update revision item using SM-2 algorithm
            item = session.query(RevisionItem).filter_by(id=revision_item_id).first()
            if item:
                item.last_reviewed = datetime.utcnow()
                item.repetitions += 1
                
                if quality >= 3:
                    item.correct_count += 1
                    # SM-2 algorithm
                    if item.repetitions == 1:
                        item.interval = 1
                    elif item.repetitions == 2:
                        item.interval = 3
                    else:
                        item.interval = int(item.interval * item.ease_factor)
                    
                    item.ease_factor = max(1.3, item.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
                else:
                    item.incorrect_count += 1
                    item.interval = 1
                    item.repetitions = 0
                
                item.next_review = datetime.utcnow() + timedelta(days=item.interval)
            
            session.flush()
            logger.info(f"Recorded review for item {revision_item_id}")
            return review
        except Exception as e:
            logger.error(f"Error recording review: {e}")
            raise


# ============================================================================
# FEATURE 4: ADAPTIVE LEARNING PATH OPERATIONS
# ============================================================================

class AdaptiveLearningOperations:
    """Operations for adaptive learning paths"""
    
    @staticmethod
    def create_learning_path(
        session: Session,
        user_id: int,
        name: str,
        subject: str,
        goal: str,
        modules_data: List[Dict]
    ) -> Any:
        """Create a learning path with modules"""
        from database.models import LearningPath, PathModule
        
        try:
            path = LearningPath(
                user_id=user_id,
                name=name,
                subject=subject,
                goal=goal,
                total_modules=len(modules_data),
                difficulty_level="medium",
                pace="normal"
            )
            
            session.add(path)
            session.flush()
            
            # Create modules
            for idx, module_data in enumerate(modules_data, 1):
                module = PathModule(
                    learning_path_id=path.id,
                    module_number=idx,
                    title=module_data.get("title"),
                    description=module_data.get("description"),
                    content_chunks=module_data.get("content_chunks"),
                    learning_objectives=module_data.get("learning_objectives"),
                    difficulty_level=module_data.get("difficulty_level", "medium"),
                    estimated_duration=module_data.get("estimated_duration", 60)
                )
                session.add(module)
            
            session.flush()
            logger.info(f"Created learning path {path.id} for user {user_id}")
            return path
        except Exception as e:
            logger.error(f"Error creating learning path: {e}")
            raise
    
    @staticmethod
    def get_performance_metrics(
        session: Session,
        user_id: int,
        subject: str
    ) -> Optional[Any]:
        """Get performance metrics for user"""
        from database.models import PerformanceMetric
        
        try:
            return session.query(PerformanceMetric)\
                .filter(
                    PerformanceMetric.user_id == user_id,
                    PerformanceMetric.subject == subject
                )\
                .order_by(PerformanceMetric.calculated_at.desc())\
                .first()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            raise
    
    @staticmethod
    def make_adaptive_recommendation(
        session: Session,
        user_id: int,
        recommendation_type: str,
        subject: str,
        current_value: str,
        recommended_value: str,
        reason: str,
        confidence: float
    ) -> Any:
        """Create an adaptive recommendation. Coerces current_value/recommended_value to float when numeric."""
        from database.models import AdaptiveRecommendation
        
        try:
            def _to_float(v):
                if v is None:
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            recommendation = AdaptiveRecommendation(
                user_id=user_id,
                recommendation_type=recommendation_type,
                subject=subject,
                current_value=_to_float(current_value),
                recommended_value=_to_float(recommended_value),
                content=reason or "",  # required; use reason as content
                reason=reason,
                confidence_score=confidence,
                expected_impact="medium"
            )
            
            session.add(recommendation)
            session.flush()
            logger.info(f"Created recommendation for user {user_id}")
            return recommendation
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            raise
