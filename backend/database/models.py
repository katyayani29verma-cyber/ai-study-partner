"""Database models - Clean implementation"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from database.config import Base


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.email}>"


class StudyMaterial(Base):
    """Study material model"""
    __tablename__ = "study_materials"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    subject = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")
    
    def __repr__(self):
        return f"<StudyMaterial {self.title}>"


class StudySession(Base):
    """Study session model"""
    __tablename__ = "study_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    material_id = Column(Integer, ForeignKey("study_materials.id"), nullable=False)
    duration_minutes = Column(Integer, nullable=True)
    performance_score = Column(Float, nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    material = relationship("StudyMaterial")
    
    def __repr__(self):
        return f"<StudySession user={self.user_id}>"


class Flashcard(Base):
    """Flashcard model for spaced repetition"""
    __tablename__ = "flashcards"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=1)
    repetitions = Column(Integer, default=0)
    next_review_date = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")
    
    def __repr__(self):
        return f"<Flashcard user={self.user_id}>"


class CognitiveLoadMetric(Base):
    """Cognitive load measurement - Production-ready schema"""
    __tablename__ = "cognitive_load_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=True, index=True)
    mental_effort = Column(Float, nullable=False)
    time_on_task = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)
    working_memory_load = Column(Float, nullable=True)
    attention_level = Column(Float, nullable=True)
    stress_level = Column(Float, nullable=True)
    overall_cognitive_load = Column(Float, nullable=True)
    is_overloaded = Column(Boolean, default=False)
    recommended_break = Column(Boolean, default=False)
    recommended_pace = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User")
    session = relationship("StudySession")
    
    def __repr__(self):
        return f"<CognitiveLoadMetric user={self.user_id} load={self.overall_cognitive_load}>"


class ContentChunk(Base):
    """Content chunk for chunking strategy - Production-ready schema"""
    __tablename__ = "content_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("study_materials.id"), nullable=False, index=True)
    chunk_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    difficulty_level = Column(String(50), nullable=True)
    estimated_time = Column(Integer, nullable=True)
    estimated_duration = Column(Integer, nullable=True)
    estimated_cognitive_load = Column(Float, nullable=True)
    learning_objectives = Column(Text, nullable=True)
    key_concepts = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    material = relationship("StudyMaterial")
    interactions = relationship("ChunkInteraction", back_populates="chunk")
    
    def __repr__(self):
        return f"<ContentChunk material={self.material_id} num={self.chunk_number}>"


class RevisionItem(Base):
    """Revision item for spaced repetition - Production-ready schema"""
    __tablename__ = "revision_items"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_type = Column(String(50), nullable=True)
    item_id = Column(Integer, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    subject = Column(String(100), nullable=True)
    difficulty = Column(Integer, default=3)
    category = Column(String(100), nullable=True)
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=1)
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime, nullable=True, index=True)
    last_reviewed = Column(DateTime, nullable=True)
    correct_count = Column(Integer, default=0)
    incorrect_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")
    reviews = relationship("RevisionReview", back_populates="item")
    
    def __repr__(self):
        return f"<RevisionItem user={self.user_id} subject={self.subject}>"


class LearningPath(Base):
    """Personalized learning path - Production-ready schema"""
    __tablename__ = "learning_paths"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    subject = Column(String(255), nullable=False)
    goal = Column(Text, nullable=True)
    difficulty_level = Column(String(50), nullable=True)
    estimated_duration = Column(Integer, nullable=True)
    total_modules = Column(Integer, default=0)
    current_module = Column(Integer, default=0)
    pace = Column(String(50), nullable=True)
    progress = Column(Float, default=0.0)
    progress_percentage = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")
    modules = relationship("PathModule", back_populates="path")
    
    def __repr__(self):
        return f"<LearningPath user={self.user_id} subject={self.subject}>"


class PerformanceMetric(Base):
    """Performance metrics - Production-ready schema"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    subject = Column(String(255), nullable=True)
    accuracy = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    retention = Column(Float, nullable=True)
    consistency = Column(Float, nullable=True)
    retention_rate = Column(Float, nullable=True)
    mastery_level = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True)
    trend = Column(String(50), nullable=True)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User")
    
    def __repr__(self):
        return f"<PerformanceMetric user={self.user_id} subject={self.subject}>"


class ChunkInteraction(Base):
    """User interaction with content chunks - Production-ready"""
    __tablename__ = "chunk_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("content_chunks.id"), nullable=False, index=True)
    time_spent = Column(Integer, nullable=True)
    completion_percentage = Column(Float, default=0.0)
    comprehension_score = Column(Float, nullable=True)
    cognitive_load_during = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    chunk = relationship("ContentChunk", back_populates="interactions")
    
    def __repr__(self):
        return f"<ChunkInteraction user={self.user_id} chunk={self.chunk_id}>"


class RevisionReview(Base):
    """Record of revision item reviews - Production-ready"""
    __tablename__ = "revision_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    revision_item_id = Column(Integer, ForeignKey("revision_items.id"), nullable=False, index=True)
    quality = Column(Integer, nullable=False)  # 0-5 rating
    was_correct = Column(Boolean, nullable=False)
    time_taken = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    ease_factor_change = Column(Float, nullable=True)
    new_interval = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    item = relationship("RevisionItem", back_populates="reviews")
    
    def __repr__(self):
        return f"<RevisionReview user={self.user_id} item={self.revision_item_id} quality={self.quality}>"


class RevisionSchedule(Base):
    """Spaced repetition schedule - Production-ready"""
    __tablename__ = "revision_schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("revision_items.id"), nullable=False, index=True)
    next_review_date = Column(DateTime, nullable=False, index=True)
    daily_target_items = Column(Integer, default=10)
    items_due_today = Column(Integer, default=0)
    items_completed_today = Column(Integer, default=0)
    preferred_study_time = Column(String(50), nullable=True)
    priority = Column(Integer, default=0)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")
    
    def __repr__(self):
        return f"<RevisionSchedule user={self.user_id} next={self.next_review_date}>"


class PathModule(Base):
    """Module within a learning path - Production-ready"""
    __tablename__ = "path_modules"
    
    id = Column(Integer, primary_key=True, index=True)
    learning_path_id = Column(Integer, ForeignKey("learning_paths.id"), nullable=False, index=True)
    module_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content_chunks = Column(Text, nullable=True)  # JSON or comma-separated IDs
    learning_objectives = Column(Text, nullable=True)
    estimated_duration = Column(Integer, nullable=True)
    difficulty_level = Column(String(50), nullable=True)
    progress = Column(Float, default=0.0)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    path = relationship("LearningPath", back_populates="modules")
    
    def __repr__(self):
        return f"<PathModule learning_path={self.learning_path_id} module={self.module_number}>"


class AdaptiveRecommendation(Base):
    """Adaptive learning recommendations - Production-ready"""
    __tablename__ = "adaptive_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    path_id = Column(Integer, ForeignKey("learning_paths.id"), nullable=True, index=True)
    recommendation_type = Column(String(100), nullable=False)
    subject = Column(String(255), nullable=True)
    current_value = Column(Float, nullable=True)
    recommended_value = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    expected_impact = Column(String(50), nullable=True)
    content = Column(Text, nullable=False)
    reason = Column(Text, nullable=True)
    priority = Column(Integer, default=0)
    is_acted_upon = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    acted_upon_at = Column(DateTime, nullable=True)
    
    user = relationship("User")
    path = relationship("LearningPath")
    
    def __repr__(self):
        return f"<AdaptiveRecommendation user={self.user_id} type={self.recommendation_type}>"
