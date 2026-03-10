# Database Module

Feature-oriented database for AI Study Partner.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```python
from database.config import DATABASE_URL, POOL_SIZE
from database.session import SessionLocal

db = SessionLocal()
```

### Basic Usage

```python
from database.operations import CognitiveLoadOperations

# Record cognitive load
metric = CognitiveLoadOperations.record_cognitive_load(
    db, user_id=1,
    mental_effort=65, working_memory_load=70,
    attention_level=75, stress_level=60
)
```

---

## Features

### 1. Cognitive Load Management
Track and optimize student cognitive load during study sessions.

**Tables:**
- `cognitive_load_metrics` - Real-time measurements
- `cognitive_load_thresholds` - User preferences

**Operations:**
```python
CognitiveLoadOperations.record_cognitive_load()
CognitiveLoadOperations.get_current_cognitive_load()
CognitiveLoadOperations.get_cognitive_load_history()
```

### 2. Content Chunking
Break down study materials into manageable chunks.

**Tables:**
- `content_chunks` - Chunked materials
- `chunk_interactions` - User interactions

**Operations:**
```python
ContentChunkingOperations.create_chunks_from_material()
ContentChunkingOperations.record_chunk_interaction()
ContentChunkingOperations.get_chunk_analytics()
```

### 3. Revision Engine
Spaced repetition scheduling using SM-2 algorithm.

**Tables:**
- `revision_items` - Items to review
- `revision_reviews` - Review sessions
- `revision_schedule` - Daily plans

**Operations:**
```python
RevisionEngineOperations.create_revision_item()
RevisionEngineOperations.get_due_items()
RevisionEngineOperations.record_review()
```

### 4. Adaptive Learning Path
Personalized learning paths that adapt to performance.

**Tables:**
- `learning_paths` - Learning journeys
- `path_modules` - Path modules
- `adaptive_recommendations` - Suggestions
- `performance_metrics` - Performance data

**Operations:**
```python
AdaptiveLearningOperations.create_learning_path()
AdaptiveLearningOperations.get_performance_metrics()
AdaptiveLearningOperations.make_adaptive_recommendation()
```

---

## File Structure

```
database/
├── __init__.py              # Package initialization
├── config.py                # Database configuration
├── models.py                # SQLAlchemy models (14 tables)
├── session.py               # Session management
├── operations.py            # Database operations
├── init.py                  # Database initialization
├── requirements.txt         # Dependencies
├── README.md                # This file
├── FEATURE_ORIENTED_SCHEMA.md  # Schema documentation
└── FEATURE_GUIDE.md         # Usage guide
```

---

## Models

### Core Tables (7)
- `User` - User accounts
- `StudyMaterial` - Study materials
- `StudySession` - Study sessions
- `Flashcard` - Flashcards
- `KnowledgeGap` - Knowledge gaps
- `CurriculumData` - Curriculum data
- `SocraticSession` - Socratic sessions
- `AuditLog` - Audit logs

### Feature 1: Cognitive Load (2)
- `CognitiveLoadMetric` - Load measurements
- `CognitiveLoadThreshold` - User thresholds

### Feature 2: Content Chunking (2)
- `ContentChunk` - Chunked content
- `ChunkInteraction` - User interactions

### Feature 3: Revision Engine (3)
- `RevisionItem` - Items to review
- `RevisionReview` - Review sessions
- `RevisionSchedule` - Daily schedules

### Feature 4: Adaptive Learning (4)
- `LearningPath` - Learning paths
- `PathModule` - Path modules
- `AdaptiveRecommendation` - Recommendations
- `PerformanceMetric` - Performance metrics

---

## Operations

### CognitiveLoadOperations
```python
from database.operations import CognitiveLoadOperations

# Record load
metric = CognitiveLoadOperations.record_cognitive_load(
    session, user_id, mental_effort, working_memory_load,
    attention_level, stress_level, session_id
)

# Get current
current = CognitiveLoadOperations.get_current_cognitive_load(session, user_id)

# Get history
history = CognitiveLoadOperations.get_cognitive_load_history(session, user_id, days=7)
```

### ContentChunkingOperations
```python
from database.operations import ContentChunkingOperations

# Create chunks
chunks = ContentChunkingOperations.create_chunks_from_material(
    session, material_id, chunks_data
)

# Record interaction
interaction = ContentChunkingOperations.record_chunk_interaction(
    session, user_id, chunk_id, time_spent,
    completion_percentage, comprehension_score, cognitive_load
)

# Get analytics
analytics = ContentChunkingOperations.get_chunk_analytics(session, chunk_id)
```

### RevisionEngineOperations
```python
from database.operations import RevisionEngineOperations

# Create item
item = RevisionEngineOperations.create_revision_item(
    session, user_id, item_type, item_id, subject, difficulty
)

# Get due items
due = RevisionEngineOperations.get_due_items(session, user_id)

# Record review (updates SM-2 parameters)
review = RevisionEngineOperations.record_review(
    session, revision_item_id, user_id, quality,
    time_taken, confidence
)
```

### AdaptiveLearningOperations
```python
from database.operations import AdaptiveLearningOperations

# Create path
path = AdaptiveLearningOperations.create_learning_path(
    session, user_id, name, subject, goal, modules_data
)

# Get metrics
metrics = AdaptiveLearningOperations.get_performance_metrics(
    session, user_id, subject
)

# Make recommendation
rec = AdaptiveLearningOperations.make_adaptive_recommendation(
    session, user_id, recommendation_type, subject,
    current_value, recommended_value, reason, confidence
)
```

---

## Session Management

### Context Manager
```python
from database.session import get_db

with get_db() as db:
    # Your operations
    pass
```

### Manual Management
```python
from database.session import SessionLocal

db = SessionLocal()
try:
    # Your operations
    db.commit()
except Exception as e:
    db.rollback()
    raise
finally:
    db.close()
```

---

## Indexes

### Performance Optimization
- 30+ indexes for fast queries
- User-based queries: O(1)
- Time-range queries: O(log n)
- Subject queries: O(log n)

### Index Types
- User indexes: `idx_*_user`
- Time indexes: `idx_*_timestamp`
- Subject indexes: `idx_*_subject`
- Status indexes: `idx_*_active`, `idx_*_next_review`

---

## Examples

### Complete Study Session
```python
from database.session import SessionLocal
from database.operations import *

db = SessionLocal()

try:
    # 1. Record cognitive load
    load = CognitiveLoadOperations.record_cognitive_load(
        db, 1, 65, 70, 75, 60, session_id=1
    )
    
    # 2. Study chunk
    interaction = ContentChunkingOperations.record_chunk_interaction(
        db, 1, 1, 900, 100, 85, 60
    )
    
    # 3. Do revision
    due = RevisionEngineOperations.get_due_items(db, 1)
    for item in due[:5]:
        RevisionEngineOperations.record_review(
            db, item.id, 1, 4, 30, 80
        )
    
    # 4. Check performance
    metrics = AdaptiveLearningOperations.get_performance_metrics(db, 1, "Math")
    
    db.commit()
except Exception as e:
    db.rollback()
    raise
finally:
    db.close()
```

---

## Configuration

### Environment Variables
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/study_partner
POOL_SIZE=20
MAX_OVERFLOW=10
POOL_RECYCLE=3600
```

### Connection Pooling
```python
# In config.py
POOL_SIZE = 20          # Connections to keep
MAX_OVERFLOW = 10       # Additional connections
POOL_RECYCLE = 3600     # Recycle after 1 hour
POOL_PRE_PING = True    # Test connections
```

---

## Documentation

### Schema
- `FEATURE_ORIENTED_SCHEMA.md` - Complete schema documentation
- `models.py` - Table definitions with docstrings

### Usage
- `FEATURE_GUIDE.md` - Feature-specific usage guide
- `operations.py` - Operation classes with docstrings

### Setup
- `DATABASE_INSTALLATION.md` - Installation guide
- `INSTALLATION_GUIDE.md` - General installation

---

## Best Practices

1. **Always use transactions**
   ```python
   try:
       # operations
       db.commit()
   except:
       db.rollback()
   ```

2. **Use batch operations**
   ```python
   chunks = ContentChunkingOperations.create_chunks_from_material(...)
   ```

3. **Query by indexed columns**
   ```python
   items = db.query(RevisionItem).filter(RevisionItem.next_review <= now)
   ```

4. **Cache metrics**
   ```python
   metrics = get_performance_metrics(...)  # Calculate once
   ```

5. **Archive old data**
   ```python
   # Move old reviews to archive table
   ```

---

## Troubleshooting

### Connection Issues
```python
# Check connection
from database.session import SessionLocal
db = SessionLocal()
print("Connected!")
```

### Query Performance
```python
# Use EXPLAIN ANALYZE
from sqlalchemy import text
result = db.execute(text("EXPLAIN ANALYZE SELECT ..."))
```

### Data Consistency
```python
# Check integrity
from database.operations import DataConsistencyChecker
checker = DataConsistencyChecker()
```

---

## Support

### Documentation
- Schema: `FEATURE_ORIENTED_SCHEMA.md`
- Guide: `FEATURE_GUIDE.md`
- Installation: `DATABASE_INSTALLATION.md`

### Examples
- See `FEATURE_GUIDE.md` for complete examples
- See `operations.py` for operation signatures

---

**Version:** 2.0  
**Last Updated:** March 5, 2026  
**Status:** Production Ready
