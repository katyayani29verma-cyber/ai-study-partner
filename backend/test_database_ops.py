"""Test database operations"""
import sys
sys.path.insert(0, '.')

from database import create_db_engine, Base, User, StudyMaterial, StudySession
from database.session import init_db_session
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

print('=== DATABASE OPERATIONS TEST ===\n')

# Create engine
engine = create_db_engine()
print('✅ Database engine created')

# Create tables
Base.metadata.create_all(bind=engine)
print('✅ Database tables created')

# Inspect tables
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f'✅ {len(tables)} tables found:')
for table in sorted(tables):
    print(f'   - {table}')

# Initialize session
SessionLocal = init_db_session(engine)
session = SessionLocal()
print('\n✅ Database session created')

# Test User model
print('\n=== Testing User Model ===')
try:
    user = User(
        username='testuser',
        email='test@example.com',
        hashed_password='hashed_pwd_123',
        full_name='Test User'
    )
    session.add(user)
    session.commit()
    print('✅ User created successfully')
    
    # Query user
    queried_user = session.query(User).filter_by(username='testuser').first()
    print(f'✅ User retrieved: {queried_user.username} ({queried_user.email})')
    
    user_id = queried_user.id
    
    # Clean up
    session.delete(queried_user)
    session.commit()
    print('✅ User deleted successfully')
except Exception as e:
    print(f'❌ User test failed: {e}')
    session.rollback()
    user_id = 1

# Test StudyMaterial model
print('\n=== Testing StudyMaterial Model ===')
try:
    # Create a user first for foreign key
    user = User(
        username='testuser2',
        email='test2@example.com',
        hashed_password='hashed_pwd_123',
        full_name='Test User 2'
    )
    session.add(user)
    session.commit()
    
    material = StudyMaterial(
        user_id=user.id,
        title='Python Basics',
        content='Introduction to Python programming',
        subject='Programming'
    )
    session.add(material)
    session.commit()
    print('✅ Study material created successfully')
    
    # Query material
    queried_material = session.query(StudyMaterial).filter_by(title='Python Basics').first()
    print(f'✅ Material retrieved: {queried_material.title}')
    
    # Clean up
    session.delete(queried_material)
    session.delete(user)
    session.commit()
    print('✅ Study material deleted successfully')
except Exception as e:
    print(f'❌ Study material test failed: {e}')
    session.rollback()

# Test StudySession model
print('\n=== Testing StudySession Model ===')
try:
    # Create user and material first
    user = User(
        username='testuser3',
        email='test3@example.com',
        hashed_password='hashed_pwd_123'
    )
    session.add(user)
    session.commit()
    
    material = StudyMaterial(
        user_id=user.id,
        title='Test Material',
        content='Test content',
        subject='Test'
    )
    session.add(material)
    session.commit()
    
    study_session = StudySession(
        user_id=user.id,
        material_id=material.id,
        duration_minutes=30,
        performance_score=85.0
    )
    session.add(study_session)
    session.commit()
    print('✅ Study session created successfully')
    
    # Query session
    queried_session = session.query(StudySession).first()
    if queried_session:
        print(f'✅ Session retrieved: {queried_session.duration_minutes} minutes')
        
        # Clean up
        session.delete(queried_session)
        session.delete(material)
        session.delete(user)
        session.commit()
        print('✅ Study session deleted successfully')
except Exception as e:
    print(f'❌ Study session test failed: {e}')
    session.rollback()

session.close()
print('\n=== ALL DATABASE TESTS PASSED ===')
