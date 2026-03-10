"""Initial schema creation

Revision ID: 001
Revises: 
Create Date: 2026-03-08 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables for the initial schema."""
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    
    # Create study_materials table
    op.create_table(
        'study_materials',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('subject', sa.String(length=100), nullable=True),
        sa.Column('difficulty_level', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_study_materials_id'), 'study_materials', ['id'], unique=False)
    op.create_index(op.f('ix_study_materials_user_id'), 'study_materials', ['user_id'], unique=False)
    
    # Create study_sessions table
    op.create_table(
        'study_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('material_id', sa.Integer(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['material_id'], ['study_materials.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_study_sessions_id'), 'study_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_study_sessions_user_id'), 'study_sessions', ['user_id'], unique=False)
    
    # Create cognitive_load_metrics table
    op.create_table(
        'cognitive_load_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('mental_effort', sa.Float(), nullable=False),
        sa.Column('working_memory_load', sa.Float(), nullable=False),
        sa.Column('attention_level', sa.Float(), nullable=False),
        sa.Column('stress_level', sa.Float(), nullable=False),
        sa.Column('overall_cognitive_load', sa.Float(), nullable=False),
        sa.Column('is_overloaded', sa.Boolean(), nullable=False),
        sa.Column('recommended_break', sa.Boolean(), nullable=False),
        sa.Column('recommended_pace', sa.String(length=50), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['study_sessions.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cognitive_load_metrics_id'), 'cognitive_load_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_cognitive_load_metrics_timestamp'), 'cognitive_load_metrics', ['timestamp'], unique=False)
    op.create_index(op.f('ix_cognitive_load_metrics_user_id'), 'cognitive_load_metrics', ['user_id'], unique=False)
    
    # Create content_chunks table
    op.create_table(
        'content_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('material_id', sa.Integer(), nullable=False),
        sa.Column('chunk_number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('estimated_cognitive_load', sa.Float(), nullable=True),
        sa.Column('estimated_duration', sa.Integer(), nullable=True),
        sa.Column('difficulty_level', sa.String(length=50), nullable=True),
        sa.Column('learning_objectives', sa.Text(), nullable=True),
        sa.Column('key_concepts', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['material_id'], ['study_materials.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_content_chunks_id'), 'content_chunks', ['id'], unique=False)
    op.create_index(op.f('ix_content_chunks_material_id'), 'content_chunks', ['material_id'], unique=False)
    
    # Create chunk_interactions table
    op.create_table(
        'chunk_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('chunk_id', sa.Integer(), nullable=False),
        sa.Column('time_spent', sa.Integer(), nullable=False),
        sa.Column('completion_percentage', sa.Float(), nullable=False),
        sa.Column('comprehension_score', sa.Float(), nullable=True),
        sa.Column('cognitive_load', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['content_chunks.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chunk_interactions_id'), 'chunk_interactions', ['id'], unique=False)
    op.create_index(op.f('ix_chunk_interactions_user_id'), 'chunk_interactions', ['user_id'], unique=False)
    
    # Create flashcards table
    op.create_table(
        'flashcards',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('material_id', sa.Integer(), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('difficulty', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['material_id'], ['study_materials.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_flashcards_id'), 'flashcards', ['id'], unique=False)
    op.create_index(op.f('ix_flashcards_user_id'), 'flashcards', ['user_id'], unique=False)
    
    # Create revision_items table
    op.create_table(
        'revision_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('item_type', sa.String(length=50), nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('difficulty', sa.String(length=50), nullable=True),
        sa.Column('ease_factor', sa.Float(), nullable=False, server_default='2.5'),
        sa.Column('interval', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('repetitions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('next_review', sa.DateTime(), nullable=False),
        sa.Column('last_reviewed', sa.DateTime(), nullable=True),
        sa.Column('correct_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('incorrect_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_revision_items_id'), 'revision_items', ['id'], unique=False)
    op.create_index(op.f('ix_revision_items_user_id'), 'revision_items', ['user_id'], unique=False)
    
    # Create revision_reviews table
    op.create_table(
        'revision_reviews',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('revision_item_id', sa.Integer(), nullable=False),
        sa.Column('quality', sa.Integer(), nullable=False),
        sa.Column('was_correct', sa.Boolean(), nullable=False),
        sa.Column('time_taken', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('ease_factor_change', sa.Float(), nullable=True),
        sa.Column('new_interval', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('reviewed_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['revision_item_id'], ['revision_items.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_revision_reviews_id'), 'revision_reviews', ['id'], unique=False)
    op.create_index(op.f('ix_revision_reviews_user_id'), 'revision_reviews', ['user_id'], unique=False)
    
    # Create revision_schedules table
    op.create_table(
        'revision_schedules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('next_review_date', sa.DateTime(), nullable=False),
        sa.Column('daily_target_items', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('items_due_today', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('items_completed_today', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('preferred_study_time', sa.String(length=50), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['item_id'], ['revision_items.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_revision_schedules_id'), 'revision_schedules', ['id'], unique=False)
    op.create_index(op.f('ix_revision_schedules_user_id'), 'revision_schedules', ['user_id'], unique=False)
    
    # Create learning_paths table
    op.create_table(
        'learning_paths',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('goal', sa.Text(), nullable=False),
        sa.Column('total_modules', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_module', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('pace', sa.String(length=50), nullable=True),
        sa.Column('progress_percentage', sa.Float(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_learning_paths_id'), 'learning_paths', ['id'], unique=False)
    op.create_index(op.f('ix_learning_paths_user_id'), 'learning_paths', ['user_id'], unique=False)
    
    # Create path_modules table
    op.create_table(
        'path_modules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('learning_path_id', sa.Integer(), nullable=False),
        sa.Column('module_number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('content_chunks', sa.Text(), nullable=True),
        sa.Column('learning_objectives', sa.Text(), nullable=True),
        sa.Column('estimated_duration', sa.Integer(), nullable=True),
        sa.Column('difficulty_level', sa.String(length=50), nullable=True),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0'),
        sa.Column('is_completed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['learning_path_id'], ['learning_paths.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_path_modules_id'), 'path_modules', ['id'], unique=False)
    op.create_index(op.f('ix_path_modules_learning_path_id'), 'path_modules', ['learning_path_id'], unique=False)
    
    # Create performance_metrics table
    op.create_table(
        'performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('speed', sa.Float(), nullable=False),
        sa.Column('consistency', sa.Float(), nullable=False),
        sa.Column('retention_rate', sa.Float(), nullable=False),
        sa.Column('mastery_level', sa.Float(), nullable=False),
        sa.Column('engagement_score', sa.Float(), nullable=False),
        sa.Column('trend', sa.String(length=50), nullable=True),
        sa.Column('calculated_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_performance_metrics_id'), 'performance_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_performance_metrics_user_id'), 'performance_metrics', ['user_id'], unique=False)
    
    # Create adaptive_recommendations table
    op.create_table(
        'adaptive_recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('path_id', sa.Integer(), nullable=True),
        sa.Column('recommendation_type', sa.String(length=100), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=True),
        sa.Column('current_value', sa.Float(), nullable=True),
        sa.Column('recommended_value', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('expected_impact', sa.String(length=50), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_acted_upon', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('acted_upon_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['path_id'], ['learning_paths.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_adaptive_recommendations_id'), 'adaptive_recommendations', ['id'], unique=False)
    op.create_index(op.f('ix_adaptive_recommendations_created_at'), 'adaptive_recommendations', ['created_at'], unique=False)
    op.create_index(op.f('ix_adaptive_recommendations_user_id'), 'adaptive_recommendations', ['user_id'], unique=False)


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index(op.f('ix_adaptive_recommendations_user_id'), table_name='adaptive_recommendations')
    op.drop_index(op.f('ix_adaptive_recommendations_created_at'), table_name='adaptive_recommendations')
    op.drop_index(op.f('ix_adaptive_recommendations_id'), table_name='adaptive_recommendations')
    op.drop_table('adaptive_recommendations')
    op.drop_index(op.f('ix_performance_metrics_user_id'), table_name='performance_metrics')
    op.drop_index(op.f('ix_performance_metrics_id'), table_name='performance_metrics')
    op.drop_table('performance_metrics')
    op.drop_index(op.f('ix_path_modules_learning_path_id'), table_name='path_modules')
    op.drop_index(op.f('ix_path_modules_id'), table_name='path_modules')
    op.drop_table('path_modules')
    op.drop_index(op.f('ix_learning_paths_user_id'), table_name='learning_paths')
    op.drop_index(op.f('ix_learning_paths_id'), table_name='learning_paths')
    op.drop_table('learning_paths')
    op.drop_index(op.f('ix_revision_schedules_user_id'), table_name='revision_schedules')
    op.drop_index(op.f('ix_revision_schedules_id'), table_name='revision_schedules')
    op.drop_table('revision_schedules')
    op.drop_index(op.f('ix_revision_reviews_user_id'), table_name='revision_reviews')
    op.drop_index(op.f('ix_revision_reviews_id'), table_name='revision_reviews')
    op.drop_table('revision_reviews')
    op.drop_index(op.f('ix_revision_items_user_id'), table_name='revision_items')
    op.drop_index(op.f('ix_revision_items_id'), table_name='revision_items')
    op.drop_table('revision_items')
    op.drop_index(op.f('ix_flashcards_user_id'), table_name='flashcards')
    op.drop_index(op.f('ix_flashcards_id'), table_name='flashcards')
    op.drop_table('flashcards')
    op.drop_index(op.f('ix_chunk_interactions_user_id'), table_name='chunk_interactions')
    op.drop_index(op.f('ix_chunk_interactions_id'), table_name='chunk_interactions')
    op.drop_table('chunk_interactions')
    op.drop_index(op.f('ix_content_chunks_material_id'), table_name='content_chunks')
    op.drop_index(op.f('ix_content_chunks_id'), table_name='content_chunks')
    op.drop_table('content_chunks')
    op.drop_index(op.f('ix_cognitive_load_metrics_user_id'), table_name='cognitive_load_metrics')
    op.drop_index(op.f('ix_cognitive_load_metrics_timestamp'), table_name='cognitive_load_metrics')
    op.drop_index(op.f('ix_cognitive_load_metrics_id'), table_name='cognitive_load_metrics')
    op.drop_table('cognitive_load_metrics')
    op.drop_index(op.f('ix_study_sessions_user_id'), table_name='study_sessions')
    op.drop_index(op.f('ix_study_sessions_id'), table_name='study_sessions')
    op.drop_table('study_sessions')
    op.drop_index(op.f('ix_study_materials_user_id'), table_name='study_materials')
    op.drop_index(op.f('ix_study_materials_id'), table_name='study_materials')
    op.drop_table('study_materials')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
