"""Database initialization"""
from sqlalchemy import text, event
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Initialize database with extensions and initial data"""
    
    @staticmethod
    def create_extensions(engine):
        """Create required PostgreSQL extensions"""
        extensions = [
            "uuid-ossp",      # UUID generation
            "pg_trgm",        # Trigram similarity search
            "pgcrypto",       # Encryption functions
            "pg_stat_statements"  # Query statistics
        ]
        
        with engine.connect() as connection:
            for ext in extensions:
                try:
                    connection.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                    logger.info(f"Created extension: {ext}")
                except Exception as e:
                    logger.warning(f"Could not create extension {ext}: {e}")
            connection.commit()
    
    @staticmethod
    def create_tables(engine, base):
        """Create all tables"""
        try:
            base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    @staticmethod
    def create_indexes(engine):
        """Create optimized indexes"""
        indexes = [
            # User indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(id) WHERE is_active = true",
            
            # Study materials indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_materials_user_id ON study_materials(user_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_materials_subject ON study_materials(subject)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_materials_processed ON study_materials(user_id) WHERE is_processed = true",
            
            # Flashcard indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flashcards_due ON flashcards(user_id, next_review_date) WHERE next_review_date IS NOT NULL",
            
            # Session indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_time ON study_sessions(user_id, start_time DESC)",
            
            # Knowledge gap indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gaps_user_status ON knowledge_gaps(user_id, status) WHERE status != 'resolved'",
            
            # Audit log indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_user_time ON audit_logs(user_id, timestamp DESC)",
        ]
        
        with engine.connect() as connection:
            for index_sql in indexes:
                try:
                    connection.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql.split('ON')[1].split('(')[0].strip()}")
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
            connection.commit()
    
    @staticmethod
    def create_constraints(engine):
        """Create database constraints"""
        constraints = [
            # Check constraints
            """
            ALTER TABLE users
            ADD CONSTRAINT check_grade_range
            CHECK (grade_standard IS NULL OR (grade_standard >= 1 AND grade_standard <= 12))
            """,
            
            """
            ALTER TABLE users
            ADD CONSTRAINT check_course_year
            CHECK (course_year IS NULL OR (course_year >= 1 AND course_year <= 6))
            """,
            
            """
            ALTER TABLE flashcards
            ADD CONSTRAINT check_ease_factor
            CHECK (ease_factor >= 1.3)
            """,
            
            """
            ALTER TABLE study_sessions
            ADD CONSTRAINT check_duration
            CHECK (duration_minutes IS NULL OR duration_minutes >= 0)
            """,
            
            """
            ALTER TABLE study_sessions
            ADD CONSTRAINT check_performance
            CHECK (performance_score IS NULL OR (performance_score >= 0 AND performance_score <= 100))
            """,
        ]
        
        with engine.connect() as connection:
            for constraint_sql in constraints:
                try:
                    connection.execute(text(constraint_sql))
                    logger.info("Created constraint")
                except Exception as e:
                    logger.warning(f"Could not create constraint: {e}")
            connection.commit()
    
    @staticmethod
    def initialize_database(engine, base):
        """Complete database initialization"""
        try:
            logger.info("Starting database initialization...")
            
            # Create extensions
            DatabaseInitializer.create_extensions(engine)
            
            # Create tables
            DatabaseInitializer.create_tables(engine, base)
            
            # Create indexes
            DatabaseInitializer.create_indexes(engine)
            
            # Create constraints
            DatabaseInitializer.create_constraints(engine)
            
            logger.info("Database initialization completed successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise


class DatabaseHealthCheck:
    """Check database health and connectivity"""
    
    @staticmethod
    def check_connection(session: Session) -> bool:
        """Check if database is reachable"""
        try:
            session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    @staticmethod
    def get_table_sizes(session: Session) -> dict:
        """Get size of all tables"""
        try:
            query = text("""
                SELECT
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            
            result = session.execute(query)
            return {row[0]: row[1] for row in result}
        except Exception as e:
            logger.error(f"Error getting table sizes: {e}")
            return {}
    
    @staticmethod
    def get_database_stats(session: Session) -> dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Get database size
            size_query = text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            size_result = session.execute(size_query).scalar()
            stats['database_size'] = size_result
            
            # Get table count
            table_query = text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            table_count = session.execute(table_query).scalar()
            stats['table_count'] = table_count
            
            # Get index count
            index_query = text("""
                SELECT COUNT(*) FROM pg_indexes
                WHERE schemaname = 'public'
            """)
            index_count = session.execute(index_query).scalar()
            stats['index_count'] = index_count
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
