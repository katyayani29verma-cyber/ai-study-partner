#!/usr/bin/env python3
"""
Production Deployment Verification Script

This script verifies that the backend is ready for production deployment.
Run this before deploying to ensure all components are properly configured.

Usage:
    python verify_deployment.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class DeploymentVerifier:
    """Verify production deployment readiness."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.backend_dir = Path(__file__).parent
        self.project_dir = self.backend_dir.parent
        
    def print_header(self, text):
        """Print section header."""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}{text}{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    def print_check(self, name, passed, message=""):
        """Print check result."""
        status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
        print(f"  {status}  {name}")
        if message:
            print(f"         {message}")
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
    
    def print_warning(self, name, message=""):
        """Print warning."""
        print(f"  {YELLOW}[WARN]{RESET}  {name}")
        if message:
            print(f"         {message}")
        self.warnings += 1
    
    def check_file_exists(self, path, description):
        """Check if a file exists."""
        full_path = self.backend_dir / path if not path.startswith('/') else Path(path)
        exists = full_path.exists()
        self.print_check(f"{description}", exists, str(full_path))
        return exists
    
    def check_file_contains(self, path, text, description):
        """Check if a file contains specific text."""
        full_path = self.backend_dir / path
        if not full_path.exists():
            self.print_check(description, False, f"File not found: {full_path}")
            return False
        
        try:
            content = full_path.read_text()
            contains = text in content
            self.print_check(description, contains)
            return contains
        except Exception as e:
            self.print_check(description, False, str(e))
            return False
    
    def verify_code_structure(self):
        """Verify code structure and files."""
        self.print_header("Code Structure Verification")
        
        # Check main files
        self.check_file_exists("api/main.py", "Main FastAPI application")
        self.check_file_exists("api/routes/auth.py", "Authentication routes")
        self.check_file_exists("api/routes/cognitive_load.py", "Cognitive load routes")
        self.check_file_exists("api/routes/content_chunking.py", "Content chunking routes")
        self.check_file_exists("api/routes/revision.py", "Revision engine routes")
        self.check_file_exists("api/routes/learning_path.py", "Learning path routes")
        
        # Check database files
        self.check_file_exists("database/models.py", "Database models")
        self.check_file_exists("database/config.py", "Database configuration")
        self.check_file_exists("database/operations.py", "Database operations")
        
        # Check security files
        self.check_file_exists("security/config.py", "Security configuration")
        self.check_file_exists("security/auth.py", "Authentication module")
        self.check_file_exists("security/rbac.py", "RBAC module")
        self.check_file_exists("security/encryption.py", "Encryption module")
    
    def verify_dependencies(self):
        """Verify dependencies are listed."""
        self.print_header("Dependencies Verification")
        
        self.check_file_exists("requirements.txt", "Root requirements.txt")
        self.check_file_exists("database/requirements.txt", "Database requirements.txt")
        self.check_file_exists("security/requirements.txt", "Security requirements.txt")
        
        # Check key dependencies
        self.check_file_contains("requirements.txt", "fastapi", "FastAPI in requirements")
        self.check_file_contains("requirements.txt", "uvicorn", "Uvicorn in requirements")
        self.check_file_contains("requirements.txt", "gunicorn", "Gunicorn in requirements")
        self.check_file_contains("requirements.txt", "sqlalchemy", "SQLAlchemy in requirements")
        self.check_file_contains("requirements.txt", "alembic", "Alembic in requirements")
        self.check_file_contains("requirements.txt", "psycopg2", "psycopg2 in requirements")
    
    def verify_database(self):
        """Verify database configuration."""
        self.print_header("Database Verification")
        
        self.check_file_exists("alembic.ini", "Alembic configuration")
        self.check_file_exists("alembic/env.py", "Alembic environment")
        self.check_file_exists("alembic/versions/001_initial_schema.py", "Initial migration")
        
        # Check migration file has tables
        self.check_file_contains(
            "alembic/versions/001_initial_schema.py",
            "def upgrade",
            "Migration has upgrade function"
        )
        self.check_file_contains(
            "alembic/versions/001_initial_schema.py",
            "def downgrade",
            "Migration has downgrade function"
        )
        
        # Check for key tables in migration
        tables = [
            "users", "study_materials", "study_sessions",
            "cognitive_load_metrics", "content_chunks", "flashcards",
            "learning_paths", "path_modules", "performance_metrics",
            "revision_items", "revision_reviews", "revision_schedules",
            "chunk_interactions", "adaptive_recommendations"
        ]
        
        for table in tables:
            self.check_file_contains(
                "alembic/versions/001_initial_schema.py",
                f"'{table}'",
                f"Migration creates '{table}' table"
            )
    
    def verify_security(self):
        """Verify security configuration."""
        self.print_header("Security Verification")
        
        # Check security modules
        self.check_file_contains("api/main.py", "SecurityHeadersMiddleware", "Security headers middleware")
        self.check_file_contains("api/main.py", "CORSMiddleware", "CORS middleware")
        self.check_file_contains("api/main.py", "ALLOWED_ORIGINS", "CORS origins configured")
        
        # Check authentication (OAuth2 in auth.py, get_current_user in routes/auth.py)
        self.check_file_contains("api/routes/auth.py", "OAuth2PasswordBearer", "OAuth2 authentication")
        self.check_file_contains("api/routes/auth.py", "def get_current_user", "Current user dependency")
        
        # Check RBAC (UserRole enum instead of class Role)
        self.check_file_contains("security/rbac.py", "class UserRole", "RBAC roles defined")
        self.check_file_contains("security/rbac.py", "class RBACManager", "RBAC manager implemented")
        
        # Check encryption
        self.check_file_contains("security/encryption.py", "def encrypt", "Encryption function")
        self.check_file_contains("security/encryption.py", "def decrypt", "Decryption function")
    
    def verify_docker(self):
        """Verify Docker configuration."""
        self.print_header("Docker Configuration Verification")
        
        self.check_file_exists("Dockerfile.prod", "Production Dockerfile")
        self.check_file_exists("docker-compose.prod.yml", "Production Docker Compose")
        self.check_file_exists("nginx.conf", "Nginx configuration")
        
        # Check Dockerfile content
        self.check_file_contains("Dockerfile.prod", "python:3.11", "Python 3.11 base image")
        self.check_file_contains("Dockerfile.prod", "gunicorn", "Gunicorn in Dockerfile")
        self.check_file_contains("Dockerfile.prod", "HEALTHCHECK", "Health check in Dockerfile")
        
        # Check Docker Compose content
        self.check_file_contains("docker-compose.prod.yml", "postgres", "PostgreSQL service")
        self.check_file_contains("docker-compose.prod.yml", "redis", "Redis service")
        self.check_file_contains("docker-compose.prod.yml", "nginx", "Nginx service")
    
    def verify_environment(self):
        """Verify environment configuration."""
        self.print_header("Environment Configuration Verification")
        
        self.check_file_exists(".env.production.example", "Production environment template")
        
        # Check template content
        self.check_file_contains(".env.production.example", "SECRET_KEY", "SECRET_KEY in template")
        self.check_file_contains(".env.production.example", "MASTER_KEY", "MASTER_KEY in template")
        self.check_file_contains(".env.production.example", "DATABASE_URL", "DATABASE_URL in template")
        self.check_file_contains(".env.production.example", "ALLOWED_ORIGINS", "ALLOWED_ORIGINS in template")
        
        # Check that .env.production is NOT in repo
        env_prod = self.backend_dir / ".env.production"
        if env_prod.exists():
            self.print_warning(
                ".env.production exists in repo",
                "This file should NOT be committed. Add to .gitignore"
            )
        else:
            self.print_check(".env.production not in repo (good)", True)
    
    def verify_endpoints(self):
        """Verify API endpoints are registered."""
        self.print_header("API Endpoints Verification")
        
        # Check routers are registered
        self.check_file_contains("api/main.py", "auth.router", "Auth router registered")
        self.check_file_contains("api/main.py", "cognitive_load.router", "Cognitive load router registered")
        self.check_file_contains("api/main.py", "content_chunking.router", "Content chunking router registered")
        self.check_file_contains("api/main.py", "revision.router", "Revision router registered")
        self.check_file_contains("api/main.py", "learning_path.router", "Learning path router registered")
        
        # Check system endpoints
        self.check_file_contains("api/main.py", "/health", "Health endpoint")
        self.check_file_contains("api/main.py", "def root", "Root endpoint")
    
    def verify_documentation(self):
        """Verify documentation exists."""
        self.print_header("Documentation Verification")
        
        docs = [
            ("../QUICK_DEPLOYMENT_REFERENCE.md", "Quick deployment reference"),
            ("../PRODUCTION_DEPLOYMENT_CHECKLIST.md", "Production deployment checklist"),
            ("DEPLOYMENT_GUIDE.md", "Backend deployment guide"),
            ("PRODUCTION_READINESS.md", "Production readiness guide"),
            ("../BACKEND_ONLINE_DEPLOYMENT.md", "Online deployment guide"),
        ]
        
        for path, desc in docs:
            self.check_file_exists(path, desc)
    
    def verify_tests(self):
        """Verify test suite."""
        self.print_header("Test Suite Verification")
        
        self.check_file_exists("test_all.py", "Comprehensive test suite")
        self.check_file_contains("test_all.py", "def test_", "Test functions defined")
    
    def print_summary(self):
        """Print verification summary."""
        self.print_header("Verification Summary")
        
        total = self.checks_passed + self.checks_failed
        
        print(f"{GREEN}[PASS] Passed: {self.checks_passed}/{total}{RESET}")
        if self.checks_failed > 0:
            print(f"{RED}[FAIL] Failed: {self.checks_failed}/{total}{RESET}")
        if self.warnings > 0:
            print(f"{YELLOW}[WARN] Warnings: {self.warnings}{RESET}")
        
        print()
        
        if self.checks_failed == 0:
            print(f"{GREEN}{BOLD}[SUCCESS] ALL CHECKS PASSED - READY FOR DEPLOYMENT{RESET}")
            return True
        else:
            print(f"{RED}{BOLD}[ERROR] SOME CHECKS FAILED - FIX ISSUES BEFORE DEPLOYMENT{RESET}")
            return False
    
    def run(self):
        """Run all verifications."""
        print(f"\n{BOLD}{BLUE}Production Deployment Verification{RESET}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.verify_code_structure()
        self.verify_dependencies()
        self.verify_database()
        self.verify_security()
        self.verify_docker()
        self.verify_environment()
        self.verify_endpoints()
        self.verify_documentation()
        self.verify_tests()
        
        success = self.print_summary()
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return 0 if success else 1


if __name__ == "__main__":
    verifier = DeploymentVerifier()
    exit_code = verifier.run()
    sys.exit(exit_code)
