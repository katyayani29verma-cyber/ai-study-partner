# AI Study Partner - Backend API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-13+-336791.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/redis-6+-DC382D.svg)](https://redis.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](#)

A production-ready FastAPI backend for an adaptive learning platform that uses AI to optimize student learning through cognitive load management, content chunking, spaced repetition, and personalized learning paths.

## 📖 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Core Features](#-core-features)
- [Security](#-security)
- [Database](#-database)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Development](#-development)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Support](#-support)

## 🚀 Quick Start

### Prerequisites

**System Requirements:**
- Python 3.11 or higher
- PostgreSQL 13 or higher
- Redis 6 or higher
- 2GB RAM minimum
- 10GB disk space minimum

**Installation Methods:**
- **Ubuntu/Debian:** `sudo apt-get install python3.11 postgresql redis-server`
- **macOS:** `brew install python@3.11 postgresql redis`
- **Windows:** Download from [python.org](https://www.python.org/), [postgresql.org](https://www.postgresql.org/), [redis.io](https://redis.io/)

### Installation (5 minutes)

1. **Clone and navigate to backend**
```bash
git clone https://github.com/your-org/ai-study-partner.git
cd ai-study-partner/backend
```

2. **Create virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env .env.local
# Edit .env.local with your settings (see Configuration section)
```

5. **Initialize database**
```bash
alembic upgrade head
```

6. **Run development server**
```bash
python api/main.py
```

The API will be available at:
- **API:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### Quick Start with Docker (Alternative)

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## ✨ Features

### Core Capabilities
- **Adaptive Learning** - AI-powered personalized learning paths
- **Cognitive Load Management** - Real-time monitoring and optimization
- **Content Chunking** - Intelligent content segmentation
- **Spaced Repetition** - SM-2 algorithm for optimal review scheduling
- **Performance Analytics** - Comprehensive learning analytics
- **Multi-user Support** - Scalable for thousands of concurrent users

### Technical Features
- **Production-Ready** - Enterprise-grade security and reliability
- **Async/Await** - High-performance async operations
- **RESTful API** - Clean, intuitive API design
- **Real-time Updates** - WebSocket support for live updates
- **Caching** - Redis-based caching for performance
- **Rate Limiting** - Built-in rate limiting and throttling
- **Audit Logging** - Complete audit trail of all operations
- **GDPR Compliant** - Data privacy and compliance features

## 📋 Project Structure

```
backend/
├── api/                          # FastAPI application layer
│   ├── main.py                   # App initialization, middleware, routes
│   └── routes/                   # API endpoint modules
│       ├── auth.py               # Authentication & authorization
│       ├── cognitive_load.py     # Cognitive load tracking
│       ├── content_chunking.py   # Content management
│       ├── revision.py           # Spaced repetition engine
│       └── learning_path.py      # Adaptive learning paths
│
├── database/                     # Data access layer
│   ├── config.py                 # Database configuration & pooling
│   ├── models.py                 # SQLAlchemy ORM models (18+ tables)
│   ├── session.py                # Session management & lifecycle
│   ├── operations.py             # Database operation classes
│   ├── init.py                   # Database initialization
│   └── README.md                 # Database documentation
│
├── security/                     # Security & authentication layer
│   ├── auth.py                   # JWT tokens & password hashing
│   ├── rbac.py                   # Role-based access control
│   ├── validation.py             # Input validation & sanitization
│   ├── encryption.py             # Field & file encryption
│   ├── csrf.py                   # CSRF token management
│   ├── rate_limit.py             # Request rate limiting
│   ├── audit.py                  # Security event logging
│   ├── privacy.py                # GDPR compliance
│   ├── headers.py                # Security headers middleware
│   ├── incident_response.py      # Incident detection & response
│   ├── config.py                 # Security configuration
│   ├── examples.py               # Usage examples
│   ├── test_security.py          # Security unit tests
│   └── README.md                 # Security documentation
│
├── ai_integration/               # AI module integration layer
│   └── adapter.py                # Adapter for AI modules
│
├── performance/                  # Performance optimization
│   ├── caching.py                # Redis caching strategies
│   ├── background_tasks.py       # Async task processing
│   ├── monitoring.py             # Performance metrics
│   └── scaling.py                # Horizontal scaling guidelines
│
├── alembic/                      # Database migration management
│   ├── env.py                    # Migration environment setup
│   ├── script.py.mako            # Migration template
│   └── versions/                 # Migration files
│
├── logs/                         # Application logs
│   └── nginx/                    # Nginx reverse proxy logs
│
├── ssl/                          # SSL/TLS certificates & keys
│   ├── certs/                    # Public certificates
│   ├── private/                  # Private keys
│   ├── scripts/                  # Certificate management scripts
│   └── configs/                  # SSL configuration files
│
├── requirements.txt              # Python package dependencies
├── alembic.ini                   # Alembic migration configuration
├── docker-compose.prod.yml       # Production Docker Compose setup
├── docker-compose.staging.yml    # Staging Docker Compose setup
├── Dockerfile.prod               # Production Docker image
├── nginx.conf                    # Nginx reverse proxy configuration
├── start-api.ps1                 # PowerShell startup script
├── start-api.bat                 # Batch startup script
├── test_all.py                   # Comprehensive test suite
├── test_database_ops.py          # Database operation tests
├── test_security_features.py     # Security feature tests
├── verify_deployment.py          # Deployment verification script
└── README.md                     # This file
```

### Directory Descriptions

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `api/` | HTTP API layer | `main.py`, `routes/*.py` |
| `database/` | Data persistence | `models.py`, `operations.py` |
| `security/` | Authentication & protection | `auth.py`, `rbac.py`, `validation.py` |
| `ai_integration/` | AI module integration | `adapter.py` |
| `performance/` | Optimization & monitoring | `caching.py`, `monitoring.py` |
| `alembic/` | Database migrations | `versions/*.py` |
| `ssl/` | SSL/TLS certificates | `certs/`, `private/` |

## 🔑 Core Features

### 1. Authentication & Authorization

Secure user authentication with JWT tokens and role-based access control.

**Features:**
- JWT-based authentication (30-minute expiration)
- Bcrypt password hashing (cost 12)
- Role-based access control (RBAC)
- Permission-based authorization
- Secure token refresh mechanism
- Multi-factor authentication ready

**Endpoints:**
```
POST   /auth/register          # User registration
POST   /auth/login             # User login
POST   /auth/refresh           # Token refresh
GET    /auth/me                # Current user info
POST   /auth/logout            # User logout
```

**Example:**
```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

### 2. Cognitive Load Management

Real-time monitoring and optimization of student cognitive load during study sessions.

**Features:**
- Real-time cognitive load tracking
- Multi-dimensional load assessment
- Adaptive load thresholds
- Historical trend analysis
- Overload prevention
- Performance optimization

**Endpoints:**
```
POST   /cognitive-load/record           # Record metrics
GET    /cognitive-load/current          # Get current load
GET    /cognitive-load/history          # Get historical data
GET    /cognitive-load/analytics        # Get analytics
```

**Metrics Tracked:**
- Mental effort (0-100)
- Working memory load (0-100)
- Attention level (0-100)
- Stress level (0-100)
- Overall cognitive load (calculated)

**Example:**
```bash
curl -X POST http://localhost:8000/cognitive-load/record \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "mental_effort": 65,
    "working_memory_load": 70,
    "attention_level": 75,
    "stress_level": 60
  }'
```

### 3. Content Chunking

Intelligent content segmentation with difficulty assessment and interaction tracking.

**Features:**
- Automatic content segmentation
- Difficulty assessment (1-5 scale)
- Interaction tracking
- Comprehension scoring
- Time tracking
- Analytics per chunk

**Endpoints:**
```
POST   /content/chunks                  # Create chunks
POST   /content/chunks/{id}/interact    # Record interaction
GET    /content/chunks/{id}/analytics   # Get analytics
GET    /content/chunks                  # List chunks
```

**Features:**
- Automatic content segmentation
- Difficulty assessment
- Interaction tracking
- Comprehension scoring

**Example:**
```bash
curl -X POST http://localhost:8000/content/chunks \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "material_id": 1,
    "chunks": [
      {
        "title": "Introduction",
        "content": "...",
        "difficulty": 2,
        "estimated_time": 300
      }
    ]
  }'
```

### 4. Spaced Repetition (Revision Engine)

SM-2 algorithm implementation for optimal review scheduling based on performance.

**Features:**
- SM-2 spaced repetition algorithm
- Adaptive scheduling
- Quality-based intervals
- Performance tracking
- Difficulty adjustment
- Retention optimization

**Endpoints:**
```
POST   /revision/items                  # Create revision item
GET    /revision/due                    # Get due items
POST   /revision/review                 # Record review
GET    /revision/schedule               # Get schedule
```

**Algorithm Details:**
- Quality 0-2: Resets interval to 1 day
- Quality 3-4: Increases interval
- Quality 5: Increases interval more aggressively
- Ease factor adjusts based on performance

**Example:**
```bash
curl -X POST http://localhost:8000/revision/review \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "revision_item_id": 1,
    "quality": 4,
    "time_taken": 30,
    "confidence": 85
  }'
```

### 5. Adaptive Learning Paths

AI-powered personalized learning paths that adapt based on student performance.

**Features:**
- Personalized path generation
- Performance-based adaptation
- Module recommendations
- Progress tracking
- Difficulty adjustment
- Learning style adaptation

**Endpoints:**
```
POST   /learning-path/create            # Create path
GET    /learning-path/{id}              # Get path details
POST   /learning-path/recommend         # Get recommendations
GET    /learning-path                   # List paths
PUT    /learning-path/{id}              # Update path
```

**Example:**
```bash
curl -X POST http://localhost:8000/learning-path/create \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Biology Fundamentals",
    "subject": "Biology",
    "goal": "Master basic concepts",
    "modules": [
      {
        "title": "Cell Structure",
        "difficulty": 2,
        "order": 1
      }
    ]
  }'
```

## 🔐 Security

### Security Features

**Authentication & Authorization:**
- ✅ JWT tokens with expiration
- ✅ Bcrypt password hashing (cost 12)
- ✅ Role-based access control (RBAC)
- ✅ Permission-based authorization
- ✅ Secure token refresh
- ✅ Multi-factor authentication ready

**Data Protection:**
- ✅ HTTPS/TLS encryption
- ✅ Database field encryption
- ✅ Secure password storage
- ✅ Input validation & sanitization
- ✅ SQL injection prevention
- ✅ XSS protection

**API Security:**
- ✅ CORS protection
- ✅ CSRF tokens
- ✅ Rate limiting
- ✅ Request throttling
- ✅ Security headers
- ✅ Audit logging

**Compliance:**
- ✅ GDPR compliance
- ✅ Data export functionality
- ✅ Data deletion support
- ✅ Consent management
- ✅ Privacy controls
- ✅ Audit trails

### Security Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `auth.py` | Authentication | JWT, password hashing, token management |
| `rbac.py` | Authorization | Role-based access control, permissions |
| `validation.py` | Input validation | Sanitization, injection prevention |
| `encryption.py` | Data encryption | Field encryption, secure storage |
| `csrf.py` | CSRF protection | Token generation and validation |
| `rate_limit.py` | Rate limiting | Request throttling, quota management |
| `audit.py` | Audit logging | Security event logging, tracking |
| `privacy.py` | Privacy/GDPR | Data export, deletion, compliance |
| `headers.py` | Security headers | HTTP security headers middleware |
| `incident_response.py` | Incident response | Detection and containment |

### Security Best Practices

**Do's:**
- ✅ Store secrets in environment variables
- ✅ Use HTTPS/TLS in production
- ✅ Validate all user inputs
- ✅ Hash passwords with bcrypt
- ✅ Use JWT for authentication
- ✅ Enable rate limiting
- ✅ Log security events
- ✅ Keep dependencies updated

**Don'ts:**
- ❌ Hardcode secrets in code
- ❌ Use HTTP in production
- ❌ Trust user input
- ❌ Store plain text passwords
- ❌ Use weak encryption
- ❌ Disable security headers
- ❌ Ignore security warnings
- ❌ Use outdated dependencies

### Security Documentation

For detailed security information, see:
- [security/README.md](security/README.md) - Security module documentation
- [ssl/README.md](ssl/README.md) - SSL/TLS setup and management
- [SECURITY.md](../SECURITY.md) - Comprehensive security guide
- [SECURITY_CHECKLIST.md](../SECURITY_CHECKLIST.md) - Implementation checklist

## 📊 Database

### Schema Overview

The database includes **18+ tables** organized by feature:

**Core Tables (8):**
| Table | Purpose | Key Fields |
|-------|---------|-----------|
| `users` | User accounts | id, email, password_hash, role |
| `study_materials` | Study content | id, title, content, subject |
| `study_sessions` | Study sessions | id, user_id, start_time, end_time |
| `flashcards` | Flashcard items | id, question, answer, difficulty |
| `knowledge_gaps` | Knowledge gaps | id, user_id, topic, severity |
| `curriculum_data` | Curriculum info | id, subject, level, content |
| `socratic_sessions` | Socratic dialogues | id, user_id, topic, transcript |
| `audit_logs` | Security audit logs | id, user_id, action, timestamp |

**Feature Tables (10):**

*Cognitive Load (2 tables):*
- `cognitive_load_metrics` - Real-time measurements
- `cognitive_load_thresholds` - User preferences

*Content Chunking (2 tables):*
- `content_chunks` - Chunked materials
- `chunk_interactions` - User interactions

*Revision Engine (3 tables):*
- `revision_items` - Items to review
- `revision_reviews` - Review sessions
- `revision_schedule` - Daily schedules

*Adaptive Learning (4 tables):*
- `learning_paths` - Learning journeys
- `path_modules` - Path modules
- `adaptive_recommendations` - Suggestions
- `performance_metrics` - Performance data

### Database Features

**Performance:**
- 30+ indexes for fast queries
- Connection pooling (20 connections)
- Query optimization
- Automatic cache invalidation

**Reliability:**
- ACID transactions
- Foreign key constraints
- Data integrity checks
- Backup support

**Scalability:**
- Horizontal scaling ready
- Read replicas support
- Partitioning support
- Archive tables

### Database Operations

Quick example:
```python
from database.operations import CognitiveLoadOperations
from database.session import SessionLocal

db = SessionLocal()
try:
    # Record cognitive load
    metric = CognitiveLoadOperations.record_cognitive_load(
        db, user_id=1,
        mental_effort=65,
        working_memory_load=70,
        attention_level=75,
        stress_level=60
    )
    
    # Get current load
    current = CognitiveLoadOperations.get_current_cognitive_load(db, user_id=1)
    
    # Get history
    history = CognitiveLoadOperations.get_cognitive_load_history(db, user_id=1, days=7)
    
    db.commit()
except Exception as e:
    db.rollback()
    raise
finally:
    db.close()
```

### Database Configuration

Edit `database/config.py` for connection pooling:
```python
POOL_SIZE = 20          # Connections to keep
MAX_OVERFLOW = 10       # Additional connections
POOL_RECYCLE = 3600     # Recycle after 1 hour
POOL_PRE_PING = True    # Test connections before use
```

### Database Documentation

For complete database documentation, see:
- [database/README.md](database/README.md) - Database module documentation
- [database/FEATURE_ORIENTED_SCHEMA.md](database/FEATURE_ORIENTED_SCHEMA.md) - Schema details
- [database/FEATURE_GUIDE.md](database/FEATURE_GUIDE.md) - Feature usage guide

## 🚀 Deployment

### Development Environment

**Local Development:**
```bash
# Start development server
python api/main.py

# With auto-reload
pip install watchfiles
uvicorn api.main:app --reload

# On specific port
python api/main.py --port 8001
```

**With Docker:**
```bash
# Start PostgreSQL and Redis
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
docker run -d --name redis -p 6379:6379 redis:7

# Run migrations
alembic upgrade head

# Start API
python api/main.py
```

### Staging Environment

**Docker Compose:**
```bash
# Start all services
docker-compose -f docker-compose.staging.yml up -d

# Run migrations
docker-compose -f docker-compose.staging.yml exec api alembic upgrade head

# View logs
docker-compose -f docker-compose.staging.yml logs -f api

# Stop services
docker-compose -f docker-compose.staging.yml down
```

### Production Environment

**Docker Compose:**
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

**Manual Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start with Gunicorn (4 workers)
gunicorn -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  api.main:app
```

### Deployment Checklist

**Pre-Deployment:**
- [ ] All tests passing
- [ ] Code reviewed
- [ ] Environment variables configured
- [ ] Database backups created
- [ ] SSL certificate valid
- [ ] Nginx configuration tested
- [ ] Monitoring configured
- [ ] Rate limiting configured
- [ ] Security headers configured
- [ ] CORS origins configured

**Post-Deployment:**
- [ ] API health check passing
- [ ] Database connectivity verified
- [ ] Redis connectivity verified
- [ ] SSL certificate working
- [ ] CORS working correctly
- [ ] Authentication working
- [ ] Rate limiting working
- [ ] Logging working
- [ ] Monitoring active
- [ ] Backups running

### Deployment Documentation

For complete deployment information, see:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [ssl/README.md](ssl/README.md) - SSL/TLS setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Deployment issues

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# ============================================
# SECURITY & SECRETS
# ============================================
SECRET_KEY=<generate with: openssl rand -hex 32>
MASTER_KEY=<generate with: openssl rand -hex 32>

# ============================================
# DATABASE
# ============================================
DATABASE_URL=postgresql://user:password@localhost:5432/study_partner
POOL_SIZE=20
MAX_OVERFLOW=10
POOL_RECYCLE=3600

# ============================================
# REDIS (Caching & Rate Limiting)
# ============================================
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=<optional>

# ============================================
# CORS & FRONTEND
# ============================================
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000,https://yourdomain.com

# ============================================
# ENVIRONMENT
# ============================================
ENVIRONMENT=development  # or production, staging

# ============================================
# API SETTINGS
# ============================================
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ============================================
# LOGGING
# ============================================
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============================================
# FILE UPLOADS
# ============================================
MAX_UPLOAD_SIZE=52428800  # 50MB in bytes

# ============================================
# SECURITY HEADERS
# ============================================
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=true
```

### Generate Secure Keys

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate MASTER_KEY
openssl rand -hex 32

# Or use Python
python -c "import secrets; print(secrets.token_hex(32))"
```

### Database Configuration

Edit `database/config.py`:

```python
# Connection pooling settings
POOL_SIZE = 20              # Connections to keep in pool
MAX_OVERFLOW = 10           # Additional connections allowed
POOL_RECYCLE = 3600         # Recycle connections after 1 hour
POOL_PRE_PING = True        # Test connections before use

# Query settings
ECHO_SQL = False            # Log all SQL queries (development only)
ECHO_POOL = False           # Log pool events (development only)
```

### API Configuration

Edit `api/main.py`:

```python
# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://yourdomain.com"
]

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 60  # seconds

# Token settings
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
```

### Security Configuration

Edit `security/config.py`:

```python
# Password hashing
PASSWORD_HASH_ALGORITHM = "bcrypt"
PASSWORD_HASH_COST = 12

# JWT settings
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 1800  # 30 minutes

# Encryption
ENCRYPTION_ALGORITHM = "AES-256-GCM"
```

### Docker Environment

Create `.env` for Docker Compose:

```env
# Database
DB_PASSWORD=secure_password_here
DATABASE_URL=postgresql://study_user:secure_password_here@db:5432/study_partner

# Redis
REDIS_PASSWORD=redis_password_here
REDIS_URL=redis://:redis_password_here@redis:6379

# Security
SECRET_KEY=<generate with: openssl rand -hex 32>
MASTER_KEY=<generate with: openssl rand -hex 32>

# Environment
ENVIRONMENT=production

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# API
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Logging
LOG_LEVEL=INFO

# File Uploads
MAX_UPLOAD_SIZE=52428800
```

### Configuration Documentation

For complete configuration information, see:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment configuration
- [security/README.md](security/README.md) - Security configuration
- [database/README.md](database/README.md) - Database configuration

## 📚 API Documentation

### Interactive Documentation

The API provides interactive documentation at:

| Tool | URL | Purpose |
|------|-----|---------|
| Swagger UI | http://localhost:8000/docs | Interactive API testing |
| ReDoc | http://localhost:8000/redoc | Beautiful API documentation |
| OpenAPI Schema | http://localhost:8000/openapi.json | Machine-readable schema |

### Health Check

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "service": "AI Study Partner API",
  "version": "1.0.0"
}
```

### API Endpoints Summary

**Authentication:**
```
POST   /auth/register          # Register new user
POST   /auth/login             # Login user
POST   /auth/refresh           # Refresh token
GET    /auth/me                # Get current user
POST   /auth/logout            # Logout user
```

**Cognitive Load:**
```
POST   /cognitive-load/record           # Record metrics
GET    /cognitive-load/current          # Get current load
GET    /cognitive-load/history          # Get historical data
GET    /cognitive-load/analytics        # Get analytics
```

**Content Chunking:**
```
POST   /content/chunks                  # Create chunks
POST   /content/chunks/{id}/interact    # Record interaction
GET    /content/chunks/{id}/analytics   # Get analytics
GET    /content/chunks                  # List chunks
```

**Revision Engine:**
```
POST   /revision/items                  # Create revision item
GET    /revision/due                    # Get due items
POST   /revision/review                 # Record review
GET    /revision/schedule               # Get schedule
```

**Learning Paths:**
```
POST   /learning-path/create            # Create path
GET    /learning-path/{id}              # Get path details
POST   /learning-path/recommend         # Get recommendations
GET    /learning-path                   # List paths
PUT    /learning-path/{id}              # Update path
```

### Example Requests

**Register User:**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Record Cognitive Load:**
```bash
curl -X POST http://localhost:8000/cognitive-load/record \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "mental_effort": 65,
    "working_memory_load": 70,
    "attention_level": 75,
    "stress_level": 60
  }'
```

**Get Current Cognitive Load:**
```bash
curl -X GET http://localhost:8000/cognitive-load/current \
  -H "Authorization: Bearer <access_token>"
```

**Create Content Chunks:**
```bash
curl -X POST http://localhost:8000/content/chunks \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "material_id": 1,
    "chunks": [
      {
        "title": "Introduction to Photosynthesis",
        "content": "Photosynthesis is the process...",
        "difficulty": 2,
        "estimated_time": 300,
        "order": 1
      }
    ]
  }'
```

**Record Chunk Interaction:**
```bash
curl -X POST http://localhost:8000/content/chunks/1/interact \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "time_spent": 450,
    "completion_percentage": 100,
    "comprehension_score": 85,
    "cognitive_load": 65
  }'
```

**Create Revision Item:**
```bash
curl -X POST http://localhost:8000/revision/items \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "item_type": "flashcard",
    "item_id": 1,
    "subject": "Biology",
    "difficulty": 3
  }'
```

**Get Due Items:**
```bash
curl -X GET http://localhost:8000/revision/due \
  -H "Authorization: Bearer <access_token>"
```

**Record Review:**
```bash
curl -X POST http://localhost:8000/revision/review \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "revision_item_id": 1,
    "quality": 4,
    "time_taken": 30,
    "confidence": 85
  }'
```

**Create Learning Path:**
```bash
curl -X POST http://localhost:8000/learning-path/create \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Biology Fundamentals",
    "subject": "Biology",
    "goal": "Master basic concepts",
    "modules": [
      {
        "title": "Cell Structure",
        "description": "Learn about cell components",
        "order": 1,
        "difficulty": 2
      }
    ]
  }'
```

### Response Format

**Success Response:**
```json
{
  "data": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2026-03-10T10:30:00Z"
  },
  "status": "success",
  "timestamp": "2026-03-10T10:30:00Z"
}
```

**Error Response:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request format",
    "details": {}
  },
  "status": "error",
  "timestamp": "2026-03-10T10:30:00Z"
}
```

### Rate Limiting

Rate limits are applied per user and endpoint:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/auth/login` | 5 requests | 1 minute |
| `/auth/register` | 3 requests | 1 hour |
| `/cognitive-load/*` | 10 requests | 1 second |
| `/content/*` | 10 requests | 1 second |
| `/revision/*` | 10 requests | 1 second |
| `/learning-path/*` | 10 requests | 1 second |

Rate limit headers in response:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1678450200
```

### API Documentation

For complete API documentation, see:
- [API_ENDPOINTS.md](API_ENDPOINTS.md) - Complete endpoint reference
- http://localhost:8000/docs - Interactive Swagger UI
- http://localhost:8000/redoc - ReDoc documentation

## 🧪 Testing

### Running Tests

**All Tests:**
```bash
pytest
```

**Specific Test File:**
```bash
pytest backend/test_security_features.py -v
```

**Specific Test:**
```bash
pytest backend/test_security_features.py::test_login_success -v
```

**With Coverage:**
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

**With Markers:**
```bash
pytest -m "not slow"
```

**Parallel Execution:**
```bash
pytest -n auto
```

### Test Files

| File | Purpose | Coverage |
|------|---------|----------|
| `test_security_features.py` | Security module tests | 30+ tests |
| `test_database_ops.py` | Database operation tests | 20+ tests |
| `test_all.py` | Comprehensive test suite | 50+ tests |
| `security/test_security.py` | Security-specific tests | 30+ tests |

### Test Coverage Goals

- **Overall:** 80% minimum
- **Security modules:** 100%
- **API endpoints:** 90%
- **Database operations:** 85%

### Writing Tests

```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

class TestAuthEndpoints:
    """Authentication endpoint tests."""
    
    def test_register_success(self):
        """Test successful user registration."""
        response = client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "SecurePass123!",
            "full_name": "Test User"
        })
        assert response.status_code == 201
        assert response.json()["email"] == "test@example.com"
    
    def test_register_duplicate_email(self):
        """Test duplicate email rejection."""
        # Create first user
        client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "SecurePass123!",
            "full_name": "Test User"
        })
        
        # Try to create duplicate
        response = client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "AnotherPass123!",
            "full_name": "Another User"
        })
        assert response.status_code == 409
```

### Testing Documentation

For complete testing information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Testing requirements
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Testing commands

## 🛠️ Development

### Development Workflow

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Update models in `database/models.py`
- Add operations in `database/operations.py`
- Create API routes in `api/routes/`
- Add security checks in `security/`

3. **Create Migration**
```bash
alembic revision --autogenerate -m "Add your feature"
```

4. **Test Changes**
```bash
pytest
black .
flake8 .
mypy .
```

5. **Commit and Push**
```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

### Code Quality

**Format Code:**
```bash
black .
```

**Lint Code:**
```bash
flake8 .
```

**Type Check:**
```bash
mypy .
```

**All Checks:**
```bash
black . && flake8 . && mypy . && pytest
```

### Development Tools

**Useful Commands:**
```bash
# Start development server with auto-reload
uvicorn api.main:app --reload

# Run specific test
pytest backend/test_file.py::test_function -v

# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View database
psql -U postgres -h localhost -d study_partner

# Check Redis
redis-cli ping
```

### Development Documentation

For complete development information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands

## 📖 Documentation

### Main Documentation Files

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [README.md](README.md) | Project overview | 15 min |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Fast lookup guide | 10 min |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide | 30 min |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow | 20 min |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Problem solving | 20 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design | 25 min |
| [API_ENDPOINTS.md](API_ENDPOINTS.md) | API reference | 25 min |
| [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) | Documentation index | 10 min |

### Module Documentation

**Database:**
- [database/README.md](database/README.md) - Database module documentation
- [database/FEATURE_ORIENTED_SCHEMA.md](database/FEATURE_ORIENTED_SCHEMA.md) - Schema details
- [database/FEATURE_GUIDE.md](database/FEATURE_GUIDE.md) - Feature usage guide

**Security:**
- [security/README.md](security/README.md) - Security module documentation
- [security/QUICK_START.md](security/QUICK_START.md) - Security quick start
- [security/DEVELOPER_REFERENCE.md](security/DEVELOPER_REFERENCE.md) - Developer reference

**SSL/TLS:**
- [ssl/README.md](ssl/README.md) - SSL/TLS setup and management
- [ssl/configs/nginx-ssl.conf](ssl/configs/nginx-ssl.conf) - Nginx SSL configuration

### Project Documentation

**Root Level:**
- [../SECURITY.md](../SECURITY.md) - Comprehensive security guide
- [../SECURITY_CHECKLIST.md](../SECURITY_CHECKLIST.md) - Implementation checklist
- [../SECURITY_IMPLEMENTATION_SUMMARY.md](../SECURITY_IMPLEMENTATION_SUMMARY.md) - What was implemented

### Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

### Documentation Quick Links

**Getting Started:**
- [Quick Start](#-quick-start) - 5-minute setup
- [Installation](#installation-5-minutes) - Detailed installation
- [Configuration](#-configuration) - Environment setup

**Understanding the System:**
- [Features](#-features) - Core capabilities
- [Project Structure](#-project-structure) - Directory layout
- [Core Features](#-core-features) - Feature details
- [Architecture](ARCHITECTURE.md) - System design

**Using the API:**
- [API Documentation](#-api-documentation) - API overview
- [API_ENDPOINTS.md](API_ENDPOINTS.md) - Complete endpoint reference
- [Example Requests](#example-requests) - Code examples

**Development:**
- [Development](#-development) - Development workflow
- [Testing](#-testing) - Testing guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines

**Deployment:**
- [Deployment](#-deployment) - Deployment options
- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [ssl/README.md](ssl/README.md) - SSL/TLS setup

**Troubleshooting:**
- [Troubleshooting](#-troubleshooting) - Common issues
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed solutions

## 🆘 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python api/main.py --port 8001
```

**Database Connection Failed**
```bash
# Check PostgreSQL is running
psql -U postgres -h localhost

# Verify DATABASE_URL
echo $DATABASE_URL

# Test connection
psql -U postgres -h localhost -d study_partner -c "SELECT 1"
```

**Redis Connection Failed**
```bash
# Check Redis is running
redis-cli ping

# Verify REDIS_URL
echo $REDIS_URL

# Test connection
redis-cli -h localhost -p 6379 ping
```

**Migration Errors**
```bash
# Check migration status
alembic current

# View migration history
alembic history

# Downgrade if needed
alembic downgrade -1

# Upgrade to head
alembic upgrade head
```

**Import Errors**
```bash
# Ensure you're in backend directory
cd ai-study-partner/backend

# Reinstall dependencies
pip install -r requirements.txt

# Verify virtual environment
which python
```

**Module Not Found**
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Getting Help

1. **Check Documentation**
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed solutions
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
   - [FAQ](#faq) - Frequently asked questions

2. **Check Logs**
   ```bash
   # Development
   tail -f logs/app.log
   
   # Docker
   docker-compose -f docker-compose.prod.yml logs -f api
   ```

3. **Debug Information**
   ```bash
   # Python version
   python --version
   
   # Dependencies
   pip list
   
   # Environment
   env | grep -E "DATABASE_URL|REDIS_URL|SECRET_KEY"
   
   # System info
   uname -a
   ```

4. **Create Issue**
   - Include error message
   - Include steps to reproduce
   - Include environment info
   - Include relevant logs

### FAQ

**Q: How do I reset the database?**
```bash
# Drop all tables
alembic downgrade base

# Recreate tables
alembic upgrade head
```

**Q: How do I backup the database?**
```bash
# PostgreSQL backup
pg_dump -U postgres -h localhost study_partner > backup.sql

# Restore
psql -U postgres -h localhost study_partner < backup.sql
```

**Q: How do I change the port?**
```bash
# Edit api/main.py or use command line
python api/main.py --port 8001
```

**Q: How do I enable debug logging?**
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env
LOG_LEVEL=DEBUG
```

**Q: How do I run tests in parallel?**
```bash
pytest -n auto
```

**Q: How do I generate a migration?**
```bash
alembic revision --autogenerate -m "Description"
```

### Troubleshooting Documentation

For more detailed troubleshooting, see:
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Comprehensive troubleshooting guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment troubleshooting
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Debugging tips

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Before Contributing

1. **Read the Documentation**
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development workflow
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Code patterns

2. **Setup Development Environment**
   - Follow [Quick Start](#-quick-start)
   - Run tests: `pytest`
   - Check code quality: `black . && flake8 . && mypy .`

3. **Check Existing Issues**
   - Search for similar issues
   - Comment on existing issues if relevant
   - Create new issue if needed

### Contributing Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/ai-study-partner.git
   cd ai-study-partner/backend
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow code standards
   - Add tests for new features
   - Update documentation
   - Ensure all tests pass

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Standards

- **Python:** PEP 8 with Black formatter
- **Type Hints:** Required for all functions
- **Docstrings:** Required for all modules, classes, and functions
- **Tests:** Required for all new features
- **Coverage:** Minimum 80% overall, 100% for security modules

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, style, refactor, perf, test, chore

**Example:**
```
feat(auth): add JWT token refresh endpoint

Added automatic token refresh mechanism to improve user experience.
Tokens now refresh automatically when expired.

Closes #123
```

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Coverage maintained/improved
- [ ] Documentation updated
- [ ] Commit messages follow guidelines
- [ ] No hardcoded secrets
- [ ] No breaking changes

## 📞 Support

### Getting Help

**Documentation:**
- [README.md](README.md) - Project overview
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solutions
- [API_ENDPOINTS.md](API_ENDPOINTS.md) - API reference

**Interactive Resources:**
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc
- http://localhost:8000/health - Health check

**Community:**
- GitHub Issues - Bug reports and feature requests
- GitHub Discussions - Questions and discussions
- Email - security@studypartner.com (security issues)

### Reporting Issues

**Bug Report:**
1. Describe the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment info
6. Relevant logs

**Feature Request:**
1. Describe the feature
2. Use case
3. Proposed solution
4. Alternative solutions

**Security Issue:**
- Email: security@studypartner.com
- Do NOT create public issue
- Include: description, impact, reproduction steps

## 📋 Additional Resources

### Official Documentation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [SQLAlchemy](https://docs.sqlalchemy.org/) - ORM
- [PostgreSQL](https://www.postgresql.org/docs/) - Database
- [Redis](https://redis.io/documentation) - Cache

### Best Practices
- [REST API Best Practices](https://restfulapi.net/)
- [Python Best Practices](https://pep8.org/)
- [Security Best Practices](https://owasp.org/www-project-top-ten/)
- [Git Workflow](https://git-scm.com/book/en/v2)

### Tools
- [Postman](https://www.postman.com/) - API testing
- [DBeaver](https://dbeaver.io/) - Database management
- [Redis Desktop Manager](https://redisdesktop.com/) - Redis management
- [VS Code](https://code.visualstudio.com/) - Code editor

## 📝 License

This project is part of the AI Study Partner project.

## 🎉 Getting Started Checklist

- [ ] Read [Quick Start](#-quick-start)
- [ ] Install dependencies
- [ ] Configure environment
- [ ] Initialize database
- [ ] Run development server
- [ ] Access http://localhost:8000/docs
- [ ] Run tests: `pytest`
- [ ] Read [CONTRIBUTING.md](CONTRIBUTING.md)
- [ ] Create first feature branch
- [ ] Make your first contribution

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Python Version | 3.11+ |
| Framework | FastAPI 0.100+ |
| Database | PostgreSQL 13+ |
| Cache | Redis 6+ |
| API Endpoints | 20+ |
| Database Tables | 18+ |
| Security Modules | 10 |
| Test Coverage | 80%+ |
| Documentation | 30,000+ words |

## 🚀 Quick Links

| Resource | Link |
|----------|------|
| Quick Start | [#-quick-start](#-quick-start) |
| Features | [#-features](#-features) |
| API Docs | http://localhost:8000/docs |
| GitHub | https://github.com/your-org/ai-study-partner |
| Issues | https://github.com/your-org/ai-study-partner/issues |
| Discussions | https://github.com/your-org/ai-study-partner/discussions |

---

**Version:** 1.0.0  
**Last Updated:** March 10, 2026  
**Status:** Production Ready ✅  
**Python:** 3.11+  
**Framework:** FastAPI 0.100+  
**Database:** PostgreSQL 13+  
**Cache:** Redis 6+  
**License:** MIT

**Made with ❤️ by the AI Study Partner Team**
