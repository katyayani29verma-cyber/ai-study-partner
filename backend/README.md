# AI Study Partner - Backend API

A production-ready FastAPI backend for an adaptive learning platform that uses AI to optimize student learning through cognitive load management, content chunking, spaced repetition, and personalized learning paths.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 13+
- Redis 6+
- pip or conda

### Installation

1. **Clone and navigate to backend**
```bash
cd ai-study-partner/backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env .env.local
# Edit .env.local with your settings
```

5. **Initialize database**
```bash
alembic upgrade head
```

6. **Run development server**
```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

## 📋 Project Structure

```
backend/
├── api/                          # FastAPI application
│   ├── main.py                   # App initialization & middleware
│   └── routes/                   # API endpoints
│       ├── auth.py               # Authentication endpoints
│       ├── cognitive_load.py     # Cognitive load tracking
│       ├── content_chunking.py   # Content management
│       ├── revision.py           # Spaced repetition
│       └── learning_path.py      # Adaptive learning paths
│
├── database/                     # Database layer
│   ├── config.py                 # Connection & pooling
│   ├── models.py                 # SQLAlchemy ORM models
│   ├── session.py                # Session management
│   ├── operations.py             # Database operations
│   ├── init.py                   # Database initialization
│   └── README.md                 # Database documentation
│
├── security/                     # Security & authentication
│   ├── auth.py                   # JWT & password management
│   ├── rbac.py                   # Role-based access control
│   ├── validation.py             # Input validation & sanitization
│   ├── encryption.py             # Data encryption
│   ├── csrf.py                   # CSRF protection
│   ├── rate_limit.py             # Rate limiting
│   ├── audit.py                  # Audit logging
│   ├── privacy.py                # GDPR compliance
│   ├── headers.py                # Security headers middleware
│   ├── incident_response.py      # Incident detection
│   ├── config.py                 # Security configuration
│   ├── examples.py               # Usage examples
│   ├── test_security.py          # Unit tests
│   └── README.md                 # Security documentation
│
├── ai_integration/               # AI module adapter
│   └── adapter.py                # Integration with AI modules
│
├── performance/                  # Performance optimization
│   ├── caching.py                # Caching strategies
│   ├── background_tasks.py       # Async tasks
│   ├── monitoring.py             # Performance monitoring
│   └── scaling.py                # Scaling guidelines
│
├── alembic/                      # Database migrations
│   ├── env.py                    # Migration environment
│   ├── script.py.mako            # Migration template
│   └── versions/                 # Migration files
│
├── logs/                         # Application logs
│   └── nginx/                    # Nginx logs
│
├── ssl/                          # SSL certificates
│
├── requirements.txt              # Python dependencies
├── alembic.ini                   # Alembic configuration
├── docker-compose.prod.yml       # Production Docker setup
├── docker-compose.staging.yml    # Staging Docker setup
├── Dockerfile.prod               # Production Docker image
├── nginx.conf                    # Nginx configuration
├── start-api.ps1                 # PowerShell startup script
├── start-api.bat                 # Batch startup script
└── README.md                     # This file
```

## 🔑 Core Features

### 1. Authentication & Authorization
- JWT-based authentication with 30-minute token expiration
- Bcrypt password hashing (cost 12)
- Role-based access control (RBAC)
- Permission-based authorization
- Secure token refresh mechanism

**Endpoints:**
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh
- `GET /auth/me` - Current user info

### 2. Cognitive Load Management
Track and optimize student cognitive load during study sessions to prevent overload and optimize learning.

**Endpoints:**
- `POST /cognitive-load/record` - Record cognitive load metrics
- `GET /cognitive-load/current` - Get current cognitive load
- `GET /cognitive-load/history` - Get historical data

**Metrics Tracked:**
- Mental effort (0-100)
- Working memory load (0-100)
- Attention level (0-100)
- Stress level (0-100)

### 3. Content Chunking
Break down study materials into manageable chunks with difficulty assessment and interaction tracking.

**Endpoints:**
- `POST /content/chunks` - Create content chunks
- `POST /content/chunks/{chunk_id}/interact` - Record chunk interaction
- `GET /content/chunks/{chunk_id}/analytics` - Get chunk analytics

**Features:**
- Automatic content segmentation
- Difficulty assessment
- Interaction tracking
- Comprehension scoring

### 4. Spaced Repetition (Revision Engine)
Implement SM-2 algorithm for optimal review scheduling based on performance.

**Endpoints:**
- `POST /revision/items` - Create revision item
- `GET /revision/due` - Get due items for review
- `POST /revision/review` - Record review session

**Algorithm:**
- SM-2 spaced repetition
- Adaptive scheduling
- Quality-based intervals
- Performance tracking

### 5. Adaptive Learning Paths
Generate personalized learning paths that adapt based on student performance and learning style.

**Endpoints:**
- `POST /learning-path/create` - Create learning path
- `GET /learning-path/{path_id}` - Get path details
- `POST /learning-path/recommend` - Get recommendations

**Features:**
- Personalized path generation
- Performance-based adaptation
- Module recommendations
- Progress tracking

## 🔐 Security Features

### Built-in Security
- ✅ HTTPS/TLS ready
- ✅ CORS protection
- ✅ CSRF tokens
- ✅ SQL injection prevention
- ✅ XSS protection
- ✅ Rate limiting
- ✅ Input validation
- ✅ Secure headers
- ✅ Audit logging
- ✅ GDPR compliance

### Security Modules
See [security/README.md](security/README.md) for detailed documentation.

**Key Modules:**
- `auth.py` - Authentication & JWT
- `rbac.py` - Role-based access control
- `validation.py` - Input validation
- `encryption.py` - Data encryption
- `rate_limit.py` - Request throttling
- `audit.py` - Security event logging

## 📊 Database

### Schema Overview
The database includes 18+ tables organized by feature:

**Core Tables (8):**
- `users` - User accounts
- `study_materials` - Study content
- `study_sessions` - Study sessions
- `flashcards` - Flashcard items
- `knowledge_gaps` - Knowledge gaps
- `curriculum_data` - Curriculum info
- `socratic_sessions` - Socratic dialogues
- `audit_logs` - Security audit logs

**Feature Tables (10):**
- Cognitive Load: `cognitive_load_metrics`, `cognitive_load_thresholds`
- Content Chunking: `content_chunks`, `chunk_interactions`
- Revision: `revision_items`, `revision_reviews`, `revision_schedule`
- Adaptive Learning: `learning_paths`, `path_modules`, `adaptive_recommendations`, `performance_metrics`

### Database Operations
See [database/README.md](database/README.md) for complete documentation.

**Quick Example:**
```python
from database.operations import CognitiveLoadOperations
from database.session import SessionLocal

db = SessionLocal()
metric = CognitiveLoadOperations.record_cognitive_load(
    db, user_id=1,
    mental_effort=65, working_memory_load=70,
    attention_level=75, stress_level=60
)
db.commit()
```

## 🚀 Deployment

### Development
```bash
python api/main.py
```

### Production with Docker
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Staging with Docker
```bash
docker-compose -f docker-compose.staging.yml up -d
```

### Manual Production Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  api.main:app
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following:

```env
# Security
SECRET_KEY=<generate with: openssl rand -hex 32>
MASTER_KEY=<generate with: openssl rand -hex 32>

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/study_partner

# Redis (for caching & rate limiting)
REDIS_URL=redis://localhost:6379

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Environment
ENVIRONMENT=development  # or production

# API Settings
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Logging
LOG_LEVEL=INFO

# File Uploads
MAX_UPLOAD_SIZE=52428800  # 50MB

# Security Headers
HSTS_MAX_AGE=31536000
```

### Database Configuration
Edit `database/config.py` for connection pooling:
```python
POOL_SIZE = 20          # Connections to keep
MAX_OVERFLOW = 10       # Additional connections
POOL_RECYCLE = 3600     # Recycle after 1 hour
```

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health Check
```bash
curl http://localhost:8000/health
```

### Example Requests

**Register User:**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password",
    "full_name": "John Doe"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password"
  }'
```

**Record Cognitive Load:**
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

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest backend/test_security_features.py -v
```

### Run with Coverage
```bash
pytest --cov=. --cov-report=html
```

### Test Files
- `test_security_features.py` - Security module tests
- `test_database_ops.py` - Database operation tests
- `test_all.py` - Comprehensive test suite
- `security/test_security.py` - Security-specific tests

## 📈 Performance & Monitoring

### Caching
- Redis-based caching for frequently accessed data
- Automatic cache invalidation
- TTL-based expiration

### Rate Limiting
- Per-user rate limits
- Per-endpoint rate limits
- Configurable thresholds

### Monitoring
- Request/response logging
- Performance metrics
- Error tracking
- Audit logging

See `performance/` directory for implementation details.

## 🔄 Database Migrations

### Create Migration
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply Migrations
```bash
alembic upgrade head
```

### Rollback Migration
```bash
alembic downgrade -1
```

### View Migration History
```bash
alembic history
```

## 🛠️ Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature
```

### 2. Make Changes
- Update models in `database/models.py`
- Add operations in `database/operations.py`
- Create API routes in `api/routes/`
- Add security checks in `security/`

### 3. Create Migration
```bash
alembic revision --autogenerate -m "Add your feature"
```

### 4. Test Changes
```bash
pytest
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

## 📖 Documentation

### Main Documentation
- [Database Documentation](database/README.md)
- [Security Documentation](security/README.md)
- [API Documentation](http://localhost:8000/docs) (interactive)

### Security
- [Security Guide](../SECURITY.md)
- [Security Checklist](../SECURITY_CHECKLIST.md)
- [Security Implementation Summary](../SECURITY_IMPLEMENTATION_SUMMARY.md)

### Database
- [Database README](database/README.md)
- [Feature-Oriented Schema](database/FEATURE_ORIENTED_SCHEMA.md)
- [Feature Guide](database/FEATURE_GUIDE.md)

## 🆘 Troubleshooting

### Database Connection Failed
```bash
# Check PostgreSQL is running
psql -U postgres -h localhost

# Verify DATABASE_URL in .env
# Format: postgresql://user:password@host:port/database
```

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Verify REDIS_URL in .env
# Format: redis://host:port
```

### Migration Errors
```bash
# Check migration status
alembic current

# View migration history
alembic history

# Downgrade if needed
alembic downgrade -1
```

### Port Already in Use
```bash
# Change port in api/main.py or use:
python api/main.py --port 8001
```

### Import Errors
```bash
# Ensure you're in the backend directory
cd ai-study-partner/backend

# Reinstall dependencies
pip install -r requirements.txt
```

## 🔗 Related Documentation

- **Frontend**: `../frontend/README.md`
- **AI Modules**: `../ai_modules/README.md`
- **Project Root**: `../README.md`

## 📝 License

Part of the AI Study Partner project.

## 🤝 Contributing

1. Follow the development workflow above
2. Ensure all tests pass
3. Update documentation
4. Submit pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review relevant documentation
3. Check existing issues
4. Create a new issue with details

---

**Version:** 1.0.0  
**Last Updated:** March 10, 2026  
**Status:** Production Ready  
**Python:** 3.11+  
**Framework:** FastAPI 0.100+  
**Database:** PostgreSQL 13+  
**Cache:** Redis 6+
