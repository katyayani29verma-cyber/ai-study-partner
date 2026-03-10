#!/bin/bash

# AI Study Partner - Security Setup Script

echo "🔒 AI Study Partner - Security Setup"
echo "===================================="
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Skipping creation."
else
    echo "📝 Creating .env file..."
    
    # Generate keys
    SECRET_KEY=$(openssl rand -hex 32)
    MASTER_KEY=$(openssl rand -hex 32)
    
    # Create .env file
    cat > .env << EOF
# JWT Configuration
SECRET_KEY=$SECRET_KEY
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Encryption
MASTER_KEY=$MASTER_KEY

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/study_partner

# CORS
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Redis (for rate limiting)
REDIS_URL=redis://localhost:6379

# File Upload
MAX_UPLOAD_SIZE=52428800
EOF
    
    echo "✅ .env file created with secure keys"
    echo ""
    echo "Generated Keys:"
    echo "  SECRET_KEY: $SECRET_KEY"
    echo "  MASTER_KEY: $MASTER_KEY"
fi

echo ""
echo "📦 Installing dependencies..."
pip install -r security/requirements.txt

echo ""
echo "✅ Security setup complete!"
echo ""
echo "Next steps:"
echo "1. Update DATABASE_URL in .env"
echo "2. Update ALLOWED_ORIGINS in .env"
echo "3. Start Redis: redis-server"
echo "4. Run: python -m uvicorn main:app --reload"
echo ""
echo "For more information, see SECURITY.md"
