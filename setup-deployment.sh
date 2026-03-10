#!/bin/bash

# Deployment Pipeline Quick Start Script
# This script automates the initial setup of the deployment pipeline

set -e

echo "================================"
echo "Deployment Pipeline Setup"
echo "================================"
echo ""

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI not found. Please install it first:"
    echo "   macOS: brew install gh"
    echo "   Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    echo "   Windows: choco install gh"
    exit 1
fi

# Check if SSH key is already set up
echo "📋 Checking prerequisites..."

if [ ! -d ".git" ]; then
    echo "❌ Not in a Git repository. Please run this from your project root."
    exit 1
fi

echo "✅ Git repository found"

# Generate SSH keys for deployment
echo ""
echo "🔑 Setting up SSH keys for deployment..."

if [ ! -f "deploy_key" ]; then
    echo "Generating SSH key pair..."
    ssh-keygen -t ed25519 -f deploy_key -C "deploy@yourapp.com" -N "" || true
    echo "✅ SSH keys generated: deploy_key (private), deploy_key.pub (public)"
else
    echo "⏭️  deploy_key already exists, skipping generation"
fi

# Generate security keys
echo ""
echo "🔐 Generating security keys..."

SECRET_KEY=$(openssl rand -hex 32)
MASTER_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -base64 20)
REDIS_PASSWORD=$(openssl rand -base64 20)

echo "✅ Security keys generated"

# Display instructions for adding secrets
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Add SSH public key to your servers:"
echo "   cat deploy_key.pub"
echo "   # Then add to ~/.ssh/authorized_keys on staging and production servers"
echo ""
echo "2. Add these GitHub Secrets (Settings → Secrets and variables → Actions):"
echo ""
echo "   Staging Secrets:"
echo "   - STAGING_HOST: <your-staging-server-ip>"
echo "   - STAGING_USER: <ssh-user>"
echo "   - STAGING_DEPLOY_KEY: (copy content of deploy_key)"
echo ""
echo "   Production Secrets:"
echo "   - PROD_HOST: <your-prod-server-ip>"
echo "   - PROD_USER: <ssh-user>"
echo "   - PROD_DEPLOY_KEY: (copy content of deploy_key)"
echo "   - PROD_API_URL: https://api.yourapp.com"
echo ""
echo "   Database & Security:"
echo "   - DB_PASSWORD: $DB_PASSWORD"
echo "   - REDIS_PASSWORD: $REDIS_PASSWORD"
echo "   - SECRET_KEY: $SECRET_KEY"
echo "   - MASTER_KEY: $MASTER_KEY"
echo ""
echo "3. Configure on staging/production servers:"
echo "   mkdir -p /app/study-partner"
echo "   cd /app/study-partner"
echo "   git clone <your-repo-url> ."
echo "   mkdir -p backups logs/nginx"
echo ""
echo "4. Create .env file in backend/ with the security keys above"
echo ""
echo "5. Test the pipeline by pushing to develop/main branch"
echo ""
echo "❓ For detailed instructions, see DEPLOYMENT_PIPELINE.md"
echo ""
