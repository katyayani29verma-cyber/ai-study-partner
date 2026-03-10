#!/bin/bash
# Configuration backup script for AI Study Partner Backend

set -e

# Configuration
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/../logs/backup.log"
DEST_DIR="${1:-.}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Create backup directories
mkdir -p "$BACKUP_DIR/config"

log "Starting configuration backup..."

# Array of files to backup
CONFIG_FILES=(
    ".env"
    ".env.production"
    "nginx.conf"
    "gunicorn_config.py"
    "logging_config.py"
    "alembic.ini"
)

# SSL files (if they exist)
SSL_FILES=(
    "ssl/configs/nginx-ssl.conf"
)

# Backup counter
BACKED_UP=0
FAILED=0

# Backup configuration files
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        BACKUP_FILE="$BACKUP_DIR/config/${file##*/}.backup_${TIMESTAMP}"
        
        # Copy file
        cp "$file" "$BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            # Set permissions (readable by owner only)
            chmod 600 "$BACKUP_FILE"
            
            # Create checksum
            sha256sum "$BACKUP_FILE" > "${BACKUP_FILE}.sha256"
            
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Backed up: $file ($SIZE)${NC}"
            ((BACKED_UP++))
        else
            log "${RED}✗ Failed to backup: $file${NC}"
            ((FAILED++))
        fi
    else
        log "${YELLOW}⊘ Not found: $file${NC}"
    fi
done

# Backup SSL files
for file in "${SSL_FILES[@]}"; do
    if [ -f "$file" ]; then
        BACKUP_FILE="$BACKUP_DIR/config/${file##*/}.backup_${TIMESTAMP}"
        
        # Copy file
        cp "$file" "$BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            # Set permissions (readable by owner only)
            chmod 600 "$BACKUP_FILE"
            
            # Create checksum
            sha256sum "$BACKUP_FILE" > "${BACKUP_FILE}.sha256"
            
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Backed up: $file ($SIZE)${NC}"
            ((BACKED_UP++))
        else
            log "${RED}✗ Failed to backup: $file${NC}"
            ((FAILED++))
        fi
    fi
done

# Summary
log ""
log "Configuration Backup Summary:"
log "  Backed up: $BACKED_UP files"
log "  Failed: $FAILED files"
log "  Location: $BACKUP_DIR/config/"

if [ $FAILED -eq 0 ]; then
    log "${GREEN}✅ Configuration backup completed successfully${NC}"
else
    log "${YELLOW}⚠️  Configuration backup completed with $FAILED failures${NC}"
fi

