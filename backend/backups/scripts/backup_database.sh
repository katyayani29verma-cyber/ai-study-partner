#!/bin/bash
# Database backup script for AI Study Partner Backend

set -e

# Configuration
BACKUP_TYPE=${1:-full}
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/../logs/backup.log"

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

# Verify environment
if [ -z "$DATABASE_URL" ]; then
    error_exit "DATABASE_URL not set"
fi

# Extract database credentials
DB_URL=$DATABASE_URL
DB_USER=$(echo $DB_URL | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
DB_HOST=$(echo $DB_URL | sed -n 's/.*@\([^:]*\).*/\1/p')
DB_NAME=$(echo $DB_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')

log "Starting $BACKUP_TYPE backup..."
log "Database: $DB_NAME"
log "Host: $DB_HOST"

# Create backup directories
mkdir -p "$BACKUP_DIR/database/full"
mkdir -p "$BACKUP_DIR/database/incremental"

case $BACKUP_TYPE in
    full)
        BACKUP_FILE="$BACKUP_DIR/database/full/db_backup_${TIMESTAMP}.sql"
        log "Creating full backup..."
        
        # Create full backup
        pg_dump -U $DB_USER -h $DB_HOST $DB_NAME > "$BACKUP_FILE" 2>> "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Full backup completed: $BACKUP_FILE ($SIZE)${NC}"
            
            # Compress backup
            gzip "$BACKUP_FILE"
            BACKUP_FILE="${BACKUP_FILE}.gz"
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Backup compressed: $SIZE${NC}"
            
            # Create checksum
            sha256sum "$BACKUP_FILE" > "${BACKUP_FILE}.sha256"
            log "${GREEN}✓ Checksum created${NC}"
        else
            error_exit "Full backup failed"
        fi
        ;;
        
    incremental)
        BACKUP_FILE="$BACKUP_DIR/database/incremental/db_backup_${TIMESTAMP}.sql"
        log "Creating incremental backup..."
        
        # For PostgreSQL, we use pg_dump with custom format for incremental support
        pg_dump -U $DB_USER -h $DB_HOST -Fc $DB_NAME > "$BACKUP_FILE" 2>> "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Incremental backup completed: $BACKUP_FILE ($SIZE)${NC}"
            
            # Create checksum
            sha256sum "$BACKUP_FILE" > "${BACKUP_FILE}.sha256"
            log "${GREEN}✓ Checksum created${NC}"
        else
            error_exit "Incremental backup failed"
        fi
        ;;
        
    test)
        BACKUP_FILE="$BACKUP_DIR/database/full/db_backup_test_${TIMESTAMP}.sql"
        log "Creating test backup..."
        
        # Create test backup
        pg_dump -U $DB_USER -h $DB_HOST $DB_NAME > "$BACKUP_FILE" 2>> "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            log "${GREEN}✓ Test backup completed: $BACKUP_FILE ($SIZE)${NC}"
            
            # Verify backup
            if [ -s "$BACKUP_FILE" ]; then
                log "${GREEN}✓ Backup file is not empty${NC}"
            else
                error_exit "Backup file is empty"
            fi
            
            # Cleanup test backup
            rm "$BACKUP_FILE"
            log "${GREEN}✓ Test backup cleaned up${NC}"
        else
            error_exit "Test backup failed"
        fi
        ;;
        
    *)
        error_exit "Unknown backup type: $BACKUP_TYPE. Use: full, incremental, or test"
        ;;
esac

log "${GREEN}✅ Backup completed successfully${NC}"
