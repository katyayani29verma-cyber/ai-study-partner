#!/bin/bash
# Database restore script for AI Study Partner Backend

set -e

# Configuration
BACKUP_FILE=${1:-""}
RESTORE_MODE=${2:-""}
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$BACKUP_DIR/../logs/restore.log"

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

# Show usage
usage() {
    echo "Usage: $0 <backup_file> [options]"
    echo ""
    echo "Options:"
    echo "  latest              Restore latest full backup"
    echo "  --test              Test restore without applying"
    echo "  --time <timestamp>  Restore to specific time"
    echo ""
    echo "Examples:"
    echo "  $0 latest"
    echo "  $0 database/full/db_backup_20260310_000000.sql.gz"
    echo "  $0 database/full/db_backup_20260310_000000.sql.gz --test"
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

# Handle "latest" option
if [ "$BACKUP_FILE" = "latest" ]; then
    BACKUP_FILE=$(ls -t "$BACKUP_DIR/database/full/"*.sql.gz 2>/dev/null | head -1)
    if [ -z "$BACKUP_FILE" ]; then
        error_exit "No backup files found"
    fi
    log "Using latest backup: $BACKUP_FILE"
fi

# Validate backup file
if [ -z "$BACKUP_FILE" ]; then
    usage
fi

if [ ! -f "$BACKUP_FILE" ]; then
    error_exit "Backup file not found: $BACKUP_FILE"
fi

log "Starting database restore..."
log "Database: $DB_NAME"
log "Host: $DB_HOST"
log "Backup file: $BACKUP_FILE"

# Verify backup integrity
if [ -f "${BACKUP_FILE}.sha256" ]; then
    log "Verifying backup integrity..."
    cd "$(dirname "$BACKUP_FILE")"
    if sha256sum -c "$(basename "$BACKUP_FILE").sha256" >> "$LOG_FILE" 2>&1; then
        log "${GREEN}✓ Backup integrity verified${NC}"
    else
        error_exit "Backup integrity check failed"
    fi
    cd - > /dev/null
fi

# Create backup of current database
log "Creating backup of current database..."
CURRENT_BACKUP="$BACKUP_DIR/database/full/db_backup_before_restore_${TIMESTAMP}.sql.gz"
pg_dump -U $DB_USER -h $DB_HOST $DB_NAME | gzip > "$CURRENT_BACKUP" 2>> "$LOG_FILE"
log "${GREEN}✓ Current database backed up: $CURRENT_BACKUP${NC}"

# Decompress backup if needed
RESTORE_FILE="$BACKUP_FILE"
if [[ "$BACKUP_FILE" == *.gz ]]; then
    log "Decompressing backup..."
    RESTORE_FILE="${BACKUP_FILE%.gz}"
    gunzip -c "$BACKUP_FILE" > "$RESTORE_FILE"
    log "${GREEN}✓ Backup decompressed${NC}"
fi

# Test restore mode
if [ "$RESTORE_MODE" = "--test" ]; then
    log "Running test restore (no changes will be applied)..."
    
    # Create temporary test database
    TEST_DB="${DB_NAME}_test_restore"
    log "Creating test database: $TEST_DB"
    
    psql -U $DB_USER -h $DB_HOST -c "DROP DATABASE IF EXISTS $TEST_DB;" 2>> "$LOG_FILE"
    psql -U $DB_USER -h $DB_HOST -c "CREATE DATABASE $TEST_DB;" 2>> "$LOG_FILE"
    
    # Restore to test database
    log "Restoring to test database..."
    psql -U $DB_USER -h $DB_HOST $TEST_DB < "$RESTORE_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}✓ Test restore successful${NC}"
        
        # Verify test database
        COUNT=$(psql -U $DB_USER -h $DB_HOST $TEST_DB -c "SELECT COUNT(*) FROM users;" 2>/dev/null | tail -1)
        log "${GREEN}✓ Test database contains $COUNT users${NC}"
        
        # Cleanup test database
        psql -U $DB_USER -h $DB_HOST -c "DROP DATABASE $TEST_DB;" 2>> "$LOG_FILE"
        log "${GREEN}✓ Test database cleaned up${NC}"
    else
        error_exit "Test restore failed"
    fi
else
    # Actual restore
    log "Restoring database..."
    log "${YELLOW}WARNING: This will overwrite the current database!${NC}"
    log "Press Ctrl+C to cancel (5 seconds)..."
    sleep 5
    
    # Drop current database and recreate
    log "Dropping current database..."
    psql -U $DB_USER -h $DB_HOST -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>> "$LOG_FILE"
    psql -U $DB_USER -h $DB_HOST -c "CREATE DATABASE $DB_NAME;" 2>> "$LOG_FILE"
    
    # Restore from backup
    log "Restoring from backup..."
    psql -U $DB_USER -h $DB_HOST $DB_NAME < "$RESTORE_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}✓ Database restore completed${NC}"
        
        # Verify restore
        COUNT=$(psql -U $DB_USER -h $DB_HOST $DB_NAME -c "SELECT COUNT(*) FROM users;" 2>/dev/null | tail -1)
        log "${GREEN}✓ Restored database contains $COUNT users${NC}"
    else
        error_exit "Database restore failed"
    fi
fi

# Cleanup
if [[ "$BACKUP_FILE" == *.gz ]] && [ -f "$RESTORE_FILE" ]; then
    rm "$RESTORE_FILE"
fi

log "${GREEN}✅ Restore completed successfully${NC}"
