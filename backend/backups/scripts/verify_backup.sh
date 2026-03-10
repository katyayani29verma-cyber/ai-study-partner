#!/bin/bash
# Backup verification script

set -e

# Configuration
BACKUP_FILE=${1:-""}
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$BACKUP_DIR/../logs/backup.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Examples:"
    echo "  $0 database/full/db_backup_20260310_000000.sql.gz"
    echo "  $0 database/incremental/db_backup_20260310_000000.sql"
    exit 1
}

if [ -z "$BACKUP_FILE" ]; then
    usage
fi

if [ ! -f "$BACKUP_FILE" ]; then
    error_exit "Backup file not found: $BACKUP_FILE"
fi

log "Starting backup verification..."
log "Backup file: $BACKUP_FILE"

# Check file size
FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log "File size: $FILE_SIZE"

if [ ! -s "$BACKUP_FILE" ]; then
    error_exit "Backup file is empty"
fi

# Verify checksum if available
if [ -f "${BACKUP_FILE}.sha256" ]; then
    log "Verifying checksum..."
    cd "$(dirname "$BACKUP_FILE")"
    if sha256sum -c "$(basename "$BACKUP_FILE").sha256" >> "$LOG_FILE" 2>&1; then
        log "${GREEN}✓ Checksum verification passed${NC}"
    else
        error_exit "Checksum verification failed"
    fi
    cd - > /dev/null
else
    log "${YELLOW}⚠ No checksum file found${NC}"
fi

# Verify backup format
log "Verifying backup format..."

if [[ "$BACKUP_FILE" == *.gz ]]; then
    # Verify gzip format
    if gzip -t "$BACKUP_FILE" 2>> "$LOG_FILE"; then
        log "${GREEN}✓ Gzip format valid${NC}"
    else
        error_exit "Gzip format invalid"
    fi
    
    # Check if it's a SQL dump
    if gunzip -c "$BACKUP_FILE" | head -1 | grep -q "PostgreSQL"; then
        log "${GREEN}✓ PostgreSQL dump format detected${NC}"
    else
        log "${YELLOW}⚠ Could not verify PostgreSQL format${NC}"
    fi
else
    # Check if it's a SQL dump
    if head -1 "$BACKUP_FILE" | grep -q "PostgreSQL"; then
        log "${GREEN}✓ PostgreSQL dump format detected${NC}"
    else
        log "${YELLOW}⚠ Could not verify PostgreSQL format${NC}"
    fi
fi

# Verify backup age
BACKUP_AGE=$(find "$BACKUP_FILE" -type f -printf '%T@\n' | xargs -I {} date -d @{} +%s)
CURRENT_TIME=$(date +%s)
AGE_SECONDS=$((CURRENT_TIME - BACKUP_AGE))
AGE_HOURS=$((AGE_SECONDS / 3600))
AGE_DAYS=$((AGE_HOURS / 24))

log "Backup age: $AGE_DAYS days, $((AGE_HOURS % 24)) hours"

if [ $AGE_DAYS -gt 30 ]; then
    log "${YELLOW}⚠ Backup is older than 30 days${NC}"
fi

# Verify backup contains data
log "Verifying backup contains data..."

if [[ "$BACKUP_FILE" == *.gz ]]; then
    LINE_COUNT=$(gunzip -c "$BACKUP_FILE" | wc -l)
else
    LINE_COUNT=$(wc -l < "$BACKUP_FILE")
fi

log "Backup contains $LINE_COUNT lines"

if [ $LINE_COUNT -lt 100 ]; then
    error_exit "Backup appears to be too small (less than 100 lines)"
fi

log "${GREEN}✓ Backup contains sufficient data${NC}"

# Summary
log ""
log "${GREEN}✅ Backup verification completed successfully${NC}"
log ""
log "Summary:"
log "  File: $BACKUP_FILE"
log "  Size: $FILE_SIZE"
log "  Age: $AGE_DAYS days"
log "  Lines: $LINE_COUNT"
log "  Status: VALID"
