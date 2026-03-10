#!/bin/bash
# Cleanup old backups script

set -e

# Configuration
RETENTION_DAYS=${1:-30}
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

log "Starting cleanup of backups older than $RETENTION_DAYS days..."

# Cleanup full backups
log "Cleaning up full backups..."
FULL_BACKUP_DIR="$BACKUP_DIR/database/full"
if [ -d "$FULL_BACKUP_DIR" ]; then
    DELETED_COUNT=0
    FREED_SPACE=0
    
    while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        rm -f "$file" "${file}.sha256"
        log "${GREEN}✓ Deleted: $(basename "$file") ($SIZE)${NC}"
        ((DELETED_COUNT++))
    done < <(find "$FULL_BACKUP_DIR" -type f -name "*.sql.gz" -mtime +$RETENTION_DAYS)
    
    log "${GREEN}✓ Deleted $DELETED_COUNT full backups${NC}"
fi

# Cleanup incremental backups (keep 7 days)
log "Cleaning up incremental backups..."
INCREMENTAL_BACKUP_DIR="$BACKUP_DIR/database/incremental"
if [ -d "$INCREMENTAL_BACKUP_DIR" ]; then
    DELETED_COUNT=0
    
    while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        rm -f "$file" "${file}.sha256"
        log "${GREEN}✓ Deleted: $(basename "$file") ($SIZE)${NC}"
        ((DELETED_COUNT++))
    done < <(find "$INCREMENTAL_BACKUP_DIR" -type f -name "*.sql" -mtime +7)
    
    log "${GREEN}✓ Deleted $DELETED_COUNT incremental backups${NC}"
fi

# Cleanup config backups (keep 90 days)
log "Cleaning up config backups..."
CONFIG_BACKUP_DIR="$BACKUP_DIR/config"
if [ -d "$CONFIG_BACKUP_DIR" ]; then
    DELETED_COUNT=0
    
    while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        rm -f "$file"
        log "${GREEN}✓ Deleted: $(basename "$file") ($SIZE)${NC}"
        ((DELETED_COUNT++))
    done < <(find "$CONFIG_BACKUP_DIR" -type f -mtime +90)
    
    log "${GREEN}✓ Deleted $DELETED_COUNT config backups${NC}"
fi

# Cleanup log archives (keep 90 days)
log "Cleaning up log archives..."
LOG_BACKUP_DIR="$BACKUP_DIR/logs"
if [ -d "$LOG_BACKUP_DIR" ]; then
    DELETED_COUNT=0
    
    while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        rm -f "$file"
        log "${GREEN}✓ Deleted: $(basename "$file") ($SIZE)${NC}"
        ((DELETED_COUNT++))
    done < <(find "$LOG_BACKUP_DIR" -type f -name "*.tar.gz" -mtime +90)
    
    log "${GREEN}✓ Deleted $DELETED_COUNT log archives${NC}"
fi

# Report storage usage
log "Current backup storage usage:"
du -sh "$BACKUP_DIR"/*/ 2>/dev/null | while read line; do
    log "  $line"
done

log "${GREEN}✅ Cleanup completed${NC}"
