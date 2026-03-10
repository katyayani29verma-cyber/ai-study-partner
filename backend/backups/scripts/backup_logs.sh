#!/bin/bash
# Log backup and archival script for AI Study Partner Backend

set -e

# Configuration
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/../logs/backup.log"
COMPRESSION="${1:-gzip}"

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
mkdir -p "$BACKUP_DIR/logs/app"
mkdir -p "$BACKUP_DIR/logs/security"
mkdir -p "$BACKUP_DIR/logs/audit"

log "Starting log backup and archival..."

# Archive counter
ARCHIVED=0
FAILED=0

# Function to archive logs
archive_logs() {
    local log_dir=$1
    local log_type=$2
    local archive_name="$BACKUP_DIR/logs/${log_type}_logs_${TIMESTAMP}"
    
    if [ -d "$log_dir" ] && [ "$(ls -A "$log_dir")" ]; then
        log "Archiving $log_type logs from $log_dir..."
        
        case $COMPRESSION in
            gzip)
                tar -czf "${archive_name}.tar.gz" -C "$(dirname "$log_dir")" "$(basename "$log_dir")" 2>> "$LOG_FILE"
                ARCHIVE_FILE="${archive_name}.tar.gz"
                ;;
            bzip2)
                tar -cjf "${archive_name}.tar.bz2" -C "$(dirname "$log_dir")" "$(basename "$log_dir")" 2>> "$LOG_FILE"
                ARCHIVE_FILE="${archive_name}.tar.bz2"
                ;;
            xz)
                tar -cJf "${archive_name}.tar.xz" -C "$(dirname "$log_dir")" "$(basename "$log_dir")" 2>> "$LOG_FILE"
                ARCHIVE_FILE="${archive_name}.tar.xz"
                ;;
            *)
                tar -cf "${archive_name}.tar" -C "$(dirname "$log_dir")" "$(basename "$log_dir")" 2>> "$LOG_FILE"
                ARCHIVE_FILE="${archive_name}.tar"
                ;;
        esac
        
        if [ $? -eq 0 ] && [ -f "$ARCHIVE_FILE" ]; then
            SIZE=$(du -h "$ARCHIVE_FILE" | cut -f1)
            
            # Create checksum
            sha256sum "$ARCHIVE_FILE" > "${ARCHIVE_FILE}.sha256"
            
            log "${GREEN}✓ Archived $log_type logs: $ARCHIVE_FILE ($SIZE)${NC}"
            ((ARCHIVED++))
            
            # Rotate old logs (keep current month + 2 previous months)
            find "$log_dir" -type f -mtime +90 -delete 2>/dev/null || true
        else
            log "${RED}✗ Failed to archive $log_type logs${NC}"
            ((FAILED++))
        fi
    else
        log "${YELLOW}⊘ No logs found in $log_dir${NC}"
    fi
}

# Archive application logs
if [ -d "../logs" ]; then
    archive_logs "../logs" "app"
fi

# Archive security logs (if they exist)
if [ -d "../security/logs" ]; then
    archive_logs "../security/logs" "security"
fi

# Archive audit logs (if they exist)
if [ -d "../logs/audit" ]; then
    archive_logs "../logs/audit" "audit"
fi

# Cleanup old archives (keep 90 days)
log "Cleaning up old log archives..."
find "$BACKUP_DIR/logs" -name "*.tar.gz" -o -name "*.tar.bz2" -o -name "*.tar.xz" -o -name "*.tar" | while read archive; do
    if [ -f "$archive" ]; then
        # Check if older than 90 days
        if [ $(find "$archive" -mtime +90 2>/dev/null | wc -l) -gt 0 ]; then
            log "Removing old archive: $archive"
            rm -f "$archive" "${archive}.sha256"
        fi
    fi
done

# Summary
log ""
log "Log Backup Summary:"
log "  Archived: $ARCHIVED log sets"
log "  Failed: $FAILED log sets"
log "  Compression: $COMPRESSION"
log "  Location: $BACKUP_DIR/logs/"

if [ $FAILED -eq 0 ]; then
    log "${GREEN}✅ Log backup completed successfully${NC}"
else
    log "${YELLOW}⚠️  Log backup completed with $FAILED failures${NC}"
fi

