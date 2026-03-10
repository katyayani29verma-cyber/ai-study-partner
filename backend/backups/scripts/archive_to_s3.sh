#!/bin/bash
# S3 archival script for AI Study Partner Backend backups

set -e

# Configuration
BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BACKUP_DIR/../logs/backup.log"

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-ai-study-partner-backups}"
S3_PREFIX="${S3_PREFIX:-backups/}"
S3_STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD}"
S3_ENCRYPTION="${S3_ENCRYPTION:-AES256}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    error_exit "AWS CLI not found. Please install AWS CLI."
fi

# Verify AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    error_exit "AWS credentials not set. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
fi

log "Starting S3 archival..."
log "S3 Bucket: $S3_BUCKET"
log "S3 Prefix: $S3_PREFIX"
log "Storage Class: $S3_STORAGE_CLASS"
log "Encryption: $S3_ENCRYPTION"

# Verify S3 bucket exists
log "Verifying S3 bucket..."
if ! aws s3 ls "s3://$S3_BUCKET" --region "$AWS_REGION" > /dev/null 2>&1; then
    error_exit "S3 bucket not found: $S3_BUCKET"
fi
log "${GREEN}✓ S3 bucket verified${NC}"

# Archive counter
UPLOADED=0
FAILED=0

# Function to upload file to S3
upload_to_s3() {
    local file=$1
    local s3_path=$2
    
    if [ ! -f "$file" ]; then
        return
    fi
    
    log "Uploading: $file"
    
    # Upload file with metadata
    if aws s3 cp "$file" "s3://$S3_BUCKET/$S3_PREFIX$s3_path" \
        --region "$AWS_REGION" \
        --storage-class "$S3_STORAGE_CLASS" \
        --sse "$S3_ENCRYPTION" \
        --metadata "backup-date=$TIMESTAMP,backup-type=database" \
        >> "$LOG_FILE" 2>&1; then
        
        SIZE=$(du -h "$file" | cut -f1)
        log "${GREEN}✓ Uploaded: $s3_path ($SIZE)${NC}"
        ((UPLOADED++))
    else
        log "${RED}✗ Failed to upload: $file${NC}"
        ((FAILED++))
    fi
}

# Archive full backups
log ""
log "Archiving full backups..."
if [ -d "$BACKUP_DIR/database/full" ]; then
    for backup in "$BACKUP_DIR/database/full"/*.sql.gz; do
        if [ -f "$backup" ]; then
            filename=$(basename "$backup")
            upload_to_s3 "$backup" "full/$filename"
            
            # Also upload checksum
            if [ -f "${backup}.sha256" ]; then
                upload_to_s3 "${backup}.sha256" "full/${filename}.sha256"
            fi
        fi
    done
fi

# Archive incremental backups
log ""
log "Archiving incremental backups..."
if [ -d "$BACKUP_DIR/database/incremental" ]; then
    for backup in "$BACKUP_DIR/database/incremental"/*; do
        if [ -f "$backup" ] && [[ ! "$backup" == *.sha256 ]]; then
            filename=$(basename "$backup")
            upload_to_s3 "$backup" "incremental/$filename"
            
            # Also upload checksum
            if [ -f "${backup}.sha256" ]; then
                upload_to_s3 "${backup}.sha256" "incremental/${filename}.sha256"
            fi
        fi
    done
fi

# Archive configuration backups
log ""
log "Archiving configuration backups..."
if [ -d "$BACKUP_DIR/config" ]; then
    for backup in "$BACKUP_DIR/config"/*; do
        if [ -f "$backup" ] && [[ ! "$backup" == *.sha256 ]]; then
            filename=$(basename "$backup")
            upload_to_s3 "$backup" "config/$filename"
            
            # Also upload checksum
            if [ -f "${backup}.sha256" ]; then
                upload_to_s3 "${backup}.sha256" "config/${filename}.sha256"
            fi
        fi
    done
fi

# Archive log backups
log ""
log "Archiving log backups..."
if [ -d "$BACKUP_DIR/logs" ]; then
    for archive in "$BACKUP_DIR/logs"/*.tar.gz "$BACKUP_DIR/logs"/*.tar.bz2 "$BACKUP_DIR/logs"/*.tar.xz; do
        if [ -f "$archive" ]; then
            filename=$(basename "$archive")
            upload_to_s3 "$archive" "logs/$filename"
            
            # Also upload checksum
            if [ -f "${archive}.sha256" ]; then
                upload_to_s3 "${archive}.sha256" "logs/${filename}.sha256"
            fi
        fi
    done
fi

# Verify uploads
log ""
log "Verifying S3 uploads..."
REMOTE_COUNT=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX" --recursive --region "$AWS_REGION" | wc -l)
log "${GREEN}✓ S3 contains $REMOTE_COUNT objects${NC}"

# Setup S3 lifecycle policy (optional)
log ""
log "Configuring S3 lifecycle policy..."

# Create lifecycle policy JSON
LIFECYCLE_POLICY=$(cat <<EOF
{
    "Rules": [
        {
            "Id": "ArchiveOldBackups",
            "Status": "Enabled",
            "Prefix": "backups/",
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 90,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        }
    ]
}
EOF
)

# Apply lifecycle policy
if aws s3api put-bucket-lifecycle-configuration \
    --bucket "$S3_BUCKET" \
    --lifecycle-configuration "$LIFECYCLE_POLICY" \
    --region "$AWS_REGION" \
    >> "$LOG_FILE" 2>&1; then
    log "${GREEN}✓ S3 lifecycle policy configured${NC}"
else
    log "${YELLOW}⚠️  Could not configure S3 lifecycle policy${NC}"
fi

# Summary
log ""
log "S3 Archival Summary:"
log "  Uploaded: $UPLOADED files"
log "  Failed: $FAILED files"
log "  S3 Location: s3://$S3_BUCKET/$S3_PREFIX"
log "  Remote Objects: $REMOTE_COUNT"

if [ $FAILED -eq 0 ]; then
    log "${GREEN}✅ S3 archival completed successfully${NC}"
else
    log "${YELLOW}⚠️  S3 archival completed with $FAILED failures${NC}"
fi

