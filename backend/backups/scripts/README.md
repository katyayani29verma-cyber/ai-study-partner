# Backup Scripts Documentation

Complete guide to all backup and recovery scripts for AI Study Partner Backend.

## 📋 Scripts Overview

| Script | Purpose | Frequency | Retention |
|--------|---------|-----------|-----------|
| `backup_database.sh` | Database backups | Daily (full), 6h (incremental) | 30 days (full), 7 days (incremental) |
| `backup_config.sh` | Configuration backups | On deployment | 90 days |
| `backup_logs.sh` | Log archival | Daily | 90 days |
| `restore_database.sh` | Database restore | On-demand | N/A |
| `restore_config.sh` | Configuration restore | On-demand | N/A |
| `cleanup_old_backups.sh` | Backup cleanup | Daily | N/A |
| `verify_backup.sh` | Backup verification | Daily | N/A |
| `archive_to_s3.sh` | S3 archival | Weekly | N/A |

## 🗄️ Database Backup Script

### Usage

```bash
# Full backup
bash backup_database.sh full

# Incremental backup
bash backup_database.sh incremental

# Test backup (no file saved)
bash backup_database.sh test
```

### Features

- **Full Backups:** Complete database snapshot with compression
- **Incremental Backups:** Changes since last backup in custom format
- **Test Mode:** Verify backup process without saving files
- **Checksums:** SHA256 verification for integrity
- **Logging:** Detailed logs with timestamps
- **Error Handling:** Automatic rollback on failure

### Output

```
[2026-03-11 10:00:00] Starting full backup...
[2026-03-11 10:00:00] Database: study_partner
[2026-03-11 10:00:00] Host: localhost
[2026-03-11 10:05:30] ✓ Full backup completed: database/full/db_backup_20260311_100000.sql (450MB)
[2026-03-11 10:06:15] ✓ Backup compressed: 85MB
[2026-03-11 10:06:16] ✓ Checksum created
[2026-03-11 10:06:16] ✅ Backup completed successfully
```

### Environment Variables

```bash
# Required
export DATABASE_URL="postgresql://user:password@localhost/study_partner"

# Optional
export BACKUP_COMPRESSION="gzip"  # gzip, bzip2, xz
export BACKUP_THREADS="4"         # Parallel threads
export BACKUP_TIMEOUT="3600"      # Timeout in seconds
```

### Backup File Structure

```
database/
├── full/
│   ├── db_backup_20260311_000000.sql.gz
│   ├── db_backup_20260311_000000.sql.gz.sha256
│   ├── db_backup_20260310_000000.sql.gz
│   └── db_backup_20260310_000000.sql.gz.sha256
└── incremental/
    ├── db_backup_20260311_060000
    ├── db_backup_20260311_060000.sha256
    ├── db_backup_20260311_120000
    └── db_backup_20260311_120000.sha256
```

## 🔧 Configuration Backup Script

### Usage

```bash
# Backup all configurations
bash backup_config.sh

# Backup specific config
bash backup_config.sh .env

# Backup with custom destination
bash backup_config.sh --dest /backup/configs
```

### Features

- **Environment Files:** .env, .env.production
- **Application Config:** nginx.conf, gunicorn_config.py
- **Security Config:** SSL certificates, keys
- **Encryption:** AES-256 encryption
- **Versioning:** Timestamped backups
- **Verification:** Checksum validation

### Configuration Files Backed Up

```
config/
├── .env.backup_20260311_100000
├── .env.production.backup_20260311_100000
├── nginx.conf.backup_20260311_100000
├── gunicorn_config.py.backup_20260311_100000
├── ssl_cert.pem.backup_20260311_100000
└── ssl_key.pem.backup_20260311_100000
```

## 📝 Log Backup Script

### Usage

```bash
# Archive all logs
bash backup_logs.sh

# Archive specific log type
bash backup_logs.sh app

# Archive with compression
bash backup_logs.sh --compress gzip
```

### Features

- **Application Logs:** API logs, worker logs
- **Security Logs:** Authentication, authorization
- **Audit Logs:** User actions, changes
- **Compression:** Reduce storage usage
- **Rotation:** Automatic log rotation
- **Retention:** Configurable retention

### Log Archives

```
logs/
├── app/
│   ├── app_logs_20260311.tar.gz
│   ├── app_logs_20260310.tar.gz
│   └── app_logs_20260309.tar.gz
├── security/
│   ├── security_logs_20260311.tar.gz
│   ├── security_logs_20260310.tar.gz
│   └── security_logs_20260309.tar.gz
└── audit/
    ├── audit_logs_20260311.tar.gz
    ├── audit_logs_20260310.tar.gz
    └── audit_logs_20260309.tar.gz
```

## ☁️ S3 Archival Script

### Usage

```bash
# Archive all backups to S3
bash archive_to_s3.sh

# Archive specific backup type
bash archive_to_s3.sh full

# Archive with custom S3 bucket
bash archive_to_s3.sh --bucket my-backups
```

### Features

- **AWS S3 Integration:** Upload to S3
- **Encryption:** Server-side encryption
- **Versioning:** S3 versioning support
- **Lifecycle:** Automatic archival policies
- **Verification:** Upload verification
- **Cleanup:** Local cleanup after upload

### Environment Variables

```bash
# Required
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
export S3_BUCKET="ai-study-partner-backups"

# Optional
export S3_PREFIX="backups/"
export S3_STORAGE_CLASS="GLACIER"  # STANDARD, GLACIER, DEEP_ARCHIVE
export S3_ENCRYPTION="AES256"
```

### S3 Structure

```
s3://ai-study-partner-backups/
├── backups/
│   ├── full/
│   │   ├── db_backup_20260311_000000.sql.gz
│   │   └── db_backup_20260310_000000.sql.gz
│   ├── incremental/
│   │   ├── db_backup_20260311_060000
│   │   └── db_backup_20260311_120000
│   ├── config/
│   │   ├── .env.backup_20260311_100000
│   │   └── nginx.conf.backup_20260311_100000
│   └── logs/
│       ├── app_logs_20260311.tar.gz
│       └── security_logs_20260311.tar.gz
```

## 🔄 Database Restore Script

### Usage

```bash
# Restore latest full backup
bash restore_database.sh latest

# Restore specific backup
bash restore_database.sh database/full/db_backup_20260310_000000.sql.gz

# Test restore (no changes applied)
bash restore_database.sh database/full/db_backup_20260310_000000.sql.gz --test

# Restore to specific time
bash restore_database.sh --time "2026-03-10 14:30:00"
```

### Features

- **Latest Restore:** Quick restore of latest backup
- **Specific Restore:** Restore any backup file
- **Test Mode:** Verify restore without applying
- **PITR:** Point-in-time recovery
- **Verification:** Integrity checks
- **Rollback:** Automatic backup before restore
- **Logging:** Detailed restore logs

### Restore Process

1. Verify backup integrity (SHA256)
2. Create backup of current database
3. Decompress backup if needed
4. Restore to database
5. Verify restored data
6. Log completion

### Output

```
[2026-03-11 10:00:00] Starting database restore...
[2026-03-11 10:00:00] Database: study_partner
[2026-03-11 10:00:00] Host: localhost
[2026-03-11 10:00:00] Backup file: database/full/db_backup_20260310_000000.sql.gz
[2026-03-11 10:00:01] ✓ Backup integrity verified
[2026-03-11 10:00:05] ✓ Current database backed up
[2026-03-11 10:00:06] ✓ Backup decompressed
[2026-03-11 10:05:30] ✓ Database restore completed
[2026-03-11 10:05:31] ✓ Restored database contains 1500 users
[2026-03-11 10:05:31] ✅ Restore completed successfully
```

## 🔐 Configuration Restore Script

### Usage

```bash
# Restore all configurations
bash restore_config.sh

# Restore specific config
bash restore_config.sh .env

# Restore from specific backup
bash restore_config.sh --backup config/.env.backup_20260310_100000
```

### Features

- **Selective Restore:** Restore specific configs
- **Verification:** Checksum validation
- **Backup:** Backup current config before restore
- **Rollback:** Easy rollback to previous version
- **Logging:** Detailed restore logs

## 🧹 Cleanup Script

### Usage

```bash
# Remove backups older than 30 days
bash cleanup_old_backups.sh 30

# Remove backups older than 7 days
bash cleanup_old_backups.sh 7

# Dry run (show what would be deleted)
bash cleanup_old_backups.sh 30 --dry-run
```

### Features

- **Age-based Cleanup:** Remove old backups
- **Dry Run:** Preview deletions
- **Logging:** Track deleted files
- **Safety:** Verify before deletion
- **Retention:** Respect retention policies

### Retention Policies

```
Backup Type         | Retention | Cleanup Command
--------------------|-----------|------------------
Full Backups        | 30 days   | cleanup_old_backups.sh 30
Incremental Backups | 7 days    | cleanup_old_backups.sh 7
Config Backups      | 90 days   | cleanup_old_backups.sh 90
Log Archives        | 90 days   | cleanup_old_backups.sh 90
```

## ✅ Verification Script

### Usage

```bash
# Verify specific backup
bash verify_backup.sh database/full/db_backup_20260310_000000.sql.gz

# Verify all backups
bash verify_backup.sh --all

# Verify and test restore
bash verify_backup.sh database/full/db_backup_20260310_000000.sql.gz --test-restore
```

### Features

- **Checksum Verification:** SHA256 validation
- **File Integrity:** Check file completeness
- **Restore Test:** Test restore process
- **Detailed Report:** Comprehensive verification report
- **Logging:** Track verification results

### Verification Checks

- ✅ File exists and is readable
- ✅ File size is reasonable
- ✅ Checksum matches
- ✅ File is not corrupted
- ✅ Restore test succeeds
- ✅ Restored data is valid

## 🔄 Cron Job Setup

### View Current Cron Jobs

```bash
crontab -l
```

### Edit Cron Jobs

```bash
crontab -e
```

### Recommended Cron Schedule

```cron
# Full backup daily at midnight
0 0 * * * cd /opt/ai-study-partner/backend/backups && bash scripts/backup_database.sh full

# Incremental backups every 6 hours
0 6,12,18 * * * cd /opt/ai-study-partner/backend/backups && bash scripts/backup_database.sh incremental

# Config backup on deployment (triggered by deployment script)
# Manual trigger: bash scripts/backup_config.sh

# Log backup daily at 11 PM
0 23 * * * cd /opt/ai-study-partner/backend/backups && bash scripts/backup_logs.sh

# Cleanup old backups daily at 2 AM
0 2 * * * cd /opt/ai-study-partner/backend/backups && bash scripts/cleanup_old_backups.sh 30

# Verify backups daily at 3 AM
0 3 * * * cd /opt/ai-study-partner/backend/backups && bash scripts/verify_backup.sh --all

# Archive to S3 weekly on Sunday at 4 AM
0 4 * * 0 cd /opt/ai-study-partner/backend/backups && bash scripts/archive_to_s3.sh
```

## 📊 Monitoring and Alerts

### Backup Monitoring

Monitor these metrics:
- Backup success/failure rate
- Backup duration
- Backup size
- Storage usage
- Restore success rate

### Alert Conditions

Send alerts if:
- Backup fails
- Backup takes >1 hour
- Backup size exceeds 2GB
- Storage usage >80%
- Restore test fails
- Verification fails

### Monitoring Tools

- **CloudWatch:** AWS monitoring
- **Datadog:** Third-party monitoring
- **Prometheus:** Open-source monitoring
- **ELK Stack:** Log aggregation

## 🧪 Testing Procedures

### Weekly Backup Test

```bash
# 1. Create test backup
bash backup_database.sh test

# 2. Verify backup integrity
bash verify_backup.sh database/full/db_backup_*.sql.gz

# 3. Test restore
bash restore_database.sh database/full/db_backup_*.sql.gz --test
```

### Monthly Full Restore Test

```bash
# 1. Restore latest backup
bash restore_database.sh latest

# 2. Verify data integrity
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 3. Test application functionality
# Run application tests

# 4. Document results
# Update BACKUP_VERIFICATION.md
```

### Quarterly Disaster Recovery Drill

```bash
# 1. Simulate data loss
# Stop application
# Delete database

# 2. Execute recovery procedures
bash restore_database.sh latest

# 3. Verify system functionality
# Start application
# Run smoke tests

# 4. Document lessons learned
# Update DISASTER_RECOVERY.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Set database URL
export DATABASE_URL="postgresql://user:password@localhost/study_partner"

# Set AWS credentials (for S3 archival)
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
export S3_BUCKET="ai-study-partner-backups"
```

### 2. Create First Backup

```bash
# Create full backup
bash backup_database.sh full

# Verify backup
bash verify_backup.sh database/full/db_backup_*.sql.gz

# List backups
ls -lh database/full/
```

### 3. Setup Cron Jobs

```bash
# Edit crontab
crontab -e

# Add backup schedule (see Cron Job Setup section)
```

### 4. Test Restore

```bash
# Test restore process
bash restore_database.sh database/full/db_backup_*.sql.gz --test

# Verify restore works
bash restore_database.sh latest --test
```

## 📞 Troubleshooting

### Backup Fails

**Problem:** Backup script exits with error

**Solutions:**
1. Check DATABASE_URL is set: `echo $DATABASE_URL`
2. Verify database connection: `psql -U postgres -h localhost -c "SELECT 1"`
3. Check disk space: `df -h`
4. Check backup directory permissions: `ls -la backups/`
5. Review backup logs: `tail -f ../logs/backup.log`

### Restore Fails

**Problem:** Restore script exits with error

**Solutions:**
1. Verify backup file exists: `ls -la database/full/`
2. Verify backup integrity: `bash verify_backup.sh <backup_file>`
3. Check database permissions: `psql -U postgres -h localhost -l`
4. Check disk space: `df -h`
5. Review restore logs: `tail -f ../logs/restore.log`

### Storage Full

**Problem:** Backup directory is full

**Solutions:**
1. Check storage usage: `du -sh ../backups/`
2. Remove old backups: `bash cleanup_old_backups.sh 30`
3. Archive to S3: `bash archive_to_s3.sh`
4. Increase storage capacity

### Verification Fails

**Problem:** Backup verification fails

**Solutions:**
1. Check checksum file exists: `ls -la database/full/*.sha256`
2. Verify checksum: `sha256sum -c database/full/db_backup_*.sql.gz.sha256`
3. Check file integrity: `file database/full/db_backup_*.sql.gz`
4. Recreate backup: `bash backup_database.sh full`

## 📚 Additional Resources

- [DISASTER_RECOVERY.md](../recovery/DISASTER_RECOVERY.md) - Disaster recovery guide
- [POINT_IN_TIME_RECOVERY.md](../recovery/POINT_IN_TIME_RECOVERY.md) - PITR guide
- [BACKUP_VERIFICATION.md](../recovery/BACKUP_VERIFICATION.md) - Verification procedures
- [../README.md](../README.md) - Main backup documentation

---

**Status:** ✅ Production Ready  
**Last Updated:** March 11, 2026  
**Version:** 1.0.0
