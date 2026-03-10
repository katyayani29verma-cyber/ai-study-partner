# Backup Operations - Quick Reference Guide

Fast reference for common backup and recovery operations.

## 🚀 Common Commands

### Backup Operations

```bash
# Full database backup
cd /opt/ai-study-partner/backend/backups
bash scripts/backup_database.sh full

# Incremental backup
bash scripts/backup_database.sh incremental

# Configuration backup
bash scripts/backup_config.sh

# Log archival
bash scripts/backup_logs.sh

# Archive to S3
bash scripts/archive_to_s3.sh
```

### Restore Operations

```bash
# Restore latest backup
bash scripts/restore_database.sh latest

# Restore specific backup
bash scripts/restore_database.sh database/full/db_backup_20260310_000000.sql.gz

# Test restore (no changes)
bash scripts/restore_database.sh latest --test

# Point-in-time recovery
bash scripts/restore_database.sh --time "2026-03-10 14:30:00"

# Restore configuration
bash scripts/restore_config.sh
```

### Verification Operations

```bash
# Verify specific backup
bash scripts/verify_backup.sh database/full/db_backup_*.sql.gz

# Verify all backups
bash scripts/verify_backup.sh --all

# Test restore verification
bash scripts/verify_backup.sh database/full/db_backup_*.sql.gz --test-restore
```

### Maintenance Operations

```bash
# Cleanup backups older than 30 days
bash scripts/cleanup_old_backups.sh 30

# Cleanup with dry-run (preview)
bash scripts/cleanup_old_backups.sh 30 --dry-run

# List backups
ls -lh database/full/
ls -lh database/incremental/

# Check storage usage
du -sh database/
du -sh ../backups/
```

## 📋 Backup Status

```bash
# Check last backup
ls -lt database/full/ | head -1

# Check backup size
du -h database/full/db_backup_*.sql.gz | tail -1

# Check backup age
find database/full/ -type f -mtime +1 -print

# Check storage usage
df -h /opt/ai-study-partner/backend/

# View backup logs
tail -f ../logs/backup.log
```

## 🔄 Recovery Scenarios

### Scenario 1: Database Corruption (15-30 min)

```bash
# 1. Stop application
systemctl stop ai-study-partner

# 2. Restore latest backup
bash scripts/restore_database.sh latest

# 3. Verify data
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 4. Start application
systemctl start ai-study-partner

# 5. Monitor
curl http://localhost:8000/health
```

### Scenario 2: Accidental Data Deletion (30-60 min)

```bash
# 1. Identify deletion time (e.g., 2026-03-10 14:00:00)

# 2. Restore to point before deletion
bash scripts/restore_database.sh --time "2026-03-10 13:59:00"

# 3. Verify recovered data
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 4. Merge with current data if needed

# 5. Notify affected users
```

### Scenario 3: Complete Data Loss (1-2 hours)

```bash
# 1. Verify backup availability
ls -lh database/full/

# 2. Restore latest backup
bash scripts/restore_database.sh latest

# 3. Verify system functionality
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 4. Start application
systemctl start ai-study-partner

# 5. Notify stakeholders
```

### Scenario 4: Security Breach (2-4 hours)

```bash
# 1. Isolate affected systems
systemctl stop ai-study-partner

# 2. Restore from pre-breach backup
bash scripts/restore_database.sh database/full/db_backup_20260309_000000.sql.gz

# 3. Change all credentials
# Update .env file with new passwords

# 4. Verify system integrity
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 5. Start application
systemctl start ai-study-partner

# 6. Investigate breach
```

### Scenario 5: Hardware Failure (1-3 hours)

```bash
# 1. Provision new hardware

# 2. Restore latest backup to new system
bash scripts/restore_database.sh latest

# 3. Verify system functionality
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# 4. Update DNS/load balancer

# 5. Monitor for issues
```

## 🧪 Testing Procedures

### Weekly Backup Test (5 min)

```bash
# 1. Create test backup
bash scripts/backup_database.sh test

# 2. Verify integrity
bash scripts/verify_backup.sh database/full/db_backup_*.sql.gz

# 3. Test restore
bash scripts/restore_database.sh database/full/db_backup_*.sql.gz --test

# 4. Document
echo "Weekly test: PASSED" >> ../logs/backup.log
```

### Monthly Full Restore Test (30 min)

```bash
# 1. Restore latest backup
bash scripts/restore_database.sh latest

# 2. Verify data
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM study_sessions;"

# 3. Test application
curl http://localhost:8000/health

# 4. Document
echo "Monthly restore test: PASSED" >> ../logs/backup.log
```

### Quarterly Disaster Recovery Drill (1-2 hours)

```bash
# 1. Simulate data loss
systemctl stop ai-study-partner
# psql -U postgres -h localhost -c "DROP DATABASE study_partner;"

# 2. Execute recovery
bash scripts/restore_database.sh latest

# 3. Verify functionality
systemctl start ai-study-partner
curl http://localhost:8000/health

# 4. Document
echo "Quarterly DR drill: PASSED" >> ../logs/backup.log
```

## 📊 Monitoring

### Daily Checks

```bash
# Check backup completed
ls -lt database/full/ | head -1

# Check backup size
du -h database/full/db_backup_*.sql.gz | tail -1

# Check for errors
grep ERROR ../logs/backup.log | tail -5

# Check storage
df -h /opt/ai-study-partner/backend/
```

### Weekly Checks

```bash
# Verify backup integrity
bash scripts/verify_backup.sh --all

# Check storage usage
du -sh database/
du -sh ../backups/

# Review backup logs
tail -50 ../logs/backup.log

# Test restore
bash scripts/restore_database.sh latest --test
```

### Monthly Checks

```bash
# Full restore test
bash scripts/restore_database.sh latest

# Verify data integrity
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"

# Test application
curl http://localhost:8000/health

# Review metrics
# Check backup success rate
# Check restore success rate
# Check storage trends
```

## 🔐 Security Checks

```bash
# Verify backup permissions
ls -la database/full/

# Verify encryption
file database/full/db_backup_*.sql.gz

# Verify checksums
sha256sum -c database/full/db_backup_*.sql.gz.sha256

# Check backup logs for errors
grep -i error ../logs/backup.log
```

## 🚨 Emergency Procedures

### Backup Failed

```bash
# 1. Check logs
tail -f ../logs/backup.log

# 2. Verify database connection
psql -U postgres -h localhost -c "SELECT 1"

# 3. Check disk space
df -h

# 4. Check permissions
ls -la database/

# 5. Retry backup
bash scripts/backup_database.sh full
```

### Restore Failed

```bash
# 1. Check logs
tail -f ../logs/restore.log

# 2. Verify backup file
ls -la database/full/db_backup_*.sql.gz

# 3. Verify backup integrity
bash scripts/verify_backup.sh database/full/db_backup_*.sql.gz

# 4. Check database permissions
psql -U postgres -h localhost -l

# 5. Retry restore
bash scripts/restore_database.sh latest --test
```

### Storage Full

```bash
# 1. Check usage
du -sh ../backups/

# 2. Remove old backups
bash scripts/cleanup_old_backups.sh 30

# 3. Archive to S3
bash scripts/archive_to_s3.sh

# 4. Verify space freed
df -h /opt/ai-study-partner/backend/
```

## 📞 Quick Help

### Environment Setup

```bash
# Set database URL
export DATABASE_URL="postgresql://user:password@localhost/study_partner"

# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
export S3_BUCKET="ai-study-partner-backups"
```

### Cron Jobs

```bash
# View cron jobs
crontab -l

# Edit cron jobs
crontab -e

# Install cron jobs
crontab cron_jobs.conf
```

### Logs

```bash
# View backup logs
tail -f ../logs/backup.log

# View restore logs
tail -f ../logs/restore.log

# Search logs
grep "ERROR" ../logs/backup.log
grep "FAILED" ../logs/backup.log
```

## 📚 Documentation Links

- [README.md](README.md) - Complete backup documentation
- [scripts/README.md](scripts/README.md) - Script documentation
- [recovery/README.md](recovery/README.md) - Recovery procedures
- [DISASTER_RECOVERY.md](recovery/DISASTER_RECOVERY.md) - Disaster recovery
- [POINT_IN_TIME_RECOVERY.md](recovery/POINT_IN_TIME_RECOVERY.md) - PITR
- [BACKUP_VERIFICATION.md](recovery/BACKUP_VERIFICATION.md) - Verification

## ⏱️ Time Estimates

| Operation | Time |
|-----------|------|
| Full backup | 5-30 min |
| Incremental backup | 1-5 min |
| Config backup | <1 min |
| Log archival | 1-5 min |
| Backup verification | 2-10 min |
| Restore test | 5-15 min |
| Full restore | 10-30 min |
| PITR restore | 15-45 min |
| Cleanup old backups | 1-5 min |
| S3 archival | 5-30 min |

## 🎯 Key Metrics

| Metric | Target |
|--------|--------|
| Backup Success Rate | 99.9% |
| Average Backup Time | <30 min |
| Average Restore Time | <15 min |
| Backup Verification | 100% |
| Monthly Restore Test | 100% |
| Disaster Recovery Drill | Quarterly |

---

**Quick Reference Version:** 1.0.0  
**Last Updated:** March 11, 2026

**For detailed information, see the full documentation in README.md and recovery/ directory.**
