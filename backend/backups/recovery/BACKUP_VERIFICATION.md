# Backup Verification Guide

Comprehensive guide for verifying backup integrity and recoverability.

## 📋 Overview

Regular backup verification ensures that backups are valid and can be restored when needed. This guide covers all verification procedures.

## 🔍 Verification Types

### 1. File Integrity Verification

**Purpose:** Ensure backup file is not corrupted

```bash
# Verify checksum
bash ../scripts/verify_backup.sh backups/database/full/db_backup_20260310_000000.sql.gz

# Manual checksum verification
sha256sum -c backups/database/full/db_backup_20260310_000000.sql.gz.sha256

# Verify gzip integrity
gzip -t backups/database/full/db_backup_20260310_000000.sql.gz
```

### 2. Content Verification

**Purpose:** Ensure backup contains valid data

```bash
# Check backup size
du -h backups/database/full/db_backup_20260310_000000.sql.gz

# Check line count
gunzip -c backups/database/full/db_backup_20260310_000000.sql.gz | wc -l

# Check for SQL statements
gunzip -c backups/database/full/db_backup_20260310_000000.sql.gz | grep -c "INSERT INTO"
```

### 3. Restore Verification

**Purpose:** Ensure backup can be restored successfully

```bash
# Test restore without applying
bash ../scripts/restore_database.sh backups/database/full/db_backup_20260310_000000.sql.gz --test

# Verify test database
psql -U postgres -h localhost -d study_partner_test -c "SELECT COUNT(*) FROM users;"
```

### 4. Data Integrity Verification

**Purpose:** Ensure restored data is correct

```bash
# Compare record counts
ORIGINAL_COUNT=$(psql -U postgres -h localhost -d study_partner -c "SELECT COUNT(*) FROM users;" | tail -1)
RESTORED_COUNT=$(psql -U postgres -h localhost -d study_partner_test -c "SELECT COUNT(*) FROM users;" | tail -1)

if [ "$ORIGINAL_COUNT" = "$RESTORED_COUNT" ]; then
    echo "Record count matches"
else
    echo "Record count mismatch!"
fi

# Compare checksums
psql -U postgres -h localhost -d study_partner -c "SELECT MD5(STRING_AGG(id::text, ',')) FROM users;" > /tmp/original.md5
psql -U postgres -h localhost -d study_partner_test -c "SELECT MD5(STRING_AGG(id::text, ',')) FROM users;" > /tmp/restored.md5

diff /tmp/original.md5 /tmp/restored.md5
```

## 📊 Verification Schedule

### Daily Verification
- [ ] Backup completed successfully
- [ ] Backup file size reasonable
- [ ] Checksum verification passed
- [ ] No errors in backup logs

### Weekly Verification
- [ ] Test restore procedure
- [ ] Verify restored data
- [ ] Check backup storage
- [ ] Review backup logs

### Monthly Verification
- [ ] Full restore test
- [ ] Data integrity check
- [ ] Application functionality test
- [ ] Document results

### Quarterly Verification
- [ ] Disaster recovery drill
- [ ] Test all recovery procedures
- [ ] Verify backup strategy
- [ ] Update documentation

## 🧪 Verification Procedures

### Automated Verification

```bash
#!/bin/bash
# Automated backup verification script

BACKUP_DIR="backups/database/full"
LOG_FILE="logs/verification.log"

for backup in $(ls -t $BACKUP_DIR/*.sql.gz | head -5); do
    echo "Verifying $backup..." | tee -a $LOG_FILE
    
    # Verify checksum
    if sha256sum -c "${backup}.sha256" >> $LOG_FILE 2>&1; then
        echo "✓ Checksum passed" | tee -a $LOG_FILE
    else
        echo "✗ Checksum failed" | tee -a $LOG_FILE
        continue
    fi
    
    # Verify gzip
    if gzip -t "$backup" >> $LOG_FILE 2>&1; then
        echo "✓ Gzip format valid" | tee -a $LOG_FILE
    else
        echo "✗ Gzip format invalid" | tee -a $LOG_FILE
        continue
    fi
    
    # Verify content
    LINE_COUNT=$(gunzip -c "$backup" | wc -l)
    if [ $LINE_COUNT -gt 100 ]; then
        echo "✓ Content valid ($LINE_COUNT lines)" | tee -a $LOG_FILE
    else
        echo "✗ Content invalid ($LINE_COUNT lines)" | tee -a $LOG_FILE
    fi
done
```

### Manual Verification

```bash
# 1. Select backup to verify
BACKUP_FILE="backups/database/full/db_backup_20260310_000000.sql.gz"

# 2. Run verification script
bash scripts/verify_backup.sh "$BACKUP_FILE"

# 3. Review results
echo "Verification complete. Check logs/backup.log for details."
```

### Restore Test Verification

```bash
# 1. Test restore
bash scripts/restore_database.sh "$BACKUP_FILE" --test

# 2. If test passes, verify data
psql -U postgres -h localhost -d study_partner_test -c "SELECT COUNT(*) FROM users;"

# 3. Check specific data
psql -U postgres -h localhost -d study_partner_test -c "SELECT * FROM users LIMIT 5;"

# 4. Cleanup test database
psql -U postgres -h localhost -c "DROP DATABASE study_partner_test;"
```

## 📈 Verification Metrics

### Success Criteria

| Metric | Target | Threshold |
|--------|--------|-----------|
| Checksum Pass Rate | 100% | >99% |
| Gzip Integrity | 100% | >99% |
| Content Validity | 100% | >99% |
| Restore Success | 100% | >99% |
| Data Integrity | 100% | >99% |

### Monitoring

```bash
# Track verification results
tail -100 logs/verification.log

# Count successes
grep "✓" logs/verification.log | wc -l

# Count failures
grep "✗" logs/verification.log | wc -l

# Calculate success rate
TOTAL=$(grep -c "Verifying" logs/verification.log)
SUCCESS=$(grep -c "✓" logs/verification.log)
RATE=$((SUCCESS * 100 / TOTAL))
echo "Success rate: $RATE%"
```

## 🔐 Security Verification

### Encryption Verification

```bash
# Verify backup is encrypted
file backups/database/full/db_backup_20260310_000000.sql.gz

# Check encryption key
echo $BACKUP_ENCRYPTION_KEY | wc -c

# Verify key rotation
ls -la backups/config/.env.backup | head -5
```

### Access Control Verification

```bash
# Check backup file permissions
ls -la backups/database/full/db_backup_20260310_000000.sql.gz

# Should be: -rw------- (600)
# Owner: backup user
# Group: backup group

# Verify backup directory permissions
ls -la backups/database/full/

# Should be: drwx------ (700)
```

### Audit Log Verification

```bash
# Check backup audit logs
grep "backup" logs/audit.log | tail -20

# Verify backup access
grep "backups/" logs/audit.log | tail -20

# Check for unauthorized access
grep "DENIED" logs/audit.log | grep "backups/"
```

## 📋 Verification Checklist

### Pre-Verification
- [ ] Backup file exists
- [ ] Backup file is readable
- [ ] Checksum file exists
- [ ] Sufficient disk space
- [ ] Database is accessible

### During Verification
- [ ] Checksum verification passed
- [ ] Gzip format valid
- [ ] Content is valid
- [ ] Restore test successful
- [ ] Data integrity verified

### Post-Verification
- [ ] All checks passed
- [ ] Results documented
- [ ] Alerts configured
- [ ] Issues resolved
- [ ] Team notified

## 🆘 Troubleshooting

### Checksum Verification Failed

**Cause:** Backup file corrupted or checksum file missing

**Solution:**
```bash
# Regenerate checksum
sha256sum backups/database/full/db_backup_20260310_000000.sql.gz > backups/database/full/db_backup_20260310_000000.sql.gz.sha256

# Or restore from backup
bash scripts/restore_database.sh latest
```

### Gzip Format Invalid

**Cause:** Backup file corrupted during transfer

**Solution:**
```bash
# Try to repair
gunzip -t backups/database/full/db_backup_20260310_000000.sql.gz

# If repair fails, restore from backup
bash scripts/restore_database.sh latest
```

### Content Invalid

**Cause:** Backup file too small or incomplete

**Solution:**
```bash
# Check backup size
du -h backups/database/full/db_backup_20260310_000000.sql.gz

# If too small, re-backup
bash scripts/backup_database.sh full

# Or restore from previous backup
bash scripts/restore_database.sh latest
```

### Restore Test Failed

**Cause:** Backup cannot be restored

**Solution:**
```bash
# Check database space
df -h

# Check database permissions
psql -U postgres -h localhost -c "SELECT 1"

# Try restore again
bash scripts/restore_database.sh "$BACKUP_FILE" --test
```

## 📞 Support

### Documentation
- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - Disaster recovery guide
- [POINT_IN_TIME_RECOVERY.md](POINT_IN_TIME_RECOVERY.md) - PITR guide
- [../scripts/README.md](../scripts/README.md) - Script documentation

### Contacts
- Database Administrator: [contact]
- Operations Lead: [contact]
- Security Officer: [contact]

---

**Status:** ✅ Production Ready  
**Last Updated:** March 10, 2026  
**Version:** 1.0.0  

**Regular backup verification is essential for production systems. Verify backups weekly and test recovery monthly.**
