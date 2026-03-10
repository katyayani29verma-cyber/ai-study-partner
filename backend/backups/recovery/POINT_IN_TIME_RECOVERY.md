# Point-in-Time Recovery (PITR) Guide

Guide for recovering database to a specific point in time.

## 📋 Overview

Point-in-Time Recovery (PITR) allows you to restore the database to any specific moment in time, not just to the time of the last backup.

## 🔄 How PITR Works

### Prerequisites
- Full backup from before the target time
- Transaction logs (WAL files) from backup time to target time
- PostgreSQL PITR support enabled

### Process
1. Restore full backup
2. Apply transaction logs up to target time
3. Verify restored state

## 🚀 PITR Procedures

### Basic PITR

```bash
# Restore to specific time
bash scripts/restore_database.sh --time "2026-03-10 14:30:00"

# Restore to 1 hour ago
bash scripts/restore_database.sh --time "$(date -d '1 hour ago' '+%Y-%m-%d %H:%M:%S')"

# Restore to specific date
bash scripts/restore_database.sh --time "2026-03-10 00:00:00"
```

### PITR with Verification

```bash
# 1. Identify target time
TARGET_TIME="2026-03-10 14:30:00"

# 2. Find appropriate backup
ls -lt backups/database/full/ | head -5

# 3. Restore to target time
bash scripts/restore_database.sh --time "$TARGET_TIME"

# 4. Verify restored state
psql -U postgres -h localhost -d study_partner -c "SELECT * FROM users LIMIT 5;"

# 5. Check recovery time
psql -U postgres -h localhost -d study_partner -c "SELECT MAX(updated_at) FROM users;"
```

### PITR with Test Mode

```bash
# Test restore without applying changes
bash scripts/restore_database.sh --time "2026-03-10 14:30:00" --test

# If test successful, apply actual restore
bash scripts/restore_database.sh --time "2026-03-10 14:30:00"
```

## 📊 PITR Scenarios

### Scenario 1: Recover Deleted User

**Timeline:**
- 14:00 - User account created
- 14:30 - User account deleted (accidental)
- 15:00 - Issue discovered

**Recovery:**

```bash
# 1. Restore to 14:25 (before deletion)
bash scripts/restore_database.sh --time "2026-03-10 14:25:00"

# 2. Verify user exists
psql -U postgres -h localhost -d study_partner -c "SELECT * FROM users WHERE email = 'user@example.com';"

# 3. Export user data
psql -U postgres -h localhost -d study_partner -c "SELECT * FROM users WHERE email = 'user@example.com';" > /tmp/user_data.txt

# 4. Restore current database
bash scripts/restore_database.sh latest

# 5. Manually recreate user or merge data
```

### Scenario 2: Recover Modified Data

**Timeline:**
- 10:00 - Data correct
- 12:00 - Data modified incorrectly
- 14:00 - Issue discovered

**Recovery:**

```bash
# 1. Restore to 11:59 (before modification)
bash scripts/restore_database.sh --time "2026-03-10 11:59:00"

# 2. Export correct data
psql -U postgres -h localhost -d study_partner -c "SELECT * FROM study_materials;" > /tmp/correct_data.sql

# 3. Restore current database
bash scripts/restore_database.sh latest

# 4. Merge correct data back
psql -U postgres -h localhost -d study_partner < /tmp/correct_data.sql
```

### Scenario 3: Recover Before Deployment

**Timeline:**
- 09:00 - Deployment starts
- 09:15 - Deployment completes
- 09:30 - Issues discovered

**Recovery:**

```bash
# 1. Find backup before deployment
ls -lt backups/database/full/ | grep "09:00"

# 2. Restore to pre-deployment state
bash scripts/restore_database.sh --time "2026-03-10 08:59:00"

# 3. Verify system works
curl http://localhost:8000/health

# 4. Investigate deployment issues
# Fix issues and redeploy
```

## 🔍 Finding the Right Recovery Point

### Check Audit Logs

```bash
# View recent audit logs
tail -100 logs/audit.log

# Find specific event
grep "user_deleted" logs/audit.log

# Get timestamp
grep "user_deleted" logs/audit.log | head -1 | cut -d' ' -f1-2
```

### Check Application Logs

```bash
# View recent errors
tail -100 logs/app.log | grep ERROR

# Find specific error
grep "data corruption" logs/app.log

# Get timestamp
grep "data corruption" logs/app.log | head -1 | cut -d' ' -f1-2
```

### Check Database Logs

```bash
# View PostgreSQL logs
tail -100 logs/postgresql.log

# Find specific event
grep "DELETE FROM" logs/postgresql.log

# Get timestamp
grep "DELETE FROM" logs/postgresql.log | head -1 | cut -d' ' -f1-2
```

## ⏰ Time Considerations

### Backup Retention
- Full backups: 30 days
- Incremental backups: 7 days
- PITR possible: Up to 30 days back

### Recovery Time
- Restore full backup: 5-15 minutes
- Apply transaction logs: 1-5 minutes
- Total PITR time: 10-20 minutes

### Data Loss
- RPO (Recovery Point Objective): 6 hours
- Maximum data loss: 6 hours of transactions

## 🧪 Testing PITR

### Monthly PITR Test

```bash
# 1. Choose random time in past week
TARGET_TIME="2026-03-03 14:30:00"

# 2. Test restore
bash scripts/restore_database.sh --time "$TARGET_TIME" --test

# 3. Verify test successful
echo "PITR test completed successfully"

# 4. Document results
echo "PITR test: $TARGET_TIME - SUCCESS" >> logs/pitr_tests.log
```

### Quarterly Full PITR Test

```bash
# 1. Provision test environment
# 2. Restore to 1 month ago
bash scripts/restore_database.sh --time "$(date -d '30 days ago' '+%Y-%m-%d %H:%M:%S')"

# 3. Verify all data
psql -U postgres -h localhost -d study_partner -c "SELECT COUNT(*) FROM users;"

# 4. Test application
curl http://localhost:8000/health

# 5. Document results
```

## 📋 PITR Checklist

### Before PITR
- [ ] Identify target time
- [ ] Verify backup exists
- [ ] Check disk space
- [ ] Notify stakeholders
- [ ] Document timeline

### During PITR
- [ ] Stop application
- [ ] Verify backup integrity
- [ ] Restore to target time
- [ ] Verify restored data
- [ ] Restart application

### After PITR
- [ ] Verify all systems
- [ ] Check data integrity
- [ ] Notify stakeholders
- [ ] Document recovery
- [ ] Review root cause

## 🔐 Security Considerations

### PITR Security
- [ ] Verify backup integrity
- [ ] Check for malware
- [ ] Audit access logs
- [ ] Monitor for anomalies
- [ ] Verify encryption

### Post-PITR
- [ ] Change passwords
- [ ] Rotate keys
- [ ] Review access logs
- [ ] Audit accounts
- [ ] Update policies

## 📞 Support

### Documentation
- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - Disaster recovery guide
- [BACKUP_VERIFICATION.md](BACKUP_VERIFICATION.md) - Verification procedures
- [../scripts/README.md](../scripts/README.md) - Script documentation

### Common Issues

**PITR Failed**
- Check backup integrity: `bash scripts/verify_backup.sh <backup_file>`
- Check disk space: `df -h`
- Check database logs: `tail -100 logs/postgresql.log`

**Restored Data Incorrect**
- Verify target time was correct
- Check audit logs for actual event time
- Try different target time
- Contact database administrator

**PITR Too Slow**
- Check system resources
- Monitor disk I/O
- Check network connectivity
- Consider using incremental backups

---

**Status:** ✅ Production Ready  
**Last Updated:** March 10, 2026  
**Version:** 1.0.0  

**PITR is a powerful recovery tool. Use it carefully and test regularly.**
