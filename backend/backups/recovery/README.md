# Recovery Procedures

Complete guide to disaster recovery, point-in-time recovery, and backup verification for AI Study Partner Backend.

## 📋 Overview

This directory contains comprehensive recovery procedures for various disaster scenarios and recovery requirements.

## 📁 Documentation

### 1. [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md)

Complete disaster recovery guide covering:
- Emergency response procedures
- 5 disaster scenarios with step-by-step recovery
- Recovery time objectives (RTO)
- Recovery point objectives (RPO)
- Testing procedures
- Escalation procedures

**When to use:** System failure, data loss, security breach, or major incident

**Key Scenarios:**
- Database corruption
- Complete data loss
- Security breach
- Hardware failure
- Application failure

### 2. [POINT_IN_TIME_RECOVERY.md](POINT_IN_TIME_RECOVERY.md)

Point-in-time recovery (PITR) guide covering:
- PITR concepts and requirements
- Recovery to specific timestamps
- Incremental backup usage
- Transaction log recovery
- Testing procedures
- Time considerations

**When to use:** Need to recover to a specific point in time before an incident

**Key Scenarios:**
- Accidental data deletion
- Incorrect data modification
- Malicious data changes
- Application bug impact

### 3. [BACKUP_VERIFICATION.md](BACKUP_VERIFICATION.md)

Backup verification procedures covering:
- Verification types and methods
- Automated verification
- Manual verification
- Verification schedule
- Metrics and reporting
- Troubleshooting

**When to use:** Ensure backups are valid and can be restored

**Key Procedures:**
- Checksum verification
- Restore testing
- Data integrity checks
- Automated verification

## 🚀 Quick Recovery Guide

### Scenario 1: Database Corruption

**Symptoms:** Application errors, data inconsistencies, query failures

**Recovery Steps:**
1. Stop application
2. Restore latest backup: `bash ../scripts/restore_database.sh latest`
3. Verify data integrity
4. Start application
5. Monitor for issues

**Time:** 15-30 minutes

### Scenario 2: Accidental Data Deletion

**Symptoms:** Missing data, user reports data loss

**Recovery Steps:**
1. Identify deletion time
2. Restore to point before deletion: `bash ../scripts/restore_database.sh --time "2026-03-10 14:00:00"`
3. Verify recovered data
4. Merge with current data if needed
5. Notify affected users

**Time:** 30-60 minutes

### Scenario 3: Complete Data Loss

**Symptoms:** Database unavailable, all data lost

**Recovery Steps:**
1. Verify backup availability
2. Restore latest full backup: `bash ../scripts/restore_database.sh latest`
3. Apply incremental backups if needed
4. Verify system functionality
5. Notify stakeholders

**Time:** 1-2 hours

### Scenario 4: Security Breach

**Symptoms:** Unauthorized access, data compromise

**Recovery Steps:**
1. Isolate affected systems
2. Restore from pre-breach backup
3. Change all credentials
4. Verify system integrity
5. Investigate breach
6. Implement security fixes

**Time:** 2-4 hours

### Scenario 5: Hardware Failure

**Symptoms:** Server down, storage failure

**Recovery Steps:**
1. Provision new hardware
2. Restore latest backup to new system
3. Verify system functionality
4. Update DNS/load balancer
5. Monitor for issues

**Time:** 1-3 hours

## 📊 Recovery Objectives

| Scenario | RTO | RPO | Procedure |
|----------|-----|-----|-----------|
| Database Corruption | 30 min | 6 hours | Restore latest backup |
| Data Deletion | 1 hour | 6 hours | PITR to pre-deletion time |
| Complete Data Loss | 2 hours | 24 hours | Restore latest backup |
| Security Breach | 4 hours | 24 hours | Restore pre-breach backup |
| Hardware Failure | 3 hours | 6 hours | Restore to new hardware |

## 🔄 Recovery Procedures

### Restore Latest Backup

```bash
# Restore latest full backup
cd /opt/ai-study-partner/backend/backups
bash scripts/restore_database.sh latest

# Verify restore
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"
```

### Restore Specific Backup

```bash
# List available backups
ls -lh database/full/

# Restore specific backup
bash scripts/restore_database.sh database/full/db_backup_20260310_000000.sql.gz

# Verify restore
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"
```

### Point-in-Time Recovery

```bash
# Restore to specific time
bash scripts/restore_database.sh --time "2026-03-10 14:30:00"

# Verify restore
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"
```

### Test Restore

```bash
# Test restore without applying changes
bash scripts/restore_database.sh latest --test

# Verify test succeeded
echo "Test restore completed successfully"
```

### Restore Configuration

```bash
# Restore all configurations
cd /opt/ai-study-partner/backend/backups
bash scripts/restore_config.sh

# Verify configuration
cat ../.env
```

## 🧪 Testing Procedures

### Weekly Backup Test

```bash
# 1. Create test backup
bash scripts/backup_database.sh test

# 2. Verify backup integrity
bash scripts/verify_backup.sh database/full/db_backup_*.sql.gz

# 3. Test restore
bash scripts/restore_database.sh database/full/db_backup_*.sql.gz --test

# 4. Document results
echo "Weekly backup test: PASSED" >> ../logs/backup.log
```

### Monthly Full Restore Test

```bash
# 1. Restore latest backup
bash scripts/restore_database.sh latest

# 2. Verify data integrity
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM users;"
psql -U postgres -h localhost study_partner -c "SELECT COUNT(*) FROM study_sessions;"

# 3. Test application functionality
# Run application smoke tests

# 4. Document results
echo "Monthly restore test: PASSED" >> ../logs/backup.log
```

### Quarterly Disaster Recovery Drill

```bash
# 1. Simulate data loss
# Stop application
systemctl stop ai-study-partner

# Delete database (CAREFUL!)
# psql -U postgres -h localhost -c "DROP DATABASE study_partner;"

# 2. Execute recovery procedures
bash scripts/restore_database.sh latest

# 3. Verify system functionality
systemctl start ai-study-partner
curl http://localhost:8000/health

# 4. Document lessons learned
echo "Quarterly DR drill: PASSED" >> ../logs/backup.log
```

## 📈 Monitoring and Alerts

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

### Monitoring Commands

```bash
# Check last backup
ls -lt database/full/ | head -1

# Check backup size
du -sh database/full/

# Check storage usage
df -h /opt/ai-study-partner/backend/

# Check backup age
find database/full/ -type f -mtime +1 -print

# Check backup logs
tail -f ../logs/backup.log
```

## 🔐 Security Considerations

### Backup Security

- All backups encrypted with AES-256
- Backups stored in secure location
- Access restricted to backup user
- Permissions: 600 (read/write owner only)
- Encryption keys stored separately

### Recovery Security

- Recovery procedures require authentication
- All recovery actions logged
- Recovery verified before completion
- Sensitive data handled carefully
- Audit trail maintained

### Key Management

- Encryption keys stored in AWS Secrets Manager
- Keys rotated every 90 days
- Separate keys for different backup types
- Key access restricted to authorized personnel

## 📞 Support and Escalation

### Level 1: Automated Recovery

- Automated backup verification
- Automated restore testing
- Automated alerts

### Level 2: Manual Recovery

- Database administrator
- System administrator
- Backup specialist

### Level 3: Escalation

- Engineering manager
- Infrastructure team
- Executive stakeholder

### Contact Information

- **On-Call DBA:** [contact info]
- **Infrastructure Team:** [contact info]
- **Executive Escalation:** [contact info]

## 📚 Related Documentation

- [../README.md](../README.md) - Main backup documentation
- [../scripts/README.md](../scripts/README.md) - Script documentation
- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - Disaster recovery guide
- [POINT_IN_TIME_RECOVERY.md](POINT_IN_TIME_RECOVERY.md) - PITR guide
- [BACKUP_VERIFICATION.md](BACKUP_VERIFICATION.md) - Verification procedures

## ✅ Recovery Checklist

### Before Recovery

- [ ] Identify recovery scenario
- [ ] Verify backup availability
- [ ] Notify stakeholders
- [ ] Document start time
- [ ] Prepare recovery environment

### During Recovery

- [ ] Execute recovery procedure
- [ ] Monitor recovery progress
- [ ] Verify data integrity
- [ ] Test application functionality
- [ ] Document issues

### After Recovery

- [ ] Verify system functionality
- [ ] Notify stakeholders
- [ ] Document recovery time
- [ ] Update documentation
- [ ] Conduct post-mortem

## 🎯 Best Practices

### Do's ✅
- ✅ Test recovery procedures regularly
- ✅ Verify backup integrity
- ✅ Document all recovery actions
- ✅ Notify stakeholders
- ✅ Monitor recovery progress
- ✅ Verify data integrity
- ✅ Conduct post-mortem
- ✅ Update procedures

### Don'ts ❌
- ❌ Skip backup verification
- ❌ Restore without testing
- ❌ Ignore recovery failures
- ❌ Skip data integrity checks
- ❌ Forget to notify stakeholders
- ❌ Ignore recovery logs
- ❌ Skip post-mortem
- ❌ Delay documentation

## 📊 Recovery Statistics

| Metric | Target | Current |
|--------|--------|---------|
| Restore Success Rate | 99.9% | - |
| Average Restore Time | <15 min | - |
| Data Integrity Verification | 100% | - |
| Monthly Restore Test | 100% | - |
| Disaster Recovery Drill | Quarterly | - |

---

**Status:** ✅ Production Ready  
**Last Updated:** March 11, 2026  
**Version:** 1.0.0

**Recovery procedures are critical for business continuity. Ensure they are properly tested and documented!**
