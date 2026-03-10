# Disaster Recovery Guide

Complete guide for recovering from system failures and data loss.

## 🚨 Emergency Response

### Immediate Actions (First 5 Minutes)

1. **Assess the Situation**
   - Determine what failed (database, application, server)
   - Check if data is accessible
   - Identify affected users/systems

2. **Notify Stakeholders**
   - Alert operations team
   - Notify management
   - Update status page
   - Prepare communication

3. **Preserve Evidence**
   - Don't restart failed systems
   - Collect error logs
   - Document timeline
   - Take screenshots

### Initial Response (5-15 Minutes)

1. **Activate Disaster Recovery**
   ```bash
   # Start recovery procedures
   bash recovery/disaster_recovery.sh
   ```

2. **Verify Backup Availability**
   ```bash
   # Check latest backup
   ls -lt backups/database/full/ | head -1
   
   # Verify backup integrity
   bash scripts/verify_backup.sh <backup_file>
   ```

3. **Prepare Recovery Environment**
   ```bash
   # Check available disk space
   df -h
   
   # Check database status
   psql -U postgres -h localhost -c "SELECT 1"
   ```

## 🔄 Recovery Procedures

### Scenario 1: Database Corruption

**Symptoms:**
- Database errors
- Query failures
- Data inconsistencies

**Recovery Steps:**

```bash
# 1. Stop application
docker-compose -f docker-compose.prod.yml stop api

# 2. Verify backup
bash scripts/verify_backup.sh backups/database/full/db_backup_latest.sql.gz

# 3. Restore database
bash scripts/restore_database.sh latest

# 4. Verify restored data
psql -U postgres -h localhost -d study_partner -c "SELECT COUNT(*) FROM users;"

# 5. Restart application
docker-compose -f docker-compose.prod.yml start api

# 6. Verify application
curl http://localhost:8000/health
```

**Estimated Recovery Time:** 15-30 minutes

### Scenario 2: Complete Data Loss

**Symptoms:**
- Database deleted
- All data gone
- Cannot connect to database

**Recovery Steps:**

```bash
# 1. Stop all services
docker-compose -f docker-compose.prod.yml down

# 2. Verify backup exists
ls -la backups/database/full/

# 3. Restore database
bash scripts/restore_database.sh latest

# 4. Verify restore
psql -U postgres -h localhost -d study_partner -c "SELECT COUNT(*) FROM users;"

# 5. Restart services
docker-compose -f docker-compose.prod.yml up -d

# 6. Run health checks
curl http://localhost:8000/health
```

**Estimated Recovery Time:** 30-60 minutes

### Scenario 3: Server Failure

**Symptoms:**
- Server down
- Cannot SSH
- Services not responding

**Recovery Steps:**

```bash
# 1. Provision new server
# - Same OS and specs
# - Same network configuration
# - Same storage capacity

# 2. Install dependencies
sudo apt-get update
sudo apt-get install -y docker.io docker-compose postgresql-client

# 3. Copy application code
git clone <repo> /opt/ai-study-partner

# 4. Restore configuration
cp backups/config/.env.production .env

# 5. Restore database
bash backups/scripts/restore_database.sh latest

# 6. Start services
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify services
curl http://localhost:8000/health
```

**Estimated Recovery Time:** 1-2 hours

### Scenario 4: Ransomware Attack

**Symptoms:**
- Files encrypted
- Cannot access data
- Ransom note displayed

**Recovery Steps:**

```bash
# 1. Isolate affected systems
# - Disconnect from network
# - Stop all services
docker-compose -f docker-compose.prod.yml down

# 2. Preserve evidence
# - Collect logs
# - Document timeline
# - Take screenshots

# 3. Restore from clean backup
# - Use backup from before attack
# - Verify backup integrity
bash scripts/verify_backup.sh <backup_file>

# 4. Restore database
bash scripts/restore_database.sh <backup_file>

# 5. Restore configuration
cp backups/config/.env.backup .env

# 6. Restart services
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify integrity
curl http://localhost:8000/health
```

**Estimated Recovery Time:** 2-4 hours

### Scenario 5: Accidental Data Deletion

**Symptoms:**
- User data deleted
- Cannot undo deletion
- Need to restore specific data

**Recovery Steps:**

```bash
# 1. Identify deletion time
# - Check audit logs
# - Find last good backup before deletion

# 2. Restore to point-in-time
bash scripts/restore_database.sh --time "2026-03-10 14:00:00"

# 3. Verify restored data
psql -U postgres -h localhost -d study_partner -c "SELECT * FROM users WHERE id = 123;"

# 4. If correct, keep restored data
# If incorrect, restore again with different time

# 5. Restart application
docker-compose -f docker-compose.prod.yml restart api
```

**Estimated Recovery Time:** 15-30 minutes

## 📋 Recovery Checklist

### Before Recovery
- [ ] Assess situation
- [ ] Notify stakeholders
- [ ] Preserve evidence
- [ ] Verify backup availability
- [ ] Check disk space
- [ ] Document timeline

### During Recovery
- [ ] Stop affected services
- [ ] Verify backup integrity
- [ ] Restore from backup
- [ ] Verify restored data
- [ ] Restart services
- [ ] Run health checks

### After Recovery
- [ ] Verify all systems operational
- [ ] Check data integrity
- [ ] Notify stakeholders
- [ ] Document recovery
- [ ] Review root cause
- [ ] Implement preventive measures

## 🧪 Testing

### Monthly Disaster Recovery Drill

```bash
# 1. Schedule drill
# - Notify team
# - Plan timeline
# - Prepare test environment

# 2. Simulate failure
# - Stop services
# - Corrupt data
# - Delete files

# 3. Execute recovery
# - Follow recovery procedures
# - Time the recovery
# - Document issues

# 4. Verify recovery
# - Check all systems
# - Verify data integrity
# - Test functionality

# 5. Debrief
# - Review what worked
# - Identify improvements
# - Update procedures
```

### Quarterly Full Recovery Test

```bash
# 1. Provision test environment
# - New server
# - Same configuration
# - Isolated network

# 2. Restore from backup
bash scripts/restore_database.sh latest

# 3. Verify all data
# - Check user count
# - Verify data integrity
# - Test functionality

# 4. Document results
# - Recovery time
# - Issues encountered
# - Improvements needed
```

## 📊 Recovery Metrics

| Metric | Target | Current |
|--------|--------|---------|
| RTO (Recovery Time Objective) | <1 hour | - |
| RPO (Recovery Point Objective) | <6 hours | - |
| Backup Success Rate | 99.9% | - |
| Restore Success Rate | 99.9% | - |
| Monthly Drill Success | 100% | - |

## 🔐 Security Considerations

### During Recovery
- [ ] Verify backup integrity
- [ ] Check for malware
- [ ] Verify encryption keys
- [ ] Audit access logs
- [ ] Monitor for anomalies

### After Recovery
- [ ] Change all passwords
- [ ] Rotate encryption keys
- [ ] Review access logs
- [ ] Audit user accounts
- [ ] Update security policies

## 📞 Escalation

### Level 1: Operations Team
- Assess situation
- Execute recovery procedures
- Monitor recovery progress

### Level 2: Engineering Team
- Investigate root cause
- Implement fixes
- Verify recovery

### Level 3: Management
- Stakeholder communication
- Business continuity decisions
- Post-incident review

## 📝 Documentation

### Recovery Procedures
- [POINT_IN_TIME_RECOVERY.md](POINT_IN_TIME_RECOVERY.md) - PITR guide
- [BACKUP_VERIFICATION.md](BACKUP_VERIFICATION.md) - Verification procedures
- [../scripts/README.md](../scripts/README.md) - Script documentation

### Backup Information
- Backup location: `/opt/ai-study-partner/backend/backups`
- Latest backup: `backups/database/full/db_backup_latest.sql.gz`
- Backup schedule: Daily at 00:00 UTC
- Retention: 30 days for full backups

### Contact Information
- Operations Lead: [contact]
- Database Administrator: [contact]
- Security Officer: [contact]
- Management: [contact]

## ✅ Post-Recovery

### Immediate Actions
1. Verify all systems operational
2. Check data integrity
3. Notify stakeholders
4. Document recovery
5. Monitor for issues

### Follow-up Actions
1. Review root cause
2. Implement preventive measures
3. Update procedures
4. Train team
5. Schedule follow-up drill

---

**Status:** ✅ Production Ready  
**Last Updated:** March 10, 2026  
**Version:** 1.0.0  

**In case of emergency, follow this guide carefully and contact your operations team immediately.**
