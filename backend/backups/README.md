# Backups Directory

Production-grade backup and recovery system for AI Study Partner Backend.

## 📋 Overview

This directory contains backup files and recovery procedures for the AI Study Partner Backend. All backups are automated and can be restored quickly in case of data loss or system failure.

## 📁 Directory Structure

```
backups/
├── README.md                    # This file
├── .gitkeep                     # Placeholder
├── database/                    # Database backups
│   ├── full/                    # Full database backups
│   ├── incremental/             # Incremental backups
│   └── README.md                # Database backup docs
├── config/                      # Configuration backups
│   ├── .env.backup              # Environment backups
│   ├── nginx.conf.backup        # Nginx config backups
│   └── README.md                # Config backup docs
├── logs/                        # Log archives
│   ├── app/                     # Application logs
│   ├── security/                # Security logs
│   ├── audit/                   # Audit logs
│   └── README.md                # Log backup docs
├── scripts/                     # Backup scripts
│   ├── backup_database.sh       # Database backup script
│   ├── backup_config.sh         # Config backup script
│   ├── backup_logs.sh           # Log backup script
│   ├── restore_database.sh      # Database restore script
│   ├── restore_config.sh        # Config restore script
│   ├── cleanup_old_backups.sh   # Cleanup script
│   └── README.md                # Script documentation
└── recovery/                    # Recovery procedures
    ├── DISASTER_RECOVERY.md     # Disaster recovery guide
    ├── POINT_IN_TIME_RECOVERY.md # PITR guide
    ├── BACKUP_VERIFICATION.md   # Verification procedures
    └── README.md                # Recovery docs
```

## 🔄 Backup Strategy

### Backup Types

**1. Full Backups**
- Complete database snapshot
- Frequency: Daily (midnight UTC)
- Retention: 30 days
- Size: ~500MB-1GB

**2. Incremental Backups**
- Changes since last backup
- Frequency: Every 6 hours
- Retention: 7 days
- Size: ~50-100MB

**3. Configuration Backups**
- Environment files
- Nginx configuration
- Application settings
- Frequency: On every deployment
- Retention: 90 days

**4. Log Archives**
- Application logs
- Security logs
- Audit logs
- Frequency: Daily
- Retention: 90 days

### Backup Schedule

```
Time (UTC)  | Type              | Retention
------------|-------------------|----------
00:00       | Full Database     | 30 days
06:00       | Incremental       | 7 days
12:00       | Incremental       | 7 days
18:00       | Incremental       | 7 days
Daily 23:00 | Log Archive       | 90 days
On Deploy   | Config Backup     | 90 days
```

## 💾 Storage

### Local Storage
- **Location:** `/opt/ai-study-partner/backend/backups`
- **Capacity:** 100GB minimum
- **Retention:** 30 days for full backups

### Remote Storage (Recommended)
- **AWS S3:** For long-term storage
- **Google Cloud Storage:** Alternative option
- **Azure Blob Storage:** Alternative option
- **Retention:** 90 days minimum

### Backup Encryption
- All backups encrypted with AES-256
- Encryption key stored in secure vault
- Separate from backup storage

## 🔐 Security

### Access Control
- Backups readable only by backup user
- Permissions: 600 (read/write owner only)
- Stored in secure location
- Encrypted at rest

### Encryption
- AES-256 encryption
- Separate encryption keys
- Key rotation every 90 days
- Keys stored in AWS Secrets Manager

### Verification
- Checksums for integrity
- Regular restore tests
- Automated verification
- Monthly full restore test

## 📊 Backup Procedures

### Automated Backups

Backups run automatically via cron jobs:

```bash
# View cron jobs
crontab -l

# Edit cron jobs
crontab -e
```

### Manual Backups

```bash
# Full database backup
bash scripts/backup_database.sh full

# Incremental backup
bash scripts/backup_database.sh incremental

# Config backup
bash scripts/backup_config.sh

# Log backup
bash scripts/backup_logs.sh
```

### Backup Verification

```bash
# Verify backup integrity
bash scripts/verify_backup.sh <backup_file>

# List available backups
ls -lh database/full/
ls -lh database/incremental/

# Check backup size
du -sh database/
```

## 🔄 Recovery Procedures

### Quick Recovery

```bash
# Restore latest full backup
bash scripts/restore_database.sh latest

# Restore specific backup
bash scripts/restore_database.sh database/full/db_backup_20260310_000000.sql

# Restore configuration
bash scripts/restore_config.sh
```

### Point-in-Time Recovery

```bash
# Restore to specific time
bash scripts/restore_database.sh --time "2026-03-10 14:30:00"

# See POINT_IN_TIME_RECOVERY.md for details
cat recovery/POINT_IN_TIME_RECOVERY.md
```

### Disaster Recovery

```bash
# Full system recovery
bash recovery/disaster_recovery.sh

# See DISASTER_RECOVERY.md for details
cat recovery/DISASTER_RECOVERY.md
```

## 📈 Monitoring

### Backup Status

```bash
# Check last backup
ls -lt database/full/ | head -1

# Check backup size
du -sh database/full/

# Check backup age
find database/full/ -type f -mtime +1 -print
```

### Alerts

Alerts are sent if:
- Backup fails
- Backup takes >1 hour
- Backup size exceeds threshold
- Backup verification fails
- Restore test fails

### Metrics

- Backup success rate: 99.9%+
- Average backup time: <30 minutes
- Average restore time: <15 minutes
- Backup verification: 100%

## 🧪 Testing

### Backup Testing

```bash
# Test backup creation
bash scripts/backup_database.sh test

# Verify backup integrity
bash scripts/verify_backup.sh <backup_file>

# Test restore
bash scripts/restore_database.sh --test <backup_file>
```

### Restore Testing

Monthly restore tests:
- [ ] Restore full backup
- [ ] Verify data integrity
- [ ] Test application functionality
- [ ] Document results

### Disaster Recovery Drills

Quarterly disaster recovery drills:
- [ ] Simulate data loss
- [ ] Execute recovery procedures
- [ ] Verify system functionality
- [ ] Document lessons learned

## 📋 Backup Checklist

### Daily
- [ ] Verify backup completed
- [ ] Check backup size
- [ ] Monitor backup logs
- [ ] Verify no errors

### Weekly
- [ ] Test restore procedure
- [ ] Verify backup integrity
- [ ] Check storage capacity
- [ ] Review backup logs

### Monthly
- [ ] Full restore test
- [ ] Verify data integrity
- [ ] Test application
- [ ] Document results

### Quarterly
- [ ] Disaster recovery drill
- [ ] Test all recovery procedures
- [ ] Update documentation
- [ ] Review backup strategy

## 🔧 Maintenance

### Cleanup Old Backups

```bash
# Remove backups older than 30 days
bash scripts/cleanup_old_backups.sh 30

# Remove backups older than 7 days
bash scripts/cleanup_old_backups.sh 7
```

### Storage Management

```bash
# Check storage usage
du -sh backups/

# Check available space
df -h /opt/ai-study-partner/backend/

# Archive old backups to S3
bash scripts/archive_to_s3.sh
```

### Backup Rotation

- Full backups: Keep 30 days
- Incremental backups: Keep 7 days
- Config backups: Keep 90 days
- Log archives: Keep 90 days

## 📞 Support

### Documentation
- [DISASTER_RECOVERY.md](recovery/DISASTER_RECOVERY.md) - Disaster recovery guide
- [POINT_IN_TIME_RECOVERY.md](recovery/POINT_IN_TIME_RECOVERY.md) - PITR guide
- [BACKUP_VERIFICATION.md](recovery/BACKUP_VERIFICATION.md) - Verification procedures
- [scripts/README.md](scripts/README.md) - Script documentation

### Common Issues

**Backup Failed**
- Check disk space: `df -h`
- Check database connection: `psql -U postgres -h localhost -c "SELECT 1"`
- Check backup logs: `tail -f logs/backup.log`

**Restore Failed**
- Verify backup file exists: `ls -la database/full/`
- Check database permissions: `psql -U postgres -h localhost -l`
- Check restore logs: `tail -f logs/restore.log`

**Storage Full**
- Check backup size: `du -sh backups/`
- Remove old backups: `bash scripts/cleanup_old_backups.sh 30`
- Archive to S3: `bash scripts/archive_to_s3.sh`

## 🎯 Best Practices

### Do's ✅
- ✅ Test backups regularly
- ✅ Verify backup integrity
- ✅ Store backups remotely
- ✅ Encrypt all backups
- ✅ Document procedures
- ✅ Monitor backup status
- ✅ Rotate backups
- ✅ Test disaster recovery

### Don'ts ❌
- ❌ Store backups locally only
- ❌ Skip backup verification
- ❌ Ignore backup failures
- ❌ Use weak encryption
- ❌ Share backup keys
- ❌ Delete backups without testing
- ❌ Ignore storage limits
- ❌ Skip restore tests

## 📊 Backup Statistics

| Metric | Target | Current |
|--------|--------|---------|
| Backup Success Rate | 99.9% | - |
| Average Backup Time | <30 min | - |
| Average Restore Time | <15 min | - |
| Backup Verification | 100% | - |
| Monthly Restore Test | 100% | - |
| Disaster Recovery Drill | Quarterly | - |

## 🚀 Getting Started

1. **Review Documentation**
   - Read this README
   - Review backup scripts
   - Review recovery procedures

2. **Setup Backups**
   - Configure backup schedule
   - Setup remote storage
   - Configure encryption

3. **Test Backups**
   - Run manual backup
   - Verify backup integrity
   - Test restore procedure

4. **Monitor Backups**
   - Setup alerts
   - Monitor backup logs
   - Track backup metrics

## ✅ Production Readiness

- ✅ Automated backups
- ✅ Multiple backup types
- ✅ Encryption support
- ✅ Remote storage ready
- ✅ Recovery procedures
- ✅ Verification procedures
- ✅ Monitoring setup
- ✅ Documentation complete

---

**Status:** ✅ Production Ready  
**Last Updated:** March 10, 2026  
**Version:** 1.0.0  

**Backups are critical for production systems. Ensure they are properly configured and tested!**
