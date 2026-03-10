# SSL/TLS Configuration for Production

Production-grade SSL/TLS setup for the AI Study Partner backend API.

## 📋 Quick Navigation

- **[Setup Guide](#setup-guide)** - Step-by-step setup instructions
- **[Certificate Management](#certificate-management)** - Renewal and rotation
- **[Configuration](#configuration)** - Nginx and application setup
- **[Troubleshooting](#troubleshooting)** - Common issues and solutions

## 🚀 Setup Guide

### Option 1: Let's Encrypt (Recommended for Production)

Let's Encrypt provides free, automated SSL certificates with 90-day validity.

#### Prerequisites
- Domain name pointing to your server
- Port 80 and 443 accessible
- Certbot installed

#### Installation

1. **Install Certbot**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# CentOS/RHEL
sudo yum install certbot python3-certbot-nginx

# macOS
brew install certbot
```

2. **Generate Certificate**
```bash
sudo certbot certonly --nginx \
  -d yourdomain.com \
  -d www.yourdomain.com \
  --email admin@yourdomain.com \
  --agree-tos \
  --non-interactive
```

3. **Verify Certificate**
```bash
sudo certbot certificates
```

Certificates are typically stored in:
- `/etc/letsencrypt/live/yourdomain.com/`

4. **Copy to SSL Directory**
```bash
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./certs/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./private/
sudo chown appuser:appuser ./certs/* ./private/*
sudo chmod 600 ./private/*
```

### Option 2: Self-Signed Certificate (Development/Testing)

For development or internal testing only.

```bash
# Generate private key
openssl genrsa -out private/server.key 2048

# Generate certificate (valid for 365 days)
openssl req -new -x509 -key private/server.key \
  -out certs/server.crt -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### Option 3: Commercial Certificate

For organizations preferring commercial support.

1. **Generate CSR (Certificate Signing Request)**
```bash
openssl req -new -key private/server.key \
  -out certs/server.csr \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"
```

2. **Submit to Certificate Authority**
   - Use the CSR with your CA (DigiCert, Comodo, etc.)
   - Follow their verification process

3. **Install Certificate**
```bash
# Copy the certificate from CA
cp /path/to/certificate.crt certs/server.crt

# If CA provides intermediate certificates
cp /path/to/intermediate.crt certs/intermediate.crt
```

## 📁 Directory Structure

```
ssl/
├── README.md                    # This file
├── certs/                       # Public certificates
│   ├── server.crt              # Server certificate
│   ├── intermediate.crt        # Intermediate certificate (if needed)
│   └── .gitkeep                # Placeholder
├── private/                     # Private keys (NEVER commit)
│   ├── server.key              # Private key
│   └── .gitkeep                # Placeholder
├── scripts/
│   ├── generate-self-signed.sh # Generate self-signed cert
│   ├── renew-certificate.sh    # Renew Let's Encrypt cert
│   ├── verify-certificate.sh   # Verify certificate
│   └── setup-letsencrypt.sh    # Automated Let's Encrypt setup
└── configs/
    ├── nginx-ssl.conf          # Nginx SSL configuration
    └── ssl-params.conf         # SSL security parameters
```

## 🔐 Certificate Management

### Automatic Renewal (Let's Encrypt)

1. **Enable Auto-Renewal**
```bash
# Test renewal
sudo certbot renew --dry-run

# Enable automatic renewal (cron job)
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Verify timer is active
sudo systemctl status certbot.timer
```

2. **Manual Renewal**
```bash
sudo certbot renew --force-renewal
```

3. **Renewal Hook Script**
```bash
# Create renewal hook
sudo nano /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh
```

Add:
```bash
#!/bin/bash
systemctl reload nginx
```

Make executable:
```bash
sudo chmod +x /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh
```

### Certificate Rotation

1. **Generate New Certificate**
```bash
sudo certbot certonly --nginx \
  -d yourdomain.com \
  --force-renewal
```

2. **Update Application**
```bash
# Copy new certificate
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./certs/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./private/

# Restart application
sudo systemctl restart api
```

3. **Verify New Certificate**
```bash
openssl x509 -in certs/server.crt -text -noout
```

## ⚙️ Configuration

### Nginx SSL Configuration

See `configs/nginx-ssl.conf` for complete configuration.

Key settings:
```nginx
# SSL protocols (TLS 1.2 and 1.3 only)
ssl_protocols TLSv1.2 TLSv1.3;

# Strong ciphers
ssl_ciphers HIGH:!aNULL:!MD5;
ssl_prefer_server_ciphers on;

# Certificate paths
ssl_certificate /app/ssl/certs/server.crt;
ssl_certificate_key /app/ssl/private/server.key;

# HSTS (HTTP Strict Transport Security)
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### Application Configuration

Update `api/main.py` for HTTPS:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ssl

app = FastAPI(title="AI Study Partner API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    
    # SSL context for production
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        certfile="ssl/certs/server.crt",
        keyfile="ssl/private/server.key"
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,
        ssl_context=ssl_context
    )
```

### Docker Configuration

Update `Dockerfile.prod`:

```dockerfile
# Copy SSL certificates
COPY ssl/certs/ /app/ssl/certs/
COPY ssl/private/ /app/ssl/private/

# Set permissions
RUN chmod 600 /app/ssl/private/*

# Expose HTTPS port
EXPOSE 443

# Run with SSL
CMD ["gunicorn", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:443", \
     "--certfile=/app/ssl/certs/server.crt", \
     "--keyfile=/app/ssl/private/server.key", \
     "api.main:app"]
```

## 🔒 Security Best Practices

### 1. Private Key Protection
```bash
# Restrict permissions
chmod 600 private/server.key

# Verify permissions
ls -la private/server.key
# Should show: -rw------- 1 appuser appuser
```

### 2. Certificate Pinning
For enhanced security, implement certificate pinning:

```python
import ssl
import certifi

# Create SSL context with certificate pinning
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations("ssl/certs/server.crt")
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED
```

### 3. HSTS (HTTP Strict Transport Security)
```nginx
# Force HTTPS for 1 year
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

### 4. Certificate Transparency
```bash
# Verify certificate transparency logs
openssl x509 -in certs/server.crt -text -noout | grep -A 5 "CT Precertificate"
```

### 5. Regular Audits
```bash
# Check certificate expiration
openssl x509 -in certs/server.crt -noout -dates

# Verify certificate chain
openssl verify -CAfile certs/intermediate.crt certs/server.crt

# Test SSL configuration
openssl s_client -connect localhost:443 -tls1_2
```

## 📊 SSL/TLS Testing

### Online Tools
- [SSL Labs](https://www.ssllabs.com/ssltest/) - Comprehensive SSL testing
- [Mozilla Observatory](https://observatory.mozilla.org/) - Security headers
- [Qualys SSL Labs](https://www.ssllabs.com/ssltest/analyze.html) - Detailed analysis

### Command Line Testing

```bash
# Test SSL connection
openssl s_client -connect yourdomain.com:443

# Check certificate validity
openssl x509 -in certs/server.crt -text -noout

# Verify certificate chain
openssl verify -CAfile certs/intermediate.crt certs/server.crt

# Test TLS version
openssl s_client -connect yourdomain.com:443 -tls1_2

# Check cipher strength
openssl s_client -connect yourdomain.com:443 -cipher 'HIGH'
```

## 🔄 Renewal Checklist

Before certificate expiration:

- [ ] Check expiration date: `openssl x509 -in certs/server.crt -noout -dates`
- [ ] Renew certificate: `sudo certbot renew`
- [ ] Copy new certificate to ssl/certs/
- [ ] Copy new key to ssl/private/
- [ ] Restart Nginx: `sudo systemctl restart nginx`
- [ ] Restart API: `sudo systemctl restart api`
- [ ] Verify new certificate: `openssl x509 -in certs/server.crt -text -noout`
- [ ] Test SSL connection: `openssl s_client -connect yourdomain.com:443`

## 🆘 Troubleshooting

### Certificate Not Found
```bash
# Check certificate location
ls -la certs/
ls -la private/

# Verify paths in configuration
grep -r "ssl_certificate" /etc/nginx/
```

### Permission Denied
```bash
# Fix permissions
sudo chown appuser:appuser certs/* private/*
sudo chmod 644 certs/*
sudo chmod 600 private/*
```

### Certificate Expired
```bash
# Check expiration
openssl x509 -in certs/server.crt -noout -dates

# Renew immediately
sudo certbot renew --force-renewal
```

### Mixed Content Warning
Ensure all resources use HTTPS:
```html
<!-- ❌ Wrong -->
<script src="http://example.com/script.js"></script>

<!-- ✅ Correct -->
<script src="https://example.com/script.js"></script>
```

### HSTS Preload Issues
```bash
# Check HSTS header
curl -I https://yourdomain.com | grep Strict-Transport-Security

# Submit to HSTS preload list
# https://hstspreload.org/
```

### Certificate Chain Issues
```bash
# Verify complete chain
openssl s_client -connect yourdomain.com:443 -showcerts

# Check intermediate certificate
openssl x509 -in certs/intermediate.crt -text -noout
```

## 📚 Additional Resources

### Let's Encrypt
- [Official Documentation](https://letsencrypt.org/docs/)
- [Certbot Documentation](https://certbot.eff.org/docs/)
- [Rate Limits](https://letsencrypt.org/docs/rate-limits/)

### SSL/TLS Best Practices
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [OWASP Transport Layer Protection](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [SSL Labs Best Practices](https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices)

### Security Standards
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)

## 🔐 Security Compliance

### Supported Standards
- ✅ TLS 1.2 and 1.3
- ✅ HSTS (HTTP Strict Transport Security)
- ✅ OCSP Stapling
- ✅ Certificate Transparency
- ✅ Perfect Forward Secrecy

### Compliance Checklist
- [ ] TLS 1.2+ only
- [ ] Strong ciphers configured
- [ ] HSTS enabled
- [ ] Certificate valid and not expired
- [ ] Certificate chain complete
- [ ] Private key protected
- [ ] Regular renewal scheduled
- [ ] Security headers configured

## 📝 Environment Variables

Add to `.env`:
```env
# SSL Configuration
SSL_ENABLED=true
SSL_CERT_PATH=/app/ssl/certs/server.crt
SSL_KEY_PATH=/app/ssl/private/server.key
SSL_INTERMEDIATE_PATH=/app/ssl/certs/intermediate.crt

# HTTPS Settings
HTTPS_PORT=443
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=true
```

## 🚀 Production Deployment

### Step-by-Step

1. **Generate Certificate**
   ```bash
   sudo certbot certonly --nginx -d yourdomain.com
   ```

2. **Copy to SSL Directory**
   ```bash
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./certs/
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./private/
   ```

3. **Set Permissions**
   ```bash
   sudo chown appuser:appuser ./certs/* ./private/*
   sudo chmod 600 ./private/*
   ```

4. **Configure Nginx**
   ```bash
   sudo cp configs/nginx-ssl.conf /etc/nginx/sites-available/api
   sudo ln -s /etc/nginx/sites-available/api /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

5. **Start Application**
   ```bash
   sudo systemctl start api
   sudo systemctl enable api
   ```

6. **Verify**
   ```bash
   curl https://yourdomain.com/health
   ```

---

**Version:** 1.0.0  
**Last Updated:** March 10, 2026  
**Status:** Production Ready  
**Compliance:** TLS 1.2+ | HSTS | OCSP Stapling
