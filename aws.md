Here’s a ready‑to‑use deployment guide you can save as `DEPLOY_AWS_EC2.md` in your repo.

```markdown
# AI Study Partner Backend – Deploy on AWS EC2

**Status:** Draft for single‑server production deployment on AWS EC2 (Ubuntu + Docker + Nginx + SSL)  
**Target:** Small/medium production workloads for the backend API only (no frontend)

---

## 1. Architecture Overview

- **EC2 instance (Ubuntu 22.04)**  
  - Runs Docker + Docker Compose  
  - Hosts:
    - FastAPI backend (from `Dockerfile.prod`)
    - PostgreSQL (inside Docker) – or external RDS (recommended for scale)
    - Redis (inside Docker)
    - Nginx reverse proxy (from `nginx.conf` / SSL config)

- **Security & networking**
  - Security Group:
    - Allow **HTTP (80)** and **HTTPS (443)** from the internet
    - Allow **SSH (22)** from your own IP only
  - Optional: use **Elastic IP** and point your domain DNS to it

---

## 2. Prerequisites

### 2.1. AWS & domain

- **AWS Account** with permission to create:
  - EC2 instances
  - Security Groups
  - Elastic IPs (optional)
- **Domain name** (e.g. `api.yourdomain.com`) managed at:
  - Route 53, or
  - Any DNS provider where you can add A/CAA records

### 2.2. Local requirements (for management)

- `ssh` client
- `git`
- Basic familiarity with:
  - Linux CLI
  - Docker / Docker Compose
  - Editing text files (e.g. `nano`, `vim`)

---

## 3. Create EC2 Instance

### 3.1. Choose AMI and size

1. In AWS Console → **EC2 → Instances → Launch instances**.
2. **Name**: `ai-study-partner-backend-prod`
3. **AMI**: `Ubuntu Server 22.04 LTS (x86_64)`
4. **Instance type**: start with `t3.small` or `t3.medium` (2 vCPU, 2–4 GB RAM).
5. **Key pair**: create or select existing. Download the `.pem` file and keep it safe.

### 3.2. Configure network and security group

1. **VPC / Subnet**: default is OK for a simple setup.
2. **Security Group**:
   - Create new, e.g. `ai-study-partner-backend-sg`
   - Inbound rules:
     - **SSH**: port `22`, source = your IP only (e.g. `203.0.113.10/32`)
     - **HTTP**: port `80`, source = `0.0.0.0/0`
     - **HTTPS**: port `443`, source = `0.0.0.0/0`
   - Outbound: allow all (default) is fine.

### 3.3. Launch and (optionally) assign Elastic IP

1. Click **Launch instance**.
2. After it’s running, note the **Public IPv4 address**.
3. (Recommended) Create an **Elastic IP** and associate it with this instance so the IP does not change.

---

## 4. Point Domain to EC2

In your DNS provider:

1. Create an **A record** for your API domain, e.g.:
   - **Name**: `api.yourdomain.com`
   - **Type**: `A`
   - **Value**: the EC2 public IP (or Elastic IP)
2. Wait for DNS to propagate (can be up to 5–10 minutes).

---

## 5. SSH Into the Instance

From your local machine:

```bash
chmod 600 path/to/your-key.pem

ssh -i path/to/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

You are now in the EC2 server as user `ubuntu`.

---

## 6. Install Docker & Docker Compose

On the EC2 instance:

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install basic tools
sudo apt-get install -y git curl ca-certificates gnupg lsb-release

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Allow current user to use Docker without sudo
sudo usermod -aG docker ubuntu
# Re-login or run:
newgrp docker

# Install docker-compose plugin (or standalone if you prefer)
sudo apt-get install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

---

## 7. Clone the Project

Choose a directory (e.g. `/opt`):

```bash
sudo mkdir -p /opt/ai-study-partner
sudo chown ubuntu:ubuntu /opt/ai-study-partner
cd /opt/ai-study-partner

# Clone your repo (replace with actual Git URL)
git clone https://github.com/your-org/ai-study-partner.git

cd ai-study-partner/backend
```

---

## 8. Prepare Environment Configuration

You already have sample `.env` and Docker env in the backend docs. On the EC2 instance:

### 8.1. Create `.env` for the app

```bash
cd /opt/ai-study-partner/ai-study-partner/backend

cp .env .env.production   # if not already there
nano .env.production
```

Fill real values, for example:

```env
# SECURITY & SECRETS
SECRET_KEY=REPLACE_WITH_SECURE_HEX_64
MASTER_KEY=REPLACE_WITH_SECURE_HEX_64

# DATABASE (inside docker-compose, service name "db")
DATABASE_URL=postgresql://study_user:secure_db_password@db:5432/study_partner
POOL_SIZE=20
MAX_OVERFLOW=10
POOL_RECYCLE=3600

# REDIS
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=

# CORS & FRONTEND
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://api.yourdomain.com

# ENVIRONMENT
ENVIRONMENT=production

# API SETTINGS
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# LOGGING
LOG_LEVEL=INFO

# FILE UPLOADS
MAX_UPLOAD_SIZE=52428800

# SECURITY HEADERS
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=true
```

> **Important:**  
> - Generate `SECRET_KEY` and `MASTER_KEY` using `openssl rand -hex 32` or Python (see `backend/README.md`).
> - Keep this file secret. Do **not** commit it to Git.

### 8.2. Create `.env` for Docker Compose (if used)

If `docker-compose.prod.yml` expects a `.env` file, create it:

```bash
nano .env
```

Example:

```env
DB_PASSWORD=secure_db_password
DATABASE_URL=postgresql://study_user:secure_db_password@db:5432/study_partner

REDIS_PASSWORD=
REDIS_URL=redis://redis:6379

SECRET_KEY=REPLACE_WITH_SECURE_HEX_64
MASTER_KEY=REPLACE_WITH_SECURE_HEX_64

ENVIRONMENT=production
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://api.yourdomain.com
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=52428800
```

Adjust keys/variable names to match `docker-compose.prod.yml`.

---

## 9. Configure Nginx & SSL (High Level)

Your backend includes:

- `nginx.conf`
- `ssl/configs/nginx-ssl.conf`
- `ssl/README.md`

There are two main options:

- **Option A (simple)**: Use **Nginx container** from `docker-compose.prod.yml`, and use **Let’s Encrypt** via a companion container (e.g. `nginx-proxy` + `letsencrypt-nginx-proxy-companion` or Certbot).
- **Option B (simpler docs)**: Use Nginx and Certbot installed directly on EC2 (host), and proxy to Docker app on port `8000`.

For many small deployments, **Option B** is easier to reason about. Below is Option B.

### 9.1. Install Nginx & Certbot on EC2

```bash
sudo apt-get update -y
sudo apt-get install -y nginx

# Allow Nginx ports through Ubuntu firewall (if ufw is used)
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
sudo ufw enable   # if not already enabled

# Install Certbot for Nginx
sudo apt-get install -y certbot python3-certbot-nginx
```

### 9.2. Configure Nginx as reverse proxy

Create a server block for your domain:

```bash
sudo nano /etc/nginx/sites-available/ai-study-partner-backend
```

Example (proxy to backend at `localhost:8000`):

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header   Host                 $host;
        proxy_set_header   X-Real-IP            $remote_addr;
        proxy_set_header   X-Forwarded-For      $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto    $scheme;
    }
}
```

Enable and test:

```bash
sudo ln -s /etc/nginx/sites-available/ai-study-partner-backend \
           /etc/nginx/sites-enabled/ai-study-partner-backend

sudo nginx -t
sudo systemctl restart nginx
```

---

## 10. Build & Run the Backend with Docker Compose

### 10.1. First run (without SSL)

From `backend`:

```bash
cd /opt/ai-study-partner/ai-study-partner/backend

# Build and start in background
docker compose -f docker-compose.prod.yml up -d --build

# Check containers
docker compose -f docker-compose.prod.yml ps

# View logs (API)
docker compose -f docker-compose.prod.yml logs -f api
```

### 10.2. Run database migrations

Inside the API container:

```bash
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### 10.3. Verify the app locally on the server

From EC2:

```bash
curl http://127.0.0.1:8000/health
```

You should see a JSON response like:

```json
{
  "status": "healthy",
  "service": "AI Study Partner API",
  "version": "1.0.0"
}
```

From your browser (remote):

- `http://api.yourdomain.com/health`
- `http://api.yourdomain.com/docs`

Once this works over HTTP, you’re ready for HTTPS.

---

## 11. Enable HTTPS with Let’s Encrypt (Certbot)

Run Certbot with the Nginx plugin:

```bash
sudo certbot --nginx -d api.yourdomain.com
```

- Follow prompts:
  - Provide email
  - Agree to terms
  - Choose whether to redirect all HTTP → HTTPS (recommended: **Yes**)

Certbot will:

- Obtain a certificate
- Update your Nginx config to listen on `443` with SSL
- Optionally configure automatic redirect from HTTP to HTTPS
- Install a cron job/systemd timer for automatic renewal

Test HTTPS:

- `https://api.yourdomain.com/health`
- `https://api.yourdomain.com/docs`

---

## 12. Production Hardening Checklist (EC2)

- **System & security**
  - [ ] Change SSH from password auth to **key‑only**
  - [ ] Restrict SSH to your IP in the Security Group
  - [ ] Regular `apt-get update && apt-get upgrade`
  - [ ] Set up basic monitoring (CPU, RAM, disk, network) via CloudWatch or other

- **Docker & app**
  - [ ] Use `ENVIRONMENT=production`
  - [ ] Do **not** run DB/Redis in the same instance for larger scale (migrate to RDS/Elasticache later)
  - [ ] Configure log rotation for Docker logs or central logging
  - [ ] Ensure `SECRET_KEY` / `MASTER_KEY` are long and random

- **Backups**
  - [ ] Use the **`backups`** system from the repo:
    - Configure `cron_jobs.conf`
    - Point S3 credentials in env
    - Test `backup_database.sh` and `restore_database.sh`
  - [ ] Store DB backups off‑instance (S3, etc.)

---

## 13. Deployment Updates (Rolling Changes)

When you push new code to the `main` branch and want to deploy:

```bash
cd /opt/ai-study-partner/ai-study-partner/backend

# Pull latest changes
git pull origin main

# Rebuild and restart containers
docker compose -f docker-compose.prod.yml up -d --build

# Run migrations if needed
docker compose -f docker-compose.prod.yml exec api alembic upgrade head

# Verify
curl http://127.0.0.1:8000/health
```

---

## 14. Troubleshooting

### 14.1. API not responding

- Check containers:

```bash
cd /opt/ai-study-partner/ai-study-partner/backend
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f api
```

- Ensure DB and Redis containers are running.
- Check `DATABASE_URL` and `REDIS_URL` in `.env` / `.env.production`.

### 14.2. Nginx / SSL issues

- Test Nginx config:

```bash
sudo nginx -t
sudo systemctl status nginx
sudo journalctl -u nginx -n 50 --no-pager
```

- Check Certbot logs:

```bash
sudo journalctl -u certbot.timer -n 20 --no-pager
sudo ls -l /etc/letsencrypt/live
```

### 14.3. Ports blocked

- Confirm EC2 Security Group allows ports `80` and `443`.
- If using `ufw`, ensure:

```bash
sudo ufw status
# Should show 'Nginx Full' and 'OpenSSH' allowed
```

---

## 15. Summary

- **You now have**:
  - Backend API running in Docker on EC2
  - Nginx reverse proxy on the host
  - Let’s Encrypt SSL for `api.yourdomain.com`
  - Environment configuration via `.env.production`
  - A path to integrate the existing backup system (`backups/`)

For more details, cross‑check with:

- `backend/README.md`
- `backend/DEPLOYMENT.md`
- `backend/ssl/README.md`
- `backend/PRODUCTION_DEPLOYMENT.md`

```