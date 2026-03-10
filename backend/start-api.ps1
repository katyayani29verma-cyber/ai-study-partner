# Start AI Study Partner API on Windows PowerShell

Write-Host ""
Write-Host "========================================"
Write-Host "AI Study Partner API Startup"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion"
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH"
    Write-Host "Please install Python 3.8+ from https://www.python.org"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if pydantic-settings is installed
try {
    python -c "import pydantic_settings" 2>&1 | Out-Null
    Write-Host "✓ pydantic-settings installed"
} catch {
    Write-Host "Installing pydantic-settings..."
    pip install pydantic-settings>=2.0.0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install pydantic-settings"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Check if fastapi is installed
try {
    python -c "import fastapi" 2>&1 | Out-Null
    Write-Host "✓ fastapi installed"
} catch {
    Write-Host "Installing fastapi..."
    pip install fastapi uvicorn
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install fastapi"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..."
    $envContent = @"
# Database Configuration
DATABASE_URL=sqlite:///./study_partner.db

# Security Keys
SECRET_KEY=your-secret-key-32-characters-minimum-required-here
MASTER_KEY=your-master-key-32-characters-minimum-required-here

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
"@
    Set-Content -Path ".env" -Value $envContent
    Write-Host "✓ .env file created"
    Write-Host ""
}

Write-Host "========================================"
Write-Host "Starting API Server..."
Write-Host "========================================"
Write-Host ""
Write-Host "API will be available at: http://localhost:8000"
Write-Host "API Documentation: http://localhost:8000/docs"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

# Start the API
python -m uvicorn api.main:app --reload

Read-Host "Press Enter to exit"
