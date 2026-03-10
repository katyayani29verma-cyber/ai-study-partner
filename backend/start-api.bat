@echo off
REM Start AI Study Partner API on Windows

echo.
echo ========================================
echo AI Study Partner API Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Check if pydantic-settings is installed
python -c "import pydantic_settings" >nul 2>&1
if errorlevel 1 (
    echo Installing pydantic-settings...
    pip install pydantic-settings>=2.0.0
    if errorlevel 1 (
        echo ERROR: Failed to install pydantic-settings
        pause
        exit /b 1
    )
)

echo ✓ pydantic-settings installed
echo.

REM Check if fastapi is installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing fastapi...
    pip install fastapi uvicorn
    if errorlevel 1 (
        echo ERROR: Failed to install fastapi
        pause
        exit /b 1
    )
)

echo ✓ fastapi installed
echo.

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file...
    (
        echo # Database Configuration
        echo DATABASE_URL=sqlite:///./study_partner.db
        echo.
        echo # Security Keys
        echo SECRET_KEY=your-secret-key-32-characters-minimum-required-here
        echo MASTER_KEY=your-master-key-32-characters-minimum-required-here
        echo.
        echo # CORS Configuration
        echo ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
        echo.
        echo # Redis Configuration
        echo REDIS_URL=redis://localhost:6379
        echo.
        echo # API Configuration
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo LOG_LEVEL=INFO
    ) > .env
    echo ✓ .env file created
    echo.
)

echo ========================================
echo Starting API Server...
echo ========================================
echo.
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the API
python -m uvicorn api.main:app --reload

pause
