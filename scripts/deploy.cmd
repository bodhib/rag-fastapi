@echo off
setlocal

echo ========================================
echo RAG API Deployment
echo ========================================

echo.
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo.
echo [2/4] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

echo.
echo [3/4] Applying database migrations...
python -m alembic upgrade head
if errorlevel 1 (
    echo Database migration failed.
    echo Application will NOT be started.
    exit /b 1
)

echo.
echo [4/4] Starting application...
python -m uvicorn app.main:app

endlocal