@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHONPATH=%~dp0
python app\grading_app.py %*
if errorlevel 1 (
    echo.
    echo An error occurred. See message above.
    pause
)
