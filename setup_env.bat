@echo off
cd /d "%~dp0"
echo Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.10+ is installed and on PATH.
    pause
    exit /b 1
)

echo Activating environment...
call .venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: PyTorch installation failed.
    pause
    exit /b 1
)

echo Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Setup complete! Use run_grader.bat to launch the application.
pause
