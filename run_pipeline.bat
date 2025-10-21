@echo off
REM Batch script to run the complete ML pipeline on Windows

echo ================================================================
echo    TELECOM CHURN PREDICTION - PIPELINE RUNNER
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
pip show pandas >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    echo.
    echo Dependencies installed!
    echo.
) else (
    echo Dependencies already installed!
    echo.
)

REM Run the pipeline
echo ================================================================
echo    STARTING PIPELINE
echo ================================================================
echo.

python run_pipeline.py

REM Deactivate virtual environment
deactivate

echo.
echo ================================================================
echo    PIPELINE EXECUTION COMPLETED
echo ================================================================
echo.

pause
