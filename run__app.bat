@echo off
setlocal
title Insurance Document Intelligence

REM Run this file from the project root (the folder containing app\main.py).
cd /d "%~dp0"

echo ================================================
echo   Insurance Document Intelligence
echo ================================================
echo.

if not exist "app\main.py" (
    echo ERROR: app\main.py was not found.
    echo Move this BAT file into the dip_project root folder,
    echo next to the app folder.
    echo.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    echo ERROR: Project virtual environment was not found:
    echo   .venv\Scripts\python.exe
    echo.
    echo Create and configure the project environment first.
    echo See the viva guide / project README for setup instructions.
    pause
    exit /b 1
)

REM Configure Tesseract if it is installed in the standard Windows location.
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    set "TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
    set "PATH=C:\Program Files\Tesseract-OCR;%PATH%"
)

echo Checking Streamlit...
"%PYTHON_EXE%" -m streamlit --version
if errorlevel 1 (
    echo.
    echo ERROR: Streamlit is not installed in this project's .venv.
    echo Activate the environment and install the requirements.
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo Your browser should open at http://localhost:8501
echo Keep this window open while using the application.
echo To stop the application, press Ctrl+C in this window.
echo.

"%PYTHON_EXE%" -m streamlit run "app\main.py"

echo.
echo Application stopped.
pause
endlocal
