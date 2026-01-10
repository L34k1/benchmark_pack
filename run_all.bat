@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv...
  python -m venv "%VENV_DIR%"
)

set "PYTHON=%VENV_DIR%\Scripts\python.exe"

"%PYTHON%" -m pip install --upgrade pip

if exist "scripts\desktop\requirements-desktop.txt" (
  "%PYTHON%" -m pip install -r "scripts\desktop\requirements-desktop.txt"
)

if exist "scripts\web\requirements-web.txt" (
  "%PYTHON%" -m pip install -r "scripts\web\requirements-web.txt"
  "%PYTHON%" -m playwright install chromium
)

"%PYTHON%" run_all.py %*
set "EXITCODE=%ERRORLEVEL%"

exit /b %EXITCODE%
