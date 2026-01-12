@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv...
  python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

"%PYTHON%" -m pip install --upgrade pip
if exist "scripts\desktop\requirements-desktop.txt" (
  "%PYTHON%" -m pip install -r "scripts\desktop\requirements-desktop.txt"
)
if exist "scripts\web\requirements-web.txt" (
  "%PYTHON%" -m pip install -r "scripts\web\requirements-web.txt"
  "%PYTHON%" -m playwright install chromium
)

set "LOG_DIR=%ROOT%"
set "STAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "STAMP=%STAMP: =0%"
set "LOG_PATH=%LOG_DIR%smoke_check_%STAMP%.log"

echo Running smoke check... > "%LOG_PATH%"
"%PYTHON%" scripts\utils\smoke_check_scripts.py %* >> "%LOG_PATH%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
echo Log saved to "%LOG_PATH%"

pause
exit /b %EXITCODE%
