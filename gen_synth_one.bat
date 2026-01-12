@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if "%~1"=="" (
  echo Usage: gen_synth_one.bat EDF^|NWB [--no-clobber]
  exit /b 1
)

set "FORMAT=%~1"
shift

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
)

%PYTHON% scripts\data\gen_synth_one.py --format %FORMAT% %*
if errorlevel 1 (
  echo.
  echo gen_synth_one failed with exit code %ERRORLEVEL%.
  echo If you double-clicked the batch file, this pause keeps the window open.
  pause
  exit /b %ERRORLEVEL%
)
exit /b 0
