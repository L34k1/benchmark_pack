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
exit /b %ERRORLEVEL%
