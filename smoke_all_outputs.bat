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

call gen_synth_one.bat EDF
if errorlevel 1 goto fail

set "DATA_EDF=%ROOT%data\_synth\synth_8ch_60s_250hz.edf"
set "OUT_ROOT=%ROOT%outputs\smoke"
set "WINDOW_S=60"
set "N_CH=8"
set "STEPS=200"
set "INTERVAL_MS=16.666"

for %%S in (PAN ZOOM_IN PAN_ZOOM) do (
  call :run_tffr VIS_PYQTGRAPH scripts\desktop\bench_pyqtgraph_tffr_v2.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_PYQTGRAPH scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_PYQTGRAPH scripts\desktop\bench_pyqtgraph_A2_cadenced_v2.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_VISPY scripts\desktop\bench_vispy.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_VISPY scripts\desktop\bench_vispy.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_VISPY scripts\desktop\bench_vispy.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_DATOVIZ scripts\desktop\bench_datoviz.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_DATOVIZ scripts\desktop\bench_datoviz.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_DATOVIZ scripts\desktop\bench_datoviz.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_FASTPLOTLIB scripts\desktop\bench_fastplotlib.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_FASTPLOTLIB scripts\desktop\bench_fastplotlib.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_FASTPLOTLIB scripts\desktop\bench_fastplotlib.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_MNE_RAWPLOT scripts\desktop\bench_mne_rawplot.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_MNE_RAWPLOT scripts\desktop\bench_mne_rawplot.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_MNE_RAWPLOT scripts\desktop\bench_mne_rawplot.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_PLOTLY scripts\web\bench_plotly_tffr.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_PLOTLY scripts\web\bench_plotly_A1_throughput.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_PLOTLY scripts\web\bench_plotly_A2_cadenced.py %%S
  if errorlevel 1 goto fail

  call :run_tffr VIS_D3_CANVAS scripts\web\gen_d3_html.py %%S
  if errorlevel 1 goto fail
  call :run_a1 VIS_D3_CANVAS scripts\web\gen_d3_html.py %%S
  if errorlevel 1 goto fail
  call :run_a2 VIS_D3_CANVAS scripts\web\gen_d3_html.py %%S
  if errorlevel 1 goto fail
)

echo Smoke outputs complete.
pause
exit /b 0

:fail
echo.
echo Smoke run failed with exit code %ERRORLEVEL%.
echo If you double-clicked the batch file, this pause keeps the window open.
pause
exit /b %ERRORLEVEL%

:run_tffr
set "TOOL=%~1"
set "SCRIPT=%~2"
set "SEQ=%~3"
set "TAG=smoke_%TOOL%_TFFR_%SEQ%"
if "%TOOL%"=="VIS_PYQTGRAPH" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
) else if "%TOOL%"=="VIS_PLOTLY" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
) else (
  "%PYTHON%" "%SCRIPT%" --bench-id TFFR --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
)
if errorlevel 1 exit /b %ERRORLEVEL%
"%PYTHON%" -m benchkit.smoke --tffr-csv "%OUT_ROOT%\TFFR\%TOOL%\%TAG%\tffr.csv"
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0

:run_a1
set "TOOL=%~1"
set "SCRIPT=%~2"
set "SEQ=%~3"
set "TAG=smoke_%TOOL%_A1_%SEQ%"
if "%TOOL%"=="VIS_PYQTGRAPH" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
) else if "%TOOL%"=="VIS_PLOTLY" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
) else if "%TOOL%"=="VIS_D3_CANVAS" (
  "%PYTHON%" "%SCRIPT%" --bench-id A1_THROUGHPUT --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
) else (
  "%PYTHON%" "%SCRIPT%" --bench-id A1_THROUGHPUT --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --out-root "%OUT_ROOT%" --tag "%TAG%"
)
if errorlevel 1 exit /b %ERRORLEVEL%
"%PYTHON%" -m benchkit.smoke --steps-csv "%OUT_ROOT%\A1_THROUGHPUT\%TOOL%\%TAG%\steps.csv" --expected-steps %STEPS%
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0

:run_a2
set "TOOL=%~1"
set "SCRIPT=%~2"
set "SEQ=%~3"
set "TAG=smoke_%TOOL%_A2_%SEQ%"
if "%TOOL%"=="VIS_PYQTGRAPH" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --target-interval-ms %INTERVAL_MS% --out-root "%OUT_ROOT%" --tag "%TAG%"
) else if "%TOOL%"=="VIS_PLOTLY" (
  "%PYTHON%" "%SCRIPT%" --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --target-interval-ms %INTERVAL_MS% --out-root "%OUT_ROOT%" --tag "%TAG%"
) else if "%TOOL%"=="VIS_D3_CANVAS" (
  "%PYTHON%" "%SCRIPT%" --bench-id A2_CADENCED --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --target-interval-ms %INTERVAL_MS% --out-root "%OUT_ROOT%" --tag "%TAG%"
) else (
  "%PYTHON%" "%SCRIPT%" --bench-id A2_CADENCED --format EDF --file "%DATA_EDF%" --window-s %WINDOW_S% --n-ch %N_CH% --sequence %SEQ% --steps %STEPS% --runs 1 --target-interval-ms %INTERVAL_MS% --out-root "%OUT_ROOT%" --tag "%TAG%"
)
if errorlevel 1 exit /b %ERRORLEVEL%
"%PYTHON%" -m benchkit.smoke --steps-csv "%OUT_ROOT%\A2_CADENCED\%TOOL%\%TAG%\steps.csv" --expected-steps %STEPS%
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0
