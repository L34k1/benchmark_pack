@echo off
setlocal enableextensions

REM Run one of each bench with a 1800s window and 64 channels.
REM Update DATA_FILE (and DATA_DIR if needed) to point at a real EDF/NWB file.
set DATA_DIR=data
set DATA_FILE=%DATA_DIR%\example.edf
set FORMAT=edf
set TAG=one_each_1800s_64ch

if not exist "%DATA_FILE%" (
  echo ERROR: data file not found: %DATA_FILE%
  echo Please update DATA_FILE in run_one_each.bat to a real EDF/NWB file.
  exit /b 1
)

echo === IO benches ===
python scripts\io\bench_io_v2.py --tag %TAG% --format %FORMAT% --data-dir %DATA_DIR% --file "%DATA_FILE%" --n-files 1 --runs 1 --window-s 1800 --n-ch 64
python scripts\io\bench_io.py --tag %TAG% --data-dir %DATA_DIR% --n-files 1 --runs 1 --windows 1800 --n-channels 64

echo === Desktop benches ===
python scripts\desktop\bench_fastplotlib.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_mne_rawplot.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_vispy.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
python scripts\desktop\bench_datoviz.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
python scripts\desktop\bench_pyqtgraph_tffr_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

python scripts\desktop\bench_vispy_A1_throughput.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_vispy_A2_cadenced.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

python scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_pyqtgraph_A2_cadenced_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

python scripts\desktop\bench_fastplotlib.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_fastplotlib.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

python scripts\desktop\bench_mne_rawplot.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
python scripts\desktop\bench_mne_rawplot.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

python scripts\desktop\bench_vispy.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
python scripts\desktop\bench_vispy.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1

python scripts\desktop\bench_datoviz.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
python scripts\desktop\bench_datoviz.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1

echo === Web benches (HTML generation) ===
python scripts\web\bench_plotly_A1_throughput.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-channels 64
python scripts\web\bench_plotly_A2_cadenced.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-channels 64

endlocal
