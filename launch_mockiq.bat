@echo off
REM ============================================================
REM MockIQ Launch Script
REM Double-click this from File Explorer or run from any terminal
REM ============================================================
SET "CONDA_ROOT=C:\Users\B10426\anaconda3"
SET "VENV=%~dp0..\mockiq_env"
SET "PATH=%CONDA_ROOT%;%CONDA_ROOT%\Library\bin;%CONDA_ROOT%\DLLs;%CONDA_ROOT%\Scripts;%PATH%"
SET "PYTHONIOENCODING=utf-8"

echo Starting MockIQ AI Interview Evaluator...
echo.
"%VENV%\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.headless false --server.port 8501 --browser.gatherUsageStats false
pause
