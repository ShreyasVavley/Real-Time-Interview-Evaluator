@echo off
REM ============================================================
REM MockIQ Setup Script
REM Run this from Anaconda Prompt (NOT regular PowerShell/CMD)
REM ============================================================

echo [1/3] Creating conda environment: mockiq (Python 3.10)...
call conda create -n mockiq python=3.10 -y

echo [2/3] Installing dependencies...
call conda activate mockiq
call pip install streamlit>=1.33.0 torch torchaudio transformers>=4.40.0 accelerate librosa soundfile openai-whisper textblob plotly

echo [3/3] Launching MockIQ...
call streamlit run app.py

pause
