@echo off
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe -m streamlit run demo\app.py
) else (
    python -m streamlit run demo\app.py
)
pause
