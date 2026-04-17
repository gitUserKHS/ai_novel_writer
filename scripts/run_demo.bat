@echo off
setlocal
cd /d %~dp0\..

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

python -m conarrative.cli --config configs\demo.yaml init

start "" http://127.0.0.1:8000
python -m conarrative.cli --config configs\demo.yaml serve --host 127.0.0.1 --port 8000
