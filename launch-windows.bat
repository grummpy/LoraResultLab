@echo off
setlocal
cd /d "%~dp0"
set LORALAB_HOST=127.0.0.1
if "%LORALAB_PORT%"=="" set LORALAB_PORT=8765
if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo LoraResultLab — ComfyUI submission is OFF unless you enable it in Settings and confirm.
start "" "http://127.0.0.1:%LORALAB_PORT%"
python -m app
