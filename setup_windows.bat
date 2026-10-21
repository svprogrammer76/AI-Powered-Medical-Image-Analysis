@echo off
setlocal
cd /d "%~dp0"

REM Reuse an existing Python 3.12 project environment if present.
set "PYTHON=.venv312\Scripts\python.exe"
if exist "%PYTHON%" goto install
set "PYTHON=.venv\Scripts\python.exe"
if exist "%PYTHON%" goto install

REM Do not use the bare `python` command: it may resolve to another version.
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Could not create the environment with Python 3.12.
  echo Install Python 3.12 from https://www.python.org/downloads/ and enable the Python launcher,
  echo then run this file again. You do not need to activate the environment.
  exit /b 1
)
set "PYTHON=.venv\Scripts\python.exe"

:install
echo Using:
"%PYTHON%" --version
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Setup complete. Run the project with run_windows.bat train --epochs 15
