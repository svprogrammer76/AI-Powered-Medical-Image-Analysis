@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv312\Scripts\python.exe"
if exist "%PYTHON%" goto run
set "PYTHON=.venv\Scripts\python.exe"
if exist "%PYTHON%" goto run

echo Project environment not found. Run setup_windows.bat first.
exit /b 1

:run
"%PYTHON%" main.py %*
