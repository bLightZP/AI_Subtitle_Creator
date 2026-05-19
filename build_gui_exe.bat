@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo Missing virtual environment: "%ROOT%.venv"
    echo Run this first from the project folder:
    echo python -m venv .venv
    echo .venv\Scripts\python.exe -m pip install -e ".[dev,build]"
    exit /b 1
)

"%ROOT%.venv\Scripts\python.exe" -m pip install -e ".[build]"
if errorlevel 1 exit /b %errorlevel%

"%ROOT%.venv\Scripts\pyinstaller.exe" --clean --noconfirm "%ROOT%aisub-gui.spec"
if errorlevel 1 exit /b %errorlevel%

echo Built "%ROOT%dist\aisub-gui.exe"

