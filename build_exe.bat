@echo off
rem OPTIONAL and UNTESTED: builds a one-folder executable with PyInstaller for machines without Python.
rem Run run_ml_model_dojo.bat once first so that the environment exists.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VENV=%ROOT%\.venv"
if not exist "%VENV%\Scripts\python.exe" set "VENV=%USERPROFILE%\.ml_model_dojo\venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo Run run_ml_model_dojo.bat once first.
    pause
    exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install pyinstaller
set "PYTHONPATH=%ROOT%\src"
"%VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --windowed --name MLModelDojo ^
    --paths "%ROOT%\src" --collect-submodules ml_model_dojo ^
    --collect-data xgboost --collect-binaries xgboost ^
    "%ROOT%\src\ml_model_dojo\__main__.py"
echo The application folder is dist\MLModelDojo
pause
