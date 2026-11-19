@echo off
rem ============================================================================
rem  ML Model Dojo - double-click launcher
rem  1. Finds Python 3.10 or newer
rem  2. Creates a private virtual environment (the system Python is not touched)
rem  3. Installs the packages in requirements.txt only when that file changed
rem  4. Checks the install, then starts the application without a console window
rem  All installer output goes to logs\setup.log
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if not exist "logs" mkdir "logs"
set "SETUP_LOG=%ROOT%\logs\setup.log"

echo ============================================================
echo   ML Model Dojo
echo ============================================================

rem --- 1. Choose where the virtual environment lives ------------------------
rem Some Qt files have very long names. Windows fails to unpack them when the
rem project sits in a deep folder (for example inside OneDrive), so for long
rem paths we use a short folder under your user profile instead (not under AppData, because Microsoft Store Python redirects AppData writes) of .venv.
set "VENV=%ROOT%\.venv"
call :strlen "%ROOT%" ROOTLEN
if !ROOTLEN! GTR 80 set "VENV=%USERPROFILE%\.ml_model_dojo\venv"
set "VPY=%VENV%\Scripts\python.exe"
set "VPYW=%VENV%\Scripts\pythonw.exe"
set "HASHFILE=%VENV%\requirements.sha256"

rem --- 2. Find Python 3.10 or newer -----------------------------------------
set "PYCMD="
if exist "%VPY%" goto :have_venv
rem Prefer versions that have wheels for every package (3.12, 3.11, 3.13, 3.10).
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.12 3.11 3.13 3.10) do (
        if not defined PYCMD (
            py -%%V -c "import sys" >nul 2>nul
            if not errorlevel 1 set "PYCMD=py -%%V"
        )
    )
)
if not defined PYCMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYCMD=py -3"
    )
)
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYCMD=python"
    )
)
if not defined PYCMD goto :no_python

rem --- 3. Create the virtual environment -------------------------------------
echo Creating a private Python environment (first run only)...
echo [%date% %time%] Creating venv at %VENV% >> "%SETUP_LOG%"
%PYCMD% -m venv "%VENV%" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail

:have_venv
rem --- 4. Install packages only when requirements.txt changed ----------------
set "NEWHASH="
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile "%ROOT%\requirements.txt" SHA256 ^| findstr /v /c:"hash of" /c:"CertUtil"') do (
    if not defined NEWHASH set "NEWHASH=%%H"
)
set "NEWHASH=!NEWHASH: =!"
set "OLDHASH="
if exist "%HASHFILE%" set /p OLDHASH=<"%HASHFILE%"
if /i "!NEWHASH!"=="!OLDHASH!" if defined NEWHASH goto :check

echo Installing packages. The first run may take a few minutes and needs an
echo internet connection. Later runs start quickly and work offline.
echo [%date% %time%] Installing requirements >> "%SETUP_LOG%"
"%VPY%" -m pip install --upgrade pip >> "%SETUP_LOG%" 2>&1
"%VPY%" -m pip install -r "%ROOT%\requirements.txt" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail
> "%HASHFILE%" echo !NEWHASH!

:check
rem --- 5. Pre-flight check ---------------------------------------------------
echo Checking packages...
set "PYTHONPATH=%ROOT%\src"
"%VPY%" -c "import PySide6, sklearn, xgboost, matplotlib, numpy, scipy" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 (
    echo A package is missing or broken. Reinstalling once...
    "%VPY%" -m pip install -r "%ROOT%\requirements.txt" >> "%SETUP_LOG%" 2>&1
    "%VPY%" -c "import PySide6, sklearn, xgboost, matplotlib, numpy, scipy" >> "%SETUP_LOG%" 2>&1
    if errorlevel 1 goto :fail
    > "%HASHFILE%" echo !NEWHASH!
)

rem --- 6. Start the application (no console window stays open) ---------------
echo Starting the playground...
start "" "%VPYW%" -m ml_model_dojo
exit /b 0

:no_python
echo.
echo Python 3.10 or newer was not found.
echo Please install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during the installation.
where winget >nul 2>nul
if not errorlevel 1 (
    echo.
    set /p ANS=Install Python 3.12 now with winget? [y/N]:
    if /i "!ANS!"=="y" (
        winget install -e --id Python.Python.3.12
        echo.
        echo Close this window and double-click run_ml_model_dojo.bat again.
    )
)
pause
exit /b 1

:fail
echo.
echo Something went wrong during setup.
echo Details were written to: %SETUP_LOG%
echo The last lines of the log are:
echo ------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -Path '%SETUP_LOG%' -Tail 12"
echo ------------------------------------------------------------
pause
exit /b 1

:strlen
rem usage: call :strlen "text" outvar
setlocal EnableDelayedExpansion
set "s=%~1"
set "n=0"
:strlen_loop
if defined s (
    set "s=!s:~1!"
    set /a n+=1
    goto :strlen_loop
)
endlocal & set "%~2=%n%"
exit /b 0
