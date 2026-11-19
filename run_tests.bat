@echo off
rem Runs the test suite in the same virtual environment as run_ml_model_dojo.bat.
rem Run run_ml_model_dojo.bat once first so that the environment exists.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VENV=%ROOT%\.venv"
call :strlen "%ROOT%" ROOTLEN
if !ROOTLEN! GTR 80 set "VENV=%USERPROFILE%\.ml_model_dojo\venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo The environment was not found. Run run_ml_model_dojo.bat once first.
    pause
    exit /b 1
)
set "QT_QPA_PLATFORM=offscreen"
set "PYTHONPATH=%ROOT%\src"
"%VENV%\Scripts\python.exe" -m pytest %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

:strlen
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
