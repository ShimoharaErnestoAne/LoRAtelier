@echo off
rem Same as start.bat, but other PCs on the same network (intranet) can open LoRAtelier too.
rem The address to open from them is shown in this window after the start.
rem Keep this file ASCII only: cmd misreads lines with Japanese (UTF-8) text (see setup.bat).
chcp 65001 >nul
cd /d "%~dp0"
rem Setup first when it is missing or out of date; setup.bat shows its own error and pauses on failure.
"envs\app\Scripts\python.exe" setup_env.py --check >nul 2>nul
if errorlevel 1 (
  set "LS_FROM_START=1"
  call "%~dp0setup.bat"
  "envs\app\Scripts\python.exe" setup_env.py --check >nul 2>nul
  if errorlevel 1 exit /b 1
)
set "PYTHONPATH=%~dp0app"
set "LORASTUDIO_LISTEN=0.0.0.0"
"envs\app\Scripts\python.exe" -m lorastudio %*
if errorlevel 1 pause
