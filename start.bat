@echo off
rem Keep the cmd part of this file ASCII only: cmd misreads lines with Japanese (UTF-8) text, even after
rem chcp 65001, and runs pieces of them as commands (see setup.bat).  The messages are at the end of the file
rem (after the marker line), which cmd never reads; :say prints one of them with PowerShell.
chcp 65001 >nul
cd /d "%~dp0"
set "LS_SELF=%~f0"
rem Setup first when it has not finished (stopped partway, the folder moved, or an update changed the
rem dependencies).  This also finds an envspp Python that does not run.
"envspp\Scripts\python.exe" setup_env.py --check >nul 2>nul
if errorlevel 1 (
  call :say 1
  set "LS_FROM_START=1"
  call "%~dp0setup.bat"
  "envspp\Scripts\python.exe" setup_env.py --check >nul 2>nul
  if errorlevel 1 (
    call :say 2
    pause
    exit /b 1
  )
)
set "PYTHONPATH=%~dp0app"
"envspp\Scripts\python.exe" -m lorastudio %*
if errorlevel 1 pause
exit /b

:say
powershell -NoProfile -Command "$s = [IO.File]::ReadAllText($env:LS_SELF, [Text.Encoding]::UTF8); Write-Host (($s.Substring($s.IndexOf('#' + 'MESSAGES#')) -split '\r?\n')[%1])"
exit /b 0

#MESSAGES#
初回セットアップ（または前回の続き）が必要です。setup.bat を実行します。
セットアップが完了していないため起動できません。上の表示を確認して、setup.bat をもう一度実行してください。
