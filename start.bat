@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem セットアップが最後まで終わっていない（途中で止まった・フォルダーを移動した・更新で依存関係が
rem 変わった）ときは、先にセットアップを実行する。envs\app の Python が動かない場合もここで分かる。
"envs\app\Scripts\python.exe" setup_env.py --check >nul 2>nul
if errorlevel 1 (
  echo 初回セットアップ（または前回の続き）が必要です。setup.bat を実行します。
  set "LS_FROM_START=1"
  call "%~dp0setup.bat"
  "envs\app\Scripts\python.exe" setup_env.py --check >nul 2>nul
  if errorlevel 1 (
    echo セットアップが完了していないため起動できません。上の表示を確認して、setup.bat をもう一度実行してください。
    pause
    exit /b 1
  )
)
set "PYTHONPATH=%~dp0app"
"envs\app\Scripts\python.exe" -m lorastudio %*
if errorlevel 1 pause
