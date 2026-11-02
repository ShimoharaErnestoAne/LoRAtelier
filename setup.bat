@echo off
rem Keep the cmd part of this file ASCII only: cmd misreads lines with Japanese (UTF-8) text, even after
rem chcp 65001, and runs pieces of them as commands.  The messages and the steps live in the PowerShell
rem part at the end of the file (after the marker line), which cmd never reads.
chcp 65001 >nul
cd /d "%~dp0"
set "LS_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = [IO.File]::ReadAllText($env:LS_SELF, [Text.Encoding]::UTF8); Invoke-Expression $s.Substring($s.IndexOf('#' + 'POWERSHELL#'))"
set "RC=%errorlevel%"
rem Called from start.bat and succeeded: continue to the app without pausing.
if "%RC%"=="0" if defined LS_FROM_START exit /b 0
pause
exit /b %RC%

#POWERSHELL#
Write-Host 'LoRAtelier 初回セットアップ（数 GB のダウンロードがあります）'
Write-Host 'ベースモデルはここでは取得しません。起動後の画面からダウンロードするか、models フォルダーに置いてください。'
# The package has no models folder: create the empty folders for model files.
foreach ($d in 'checkpoints', 'loras', 'diffusion_models', 'text_encoders', 'vae') {
  New-Item -ItemType Directory -Force -Path (Join-Path 'models' $d) | Out-Null
}

# Do not use a Python already on the PC (e.g. miniconda): its old DLLs can break torch / onnxruntime.
# Fetch uv into tools\uv and run the setup on the Python 3.11 that uv provides (nothing installed PC-wide).
# uv.exe is used only after it runs (a half-extracted uv is removed and fetched again).
$uvVer = '0.11.14'
$uv = Join-Path $PWD.Path 'tools\uv\uv.exe'
function Test-Uv { (Test-Path $uv) -and ((& $uv --version 2>$null) -ne $null) }
if (-not (Test-Uv)) {
  if (Test-Path 'tools\uv') { Remove-Item -Recurse -Force 'tools\uv' }
  Write-Host 'セットアップ用の Python を、このフォルダーの中に用意します。'
  try {
    # old Windows 10 PowerShell connects with TLS 1.0, which GitHub refuses
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $z = Join-Path $env:TEMP ('lorastudio_uv_' + [guid]::NewGuid().ToString('N') + '.zip')
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$uvVer/uv-x86_64-pc-windows-msvc.zip" -OutFile $z -ErrorAction Stop
    Expand-Archive -Force $z 'tools\uv' -ErrorAction Stop
    Remove-Item $z
  } catch {
    Write-Host $_.Exception.Message
  }
}
if (-not (Test-Uv)) {
  if (Test-Path 'tools\uv') { Remove-Item -Recurse -Force 'tools\uv' }
  Write-Host 'uv を取得できませんでした。インターネットへの接続（プロキシを含む）を確認してから、setup.bat をもう一度実行してください。'
  exit 1
}
$env:PATH = (Split-Path $uv) + ';' + $env:PATH
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD.Path 'tools\python'
& $uv run --no-project --python 3.11 --python-preference only-managed setup_env.py
exit $LASTEXITCODE
