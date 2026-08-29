$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "dashboard.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found: $pythonPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000 *>> $logPath
