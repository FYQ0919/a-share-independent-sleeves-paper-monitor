$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
    & $PythonPath -m pip install --upgrade pip
    & $PythonPath -m pip install -e "$ProjectRoot[dev,research]"
}

& $PythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8765
