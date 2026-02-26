$ErrorActionPreference = 'Stop'

Write-Host "Starting Ultima Alpha API..." -ForegroundColor Cyan

$root = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $root '.venv\Scripts\Activate.ps1'

if (Test-Path $venvActivate) {
    . $venvActivate
}

Set-Location $PSScriptRoot
try {
    python -m pip install -r requirements.txt
    python alpha_api.py
}
catch {
    Write-Host "Falling back to py launcher..." -ForegroundColor Yellow
    py -3 -m pip install -r requirements.txt
    py -3 alpha_api.py
}
