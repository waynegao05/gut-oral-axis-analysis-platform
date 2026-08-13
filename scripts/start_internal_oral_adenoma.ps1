$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$env:GOA_ENABLE_INTERNAL_ORAL_ADENOMA = "1"
npm run build
python enhanced_app.py
