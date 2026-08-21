[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApplicationDirectory,

    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$applicationRoot = (Resolve-Path -LiteralPath $ApplicationDirectory).Path
$executable = Join-Path $applicationRoot "GutOralAxis.Desktop.exe"
$engineExecutable = Join-Path $applicationRoot "Runtime\Engine\goa-ai-engine.exe"
$applicationPri = Join-Path $applicationRoot "GutOralAxis.Desktop.pri"
$applicationXbf = Join-Path $applicationRoot "App.xbf"
$windowXbf = Join-Path $applicationRoot "MainWindow.xbf"
$applicationIcon = Join-Path $applicationRoot "Assets\AppIcon.ico"
$titleBarIcon = Join-Path $applicationRoot "Assets\AppIcon.png"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Desktop executable is missing: $executable"
}
if (-not (Test-Path -LiteralPath $engineExecutable -PathType Leaf)) {
    throw "Bundled AI Engine is missing: $engineExecutable"
}
foreach ($resource in @($applicationPri, $applicationXbf, $windowXbf, $applicationIcon, $titleBarIcon)) {
    if (-not (Test-Path -LiteralPath $resource -PathType Leaf)) {
        throw "Published WinUI resource is missing: $resource"
    }
}

$repositoryRoot = Split-Path $PSScriptRoot -Parent
$testRoot = Join-Path $repositoryRoot ".test-tmp\packaged-desktop-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
$env:GOA_DESKTOP_DATA_ROOT = $testRoot
$env:GOA_DESKTOP_SMOKE_EXIT = "1"
$env:GOA_DESKTOP_PYTHON = $null
$env:GOA_DESKTOP_ENGINE_ROOT = $null
$env:PYTHONPATH = $null

$process = Start-Process -FilePath $executable -WorkingDirectory $applicationRoot -PassThru -WindowStyle Hidden
$exited = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $exited) {
    Stop-Process -Id $process.Id
    throw "Packaged desktop smoke test timed out after $TimeoutSeconds seconds."
}
if ($process.ExitCode -ne 0) {
    throw "Packaged desktop exited with code $($process.ExitCode) before completing startup."
}

$logs = Get-ChildItem (Join-Path $testRoot "Logs") -Filter "*.log" -ErrorAction SilentlyContinue
$logText = ($logs | Get-Content -ErrorAction SilentlyContinue) -join "`n"
$requiredEvents = @(
    "application.ready",
    "engine.started",
    "application.stop_requested",
    "application.stop "
)
foreach ($eventName in $requiredEvents) {
    if ($logText -notmatch [regex]::Escape($eventName)) {
        throw "Packaged desktop smoke test did not record $eventName. Logs: $logText"
    }
}
foreach ($errorEvent in @("application.initialization_failed", "engine.start_failed")) {
    if ($logText -match [regex]::Escape($errorEvent)) {
        throw "Packaged desktop smoke test recorded $errorEvent. Logs: $logText"
    }
}
if (-not (Test-Path (Join-Path $testRoot "Data\Database\gut-oral-axis.db"))) {
    throw "Packaged desktop smoke test did not initialize SQLite."
}

[pscustomobject]@{
    status = "passed"
    application = $executable
    data_root = $testRoot
    process_exit_code = $process.ExitCode
} | ConvertTo-Json
