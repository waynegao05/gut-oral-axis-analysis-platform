[CmdletBinding()]
param(
    [ValidateSet("smoke", "single", "five-seed", "two-split")]
    [string]$Mode = "smoke",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$SplitSeed = 42,
    [int]$BatchSize = 256,
    [int]$Epochs = 140,
    [int]$Patience = 24,
    [string]$Config = "research_config_v7_gnn_optimized.yaml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $projectRoot $Config)).Path
$outputRoot = "outputs/current_mainline_v3/topology_v7_generator_v2/gnn_optimized_v1"
$seeds = @(7, 21, 42, 123, 2026)

function Invoke-Python {
    param([string[]]$Arguments)

    if ($DryRun) {
        [PSCustomObject]@{
            executable = "python"
            arguments = $Arguments
        } | ConvertTo-Json -Compress -Depth 3
        return
    }

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    if ($Mode -eq "smoke") {
        Invoke-Python @(
            "-m", "research.train_v2",
            "--config", $configPath,
            "--device", $Device,
            "--split-seed", "$SplitSeed",
            "--epochs-override", "8",
            "--patience-override", "8",
            "--batch-size-override", "$BatchSize",
            "--output-dir-override", "$outputRoot/smoke/split${SplitSeed}_seed42"
        )
        return
    }

    if ($Mode -eq "single") {
        Invoke-Python @(
            "-m", "research.train_v2",
            "--config", $configPath,
            "--device", $Device,
            "--split-seed", "$SplitSeed",
            "--epochs-override", "$Epochs",
            "--patience-override", "$Patience",
            "--batch-size-override", "$BatchSize",
            "--output-dir-override", "$outputRoot/single/split${SplitSeed}_seed42"
        )
        return
    }

    $splitSeeds = if ($Mode -eq "two-split") { @(42, 43) } else { @($SplitSeed) }
    foreach ($currentSplit in $splitSeeds) {
        $repeatArguments = @(
            "-m", "research.repeat_runs_v2",
            "--config", $configPath,
            "--seeds"
        )
        $repeatArguments += @($seeds | ForEach-Object { "$_" })
        $repeatArguments += @(
            "--device", $Device,
            "--split-seed", "$currentSplit",
            "--epochs-override", "$Epochs",
            "--patience-override", "$Patience",
            "--batch-size-override", "$BatchSize",
            "--output-root", "$outputRoot/split${currentSplit}_five_seed"
        )
        Invoke-Python -Arguments $repeatArguments
    }
}
finally {
    Pop-Location
}
