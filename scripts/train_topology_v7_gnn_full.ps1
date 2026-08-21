[CmdletBinding()]
param(
    [ValidateSet("single", "five-seed", "two-split")]
    [string]$Mode = "single",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$SplitSeed = 42,
    [int]$BatchSize = 4096,
    [int]$Epochs = 120,
    [int]$Patience = 25,
    [string]$Config = "config/research/research_config_v7_gnn_final.yaml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $projectRoot $Config)).Path
$outputRoot = "outputs/current_mainline_v3/topology_v7_generator_v2/gnn_identity_fullrisk_v1"
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
        $arguments = @(
            "-m", "research.repeat_runs_v2",
            "--config", $configPath,
            "--seeds"
        )
        $arguments += @($seeds | ForEach-Object { "$_" })
        $arguments += @(
            "--device", $Device,
            "--split-seed", "$currentSplit",
            "--epochs-override", "$Epochs",
            "--patience-override", "$Patience",
            "--batch-size-override", "$BatchSize",
            "--resume-existing",
            "--output-root", "$outputRoot/split${currentSplit}_five_seed"
        )
        Invoke-Python -Arguments $arguments
    }
}
finally {
    Pop-Location
}
