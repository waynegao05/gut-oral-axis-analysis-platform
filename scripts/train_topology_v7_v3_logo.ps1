[CmdletBinding()]
param(
    [ValidateSet("smoke", "fold", "full")]
    [string]$Mode = "smoke",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateRange(0, 4)]
    [int]$TestGroup = 0,
    [int]$BatchSize = 4096,
    [int]$Epochs = 120,
    [int]$Patience = 25,
    [string]$Config = "research_config_v7_v3_gnn_locked.yaml",
    [switch]$ResumeExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $projectRoot $Config)).Path
$outputRoot = "outputs/topology_v7_generator_v3_formal/gnn_locked_logo"
$seeds = @(7, 21, 42, 123, 2026)

function Invoke-Python {
    param([string[]]$Arguments)

    if ($DryRun) {
        [PSCustomObject]@{
            executable = "python"
            arguments = $Arguments
        } | ConvertTo-Json -Compress -Depth 4
        return
    }

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Fold {
    param(
        [ValidateRange(0, 4)]
        [int]$OuterTestGroup
    )

    $validationGroup = ($OuterTestGroup + 1) % 5
    $foldRoot = "$outputRoot/outer_group${OuterTestGroup}_val${validationGroup}_five_seed"
    $repeatArguments = @(
        "-m", "research.repeat_runs_v2",
        "--config", $configPath,
        "--seeds"
    )
    $repeatArguments += @($seeds | ForEach-Object { "$_" })
    $repeatArguments += @(
        "--device", $Device,
        "--split-seed", "42",
        "--validation-group", "$validationGroup",
        "--test-group", "$OuterTestGroup",
        "--epochs-override", "$Epochs",
        "--patience-override", "$Patience",
        "--batch-size-override", "$BatchSize",
        "--output-root", $foldRoot
    )
    if ($ResumeExisting) {
        $repeatArguments += "--resume-existing"
    }
    Invoke-Python -Arguments $repeatArguments

    $checkpointGlob = "$foldRoot/research_seed*/best_model.pt"
    foreach ($splitName in @("train", "val", "test")) {
        Invoke-Python @(
            "-m", "research.ensemble_v2",
            "--config", $configPath,
            "--checkpoint-glob", $checkpointGlob,
            "--split", $splitName,
            "--device", $Device,
            "--split-seed", "42",
            "--validation-group", "$validationGroup",
            "--test-group", "$OuterTestGroup",
            "--output", "$foldRoot/${splitName}_ensemble_summary.json",
            "--quiet"
        )
    }
}

Push-Location $projectRoot
try {
    if ($Mode -eq "smoke") {
        $validationGroup = ($TestGroup + 1) % 5
        Invoke-Python @(
            "-m", "research.train_v2",
            "--config", $configPath,
            "--device", $Device,
            "--split-seed", "42",
            "--validation-group", "$validationGroup",
            "--test-group", "$TestGroup",
            "--epochs-override", "2",
            "--patience-override", "1",
            "--batch-size-override", "$BatchSize",
            "--output-dir-override", "$outputRoot/smoke/outer_group${TestGroup}_val${validationGroup}_seed42"
        )
        return
    }

    if ($Mode -eq "fold") {
        Invoke-Fold -OuterTestGroup $TestGroup
        return
    }

    foreach ($outerGroup in 0..4) {
        Invoke-Fold -OuterTestGroup $outerGroup
    }
}
finally {
    Pop-Location
}
