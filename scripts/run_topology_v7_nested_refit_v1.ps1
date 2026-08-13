[CmdletBinding()]
param(
    [ValidateSet("smoke", "development", "audit", "full")]
    [string]$Mode = "smoke",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$experimentRoot = "outputs/topology_v7_nested_refit_v1"
$planPath = "experiments/topology_v7_nested_refit_v1/experiment_plan.json"
$plan = Get-Content -Raw -LiteralPath (Join-Path $projectRoot $planPath) | ConvertFrom-Json
$developmentData = "$experimentRoot/cohorts/development_seed$($plan.development_generation_seed)"
$auditData = "$experimentRoot/cohorts/audit_seed$($plan.audit_generation_seed)"
$sourceSnapshot = "$experimentRoot/source_snapshot/topology_v6"

function Invoke-Python {
    param([string[]]$Arguments)

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Ensure-Cohort {
    param(
        [string]$DataDirectory,
        [int]$Seed
    )

    $manifestPath = Join-Path $projectRoot "$DataDirectory/topology_v7_manifest.json"
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ([int]$manifest.seed -ne $Seed) {
            throw "Existing cohort seed does not match the locked seed: $manifestPath"
        }
        Write-Host "Verified existing cohort: $DataDirectory"
        return
    }

    Invoke-Python @(
        "-m", "experiments.public_data_v1.build_topology_v7_v3",
        "--samples", "$($plan.sample_count)",
        "--seed", "$Seed",
        "--output-dir", $DataDirectory,
        "--archive-dir", $sourceSnapshot,
        "--local-anchor-weight", "0.55",
        "--anchor-balance-searches", "20000",
        "--latent-noise-scale", "0.80"
    )
}

function Invoke-Runner {
    param(
        [string]$RunMode,
        [string]$DataDirectory
    )

    $arguments = @(
        "-m", "experiments.topology_v7_nested_refit_v1.runner",
        "--mode", $RunMode,
        "--data-dir", $DataDirectory,
        "--output-root", $experimentRoot,
        "--plan", $planPath,
        "--device", $Device
    )
    if ($Resume) {
        $arguments += "--resume"
    }
    Invoke-Python -Arguments $arguments
}

Push-Location $projectRoot
try {
    if ($Mode -eq "smoke") {
        Invoke-Runner -RunMode "smoke" -DataDirectory "data/research/topology_v7_generator_v3"
        return
    }

    if ($Mode -eq "development" -or $Mode -eq "full") {
        Ensure-Cohort -DataDirectory $developmentData -Seed ([int]$plan.development_generation_seed)
        Invoke-Runner -RunMode "development" -DataDirectory $developmentData
    }

    if ($Mode -eq "audit" -or $Mode -eq "full") {
        $lockPath = Join-Path $projectRoot "experiments/topology_v7_nested_refit_v1/protocol_lock.json"
        if (-not (Test-Path -LiteralPath $lockPath)) {
            throw "Development must complete and lock the protocol before audit generation."
        }
        Ensure-Cohort -DataDirectory $auditData -Seed ([int]$plan.audit_generation_seed)
        Invoke-Runner -RunMode "audit" -DataDirectory $auditData
    }
}
finally {
    Pop-Location
}
