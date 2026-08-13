[CmdletBinding()]
param(
    [ValidateSet("development", "audit", "full")]
    [string]$Mode = "development",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$experimentName = "topology_v7_exact_edge_v5"
$experimentRoot = "outputs/$experimentName"
$planPath = "experiments/$experimentName/experiment_plan.json"
$lockPath = "experiments/$experimentName/protocol_lock.json"
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

    Invoke-Python -Arguments @(
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
        [string]$Phase,
        [string]$DataDirectory
    )

    $arguments = @(
        "-m", "experiments.topology_v7_exact_edge_v5.runner",
        $Phase,
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

function Invoke-EdgeFidelity {
    param([string]$DataDirectory)

    Invoke-Python -Arguments @(
        "-m", "experiments.topology_v7_exact_edge_v5.edge_fidelity_diagnostic",
        "--data-dir", $DataDirectory,
        "--plan", $planPath,
        "--output", "$experimentRoot/diagnostics/edge_fidelity_summary.json"
    )
}

Push-Location $projectRoot
try {
    if ($Mode -eq "development" -or $Mode -eq "full") {
        Ensure-Cohort -DataDirectory $developmentData -Seed ([int]$plan.development_generation_seed)
        Invoke-EdgeFidelity -DataDirectory $developmentData
        Invoke-Runner -Phase "development" -DataDirectory $developmentData
    }

    if ($Mode -eq "audit" -or $Mode -eq "full") {
        $resolvedLockPath = Join-Path $projectRoot $lockPath
        if (-not (Test-Path -LiteralPath $resolvedLockPath)) {
            throw "V5 development must complete before audit generation."
        }
        $lock = Get-Content -Raw -LiteralPath $resolvedLockPath | ConvertFrom-Json
        if ($lock.status -ne "locked_after_development_before_audit_generation") {
            throw "No V5 candidate passed the performance gate; audit generation is prohibited."
        }
        Ensure-Cohort -DataDirectory $auditData -Seed ([int]$plan.audit_generation_seed)
        Invoke-Runner -Phase "audit" -DataDirectory $auditData
    }
}
finally {
    Pop-Location
}
