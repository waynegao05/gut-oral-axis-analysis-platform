[CmdletBinding()]
param(
    [string]$SourceRoot = "outputs/current_mainline_v3/topology_v7",
    [string]$ArchiveRoot = "archive/model_releases/topology_v7_generator_v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path (Join-Path $projectRoot $SourceRoot)).Path
$archivePath = Join-Path $projectRoot $ArchiveRoot
$artifactPath = Join-Path $archivePath "artifacts/current_mainline_v3_topology_v7"

if (-not $sourcePath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "SourceRoot must stay inside the project workspace."
}
if (-not $archivePath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ArchiveRoot must stay inside the project workspace."
}

New-Item -ItemType Directory -Path $artifactPath -Force | Out-Null
$sourceFiles = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -File | Sort-Object FullName)
if ($sourceFiles.Count -eq 0) {
    throw "No topology_v7 generator_v1 model artifacts were found."
}

$entries = [System.Collections.Generic.List[object]]::new()
$totalBytes = [int64]0
foreach ($sourceFile in $sourceFiles) {
    $relativePath = $sourceFile.FullName.Substring($sourcePath.Length).TrimStart('\', '/')
    $destination = Join-Path $artifactPath $relativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    $sourceHash = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $destination) {
        $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($destinationHash -ne $sourceHash) {
            throw "Archive already contains a different file: $relativePath"
        }
    }
    else {
        Copy-Item -LiteralPath $sourceFile.FullName -Destination $destination
        $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($destinationHash -ne $sourceHash) {
            throw "Archive verification failed after copying: $relativePath"
        }
    }
    $totalBytes += [int64]$sourceFile.Length
    $entries.Add([PSCustomObject]@{
        path = $relativePath.Replace('\', '/')
        bytes = [int64]$sourceFile.Length
        sha256 = $sourceHash
    })
}

$releaseFilesPath = Join-Path $archivePath "release_files"
New-Item -ItemType Directory -Path $releaseFilesPath -Force | Out-Null
foreach ($relativeFile in @("research_config_v2.yaml", "data/research/topology_v7_manifest.json")) {
    $source = Join-Path $projectRoot $relativeFile
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required generator_v1 release file is missing: $relativeFile"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $releaseFilesPath ([IO.Path]::GetFileName($source))) -Force
}

$manifest = [ordered]@{
    schema_version = 1
    release = "topology_v7_generator_v1_training_runs"
    dataset = "topology_v7"
    generator_version = "topology_v7_hybrid_generator_v1"
    archived_at_utc = [DateTime]::UtcNow.ToString("o")
    source = $SourceRoot.Replace('\', '/')
    artifact_root = "$($ArchiveRoot.Replace('\', '/'))/artifacts/current_mainline_v3_topology_v7"
    file_count = $sourceFiles.Count
    total_bytes = $totalBytes
    sha256_verified = $true
    source_files_preserved = $true
    files = $entries
}
$manifestPath = Join-Path $archivePath "archive_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Output ([PSCustomObject]@{
    source = $sourcePath
    archive = $artifactPath
    files = $sourceFiles.Count
    mib = [math]::Round($totalBytes / 1MB, 2)
    verified = $true
    manifest = $manifestPath
} | Format-List | Out-String)
