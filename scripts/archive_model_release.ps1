[CmdletBinding()]
param(
    [string]$SourceRoot = "outputs/current_mainline_v2",
    [string]$ArchiveRoot = "archive/model_releases/temporal_topology_v6"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path (Join-Path $projectRoot $SourceRoot)).Path
$archivePath = Join-Path $projectRoot $ArchiveRoot
$artifactPath = Join-Path $archivePath "artifacts/current_mainline_v2"

if (-not $sourcePath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "SourceRoot must stay inside the project workspace."
}
if (-not $archivePath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ArchiveRoot must stay inside the project workspace."
}

New-Item -ItemType Directory -Path $artifactPath -Force | Out-Null
$sourceFiles = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -File | Sort-Object FullName)
if ($sourceFiles.Count -eq 0) {
    throw "No model artifacts were found under $sourcePath."
}

$entries = [System.Collections.Generic.List[object]]::new()
$totalBytes = [int64]0
foreach ($sourceFile in $sourceFiles) {
    $relativePath = $sourceFile.FullName.Substring($sourcePath.Length).TrimStart('\', '/')
    $destination = Join-Path $artifactPath $relativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

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
$releaseFiles = @(
    "config/releases/temporal_topology_v6.yaml",
    "archive/datasets/topology_v6/README.md"
)
foreach ($relativeReleaseFile in $releaseFiles) {
    $releaseSource = Join-Path $projectRoot $relativeReleaseFile
    if (-not (Test-Path -LiteralPath $releaseSource)) {
        throw "Required release file is missing: $relativeReleaseFile"
    }
    $releaseDestination = Join-Path $releaseFilesPath ([System.IO.Path]::GetFileName($releaseSource))
    Copy-Item -LiteralPath $releaseSource -Destination $releaseDestination -Force
}

$manifest = [ordered]@{
    schema_version = 1
    release = "temporal_topology_aft_cross_split_consensus_v1"
    dataset = "topology_v6"
    archived_at_utc = [DateTime]::UtcNow.ToString("o")
    source = $SourceRoot.Replace('\', '/')
    artifact_root = "$($ArchiveRoot.Replace('\', '/'))/artifacts/current_mainline_v2"
    file_count = $sourceFiles.Count
    total_bytes = $totalBytes
    sha256_verified = $true
    production_config = "config/releases/temporal_topology_v6.yaml"
    dataset_archive = "archive/datasets/topology_v6"
    files = $entries
}
$manifestPath = Join-Path $archivePath "archive_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Output ([PSCustomObject]@{
    release = $manifest.release
    source = $sourcePath
    archive = $artifactPath
    files = $sourceFiles.Count
    gib = [math]::Round($totalBytes / 1GB, 3)
    verified = $true
    manifest = $manifestPath
} | Format-List | Out-String)
