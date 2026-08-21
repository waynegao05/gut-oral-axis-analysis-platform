<#
.SYNOPSIS
    Build distributable artifacts (.exe setup, .msi package, checksum list) from an
    existing self-contained release directory.

.DESCRIPTION
    This script does not build the application itself. Run
    scripts/build_windows_desktop.ps1 first to produce
    artifacts/windows/TCM-Desktop-<version>-win-x64-<build-id>/ .

    Distribution formats:
      1. Portable ZIP - produced by build_windows_desktop.ps1; this script only
         records its checksum.
      2. .exe setup   - Inno Setup. Per-user install by default, downloads the
         WebView2 evergreen bootstrapper when the runtime is missing.
      3. .msi package - WiX v5. Per-machine install for managed deployment.

    This file is intentionally ASCII-only, matching the other scripts in this
    repository. Windows PowerShell 5.1 reads .ps1 files using the system ANSI
    code page unless a UTF-8 BOM is present, so non-ASCII text here would be
    mojibake on a Chinese Windows install and break the parser. User facing
    Chinese text lives in desktop/packaging/installer.iss and
    desktop/packaging/installer-notice.txt, which are saved with a UTF-8 BOM.

.PARAMETER ReleaseDirectory
    Release directory. Defaults to the newest artifacts/windows/TCM-Desktop-* .

.PARAMETER Targets
    Which formats to build: Exe, Msi, or both. Defaults to Exe.

.PARAMETER InnoSetupPath
    Path to ISCC.exe. Probed in the usual install locations when omitted.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installers.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installers.ps1 -Targets Exe,Msi
#>
[CmdletBinding()]
param(
    [string]$ReleaseDirectory = "",
    # Accepts an array (-Targets Exe,Msi when dot-sourced or via -Command) and a
    # single comma separated string ("Exe,Msi"), which is what powershell.exe
    # -File hands over because -File does not parse PowerShell array syntax.
    [string[]]$Targets = @("Exe"),
    [string]$InnoSetupPath = ""
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- helpers
# ISCC.exe is looked up in four places, in order: PATH, the default install
# directories, and finally the uninstall registry entries, which is where a
# non-default install location shows up.
function Find-InnoSetupCompiler {
    $onPath = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $defaults = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $defaults) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }

    $uninstallRoots = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    )
    foreach ($root in $uninstallRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $entries = Get-ChildItem -LiteralPath $root -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue } |
            Where-Object { $_.DisplayName -like "Inno Setup*" -and $_.InstallLocation }
        foreach ($entry in $entries) {
            $candidate = Join-Path $entry.InstallLocation "ISCC.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }

    return $null
}

# ISCC.exe opens source files through the classic Win32 file APIs, which cap a
# full path at 259 characters. This repository lives under a deep OneDrive path
# and the PyInstaller engine bundle nests training outputs several levels below
# Runtime\Engine\_internal, so the deepest source files land past that cap. The
# symptom is expensive and unhelpful: the compile spends minutes compressing,
# then aborts with
#     Error in installer.iss: The system cannot find the path specified.
# carrying neither a line number nor the name of the file that could not be
# opened.
#
# Shorten the prefix instead of the files: map the release directory onto a
# spare drive letter for the duration of the build. Inno Setup and WiX both
# record paths relative to SourceDir, so the artifacts are identical to what an
# unmapped build would have produced.
function New-ShortSourceMapping {
    param(
        [Parameter(Mandatory = $true)][string]$TargetDirectory,
        # A file that has to be reachable through the mapping. Checking it
        # catches a subst that claimed success but mapped somewhere else - for
        # example when a non-ASCII target path did not survive the console code
        # page on its way into subst.exe.
        [Parameter(Mandatory = $true)][string]$MarkerFile
    )

    $taken = @{}
    foreach ($drive in [System.IO.DriveInfo]::GetDrives()) {
        $taken[$drive.Name.Substring(0, 1).ToUpperInvariant()] = $true
    }
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        if ($drive.Name.Length -eq 1) { $taken[$drive.Name.ToUpperInvariant()] = $true }
    }

    # subst reports failures on its own streams rather than by throwing, and
    # under $ErrorActionPreference = "Stop" a redirected native stderr becomes a
    # terminating NativeCommandError. Relax it while shelling out.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        foreach ($letter in [char[]]"ZYXWVUTSRQP") {
            $name = [string]$letter
            if ($taken.ContainsKey($name)) { continue }

            & subst "${name}:" $TargetDirectory 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { continue }

            if (Test-Path -LiteralPath "${name}:\$MarkerFile" -PathType Leaf) {
                return "${name}:"
            }

            & subst "${name}:" /d 2>&1 | Out-Null
        }
    }
    finally {
        $ErrorActionPreference = $previous
    }

    return $null
}

function Remove-ShortSourceMapping {
    param([string]$Drive)

    if ([string]::IsNullOrEmpty($Drive)) { return }

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & subst $Drive /d 2>&1 | Out-Null
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

# ---------------------------------------------------------------- targets
$knownTargets = @{ "exe" = "Exe"; "msi" = "Msi" }
$requestedTargets = New-Object System.Collections.Generic.List[string]
foreach ($entry in $Targets) {
    foreach ($piece in ($entry -split "[,;\s]+")) {
        $token = $piece.Trim()
        if ([string]::IsNullOrEmpty($token)) { continue }
        $canonical = $knownTargets[$token.ToLowerInvariant()]
        if (-not $canonical) {
            throw "Unknown target '$token'. Valid targets are: Exe, Msi (for example -Targets Exe,Msi)."
        }
        if (-not $requestedTargets.Contains($canonical)) {
            $requestedTargets.Add($canonical)
        }
    }
}
if ($requestedTargets.Count -eq 0) {
    throw "No build target requested. Use -Targets Exe, -Targets Msi, or -Targets Exe,Msi."
}
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$packagingRoot = Join-Path $repositoryRoot "desktop\packaging"
$outputRoot = Join-Path $repositoryRoot "artifacts\windows"

# ---------------------------------------------------------------- release dir
if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
    $candidate = Get-ChildItem -LiteralPath $outputRoot -Directory -Filter "TCM-Desktop-*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No release directory found. Run scripts/build_windows_desktop.ps1 first."
    }
    $ReleaseDirectory = $candidate.FullName
}
$ReleaseDirectory = (Resolve-Path -LiteralPath $ReleaseDirectory).Path

foreach ($required in @("GutOralAxis.Desktop.exe", "WebUI\index.html", "Runtime\Engine\goa-ai-engine.exe", "version-manifest.json")) {
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory $required))) {
        throw "Incomplete release directory, missing ${required}: $ReleaseDirectory"
    }
}

$versions = Get-Content -Raw -LiteralPath (Join-Path $ReleaseDirectory "version-manifest.json") | ConvertFrom-Json
$appVersion = $versions.application
if ([string]::IsNullOrWhiteSpace($appVersion)) {
    throw "version-manifest.json has no 'application' field."
}

Write-Host "Release directory : $ReleaseDirectory"
Write-Host "Application       : $appVersion"
Write-Host "Model             : $($versions.model)"
Write-Host "Targets           : $($requestedTargets -join ', ')"
Write-Host ""

$produced = New-Object System.Collections.Generic.List[string]

# Keep every source path well under the 259 character Win32 limit regardless of
# how deep the repository happens to sit. See New-ShortSourceMapping above.
$shortSourceDrive = New-ShortSourceMapping -TargetDirectory $ReleaseDirectory -MarkerFile "GutOralAxis.Desktop.exe"
if ($shortSourceDrive) {
    $packagingSource = $shortSourceDrive
    Write-Host "Source mapping    : $shortSourceDrive\  ->  $ReleaseDirectory"
}
else {
    $packagingSource = $ReleaseDirectory
    Write-Warning "No spare drive letter available - packaging straight from the release directory."
    Write-Warning "If the compile aborts with 'The system cannot find the path specified' after"
    Write-Warning "compressing for a while, a source file exceeded the 259 character path limit."
    Write-Warning "Free a drive letter, or move the repository somewhere shallower, then retry."
}
Write-Host ""

try {
    # ---------------------------------------------------------------- .exe
    if ($requestedTargets -contains "Exe") {
        if ([string]::IsNullOrWhiteSpace($InnoSetupPath)) {
            $InnoSetupPath = Find-InnoSetupCompiler
            if (-not $InnoSetupPath) {
                throw "ISCC.exe not found. Install Inno Setup 6 (winget install --id JRSoftware.InnoSetup -e, or https://jrsoftware.org/isdl.php), or pass -InnoSetupPath."
            }
        }
        if (-not (Test-Path -LiteralPath $InnoSetupPath -PathType Leaf)) {
            throw "The supplied -InnoSetupPath does not exist: $InnoSetupPath"
        }
        Write-Host "Inno Setup        : $InnoSetupPath"

        # ChineseSimplified.isl is an UNOFFICIAL translation and is not shipped with
        # Inno Setup. installer.iss detects it at compile time and falls back to an
        # English-only wizard, so the build never breaks; this block only makes that
        # fallback visible instead of it being a surprise after the fact.
        $compilerIsl = Join-Path (Split-Path $InnoSetupPath -Parent) "Languages\ChineseSimplified.isl"
        $vendoredIsl = Join-Path $packagingRoot "ChineseSimplified.isl"
        if ((Test-Path -LiteralPath $vendoredIsl -PathType Leaf) -or
            (Test-Path -LiteralPath $compilerIsl -PathType Leaf)) {
            Write-Host "Wizard language   : Simplified Chinese + English"
        }
        else {
            Write-Warning "ChineseSimplified.isl not found - the setup wizard will be ENGLISH ONLY."
            Write-Host "                    Download it from https://jrsoftware.org/files/istrans/"
            Write-Host "                    and save it as either of these, then rebuild:"
            Write-Host "                      $compilerIsl"
            Write-Host "                      $vendoredIsl"
        }

        $issFile = Join-Path $packagingRoot "installer.iss"
        if (-not (Test-Path -LiteralPath $issFile)) {
            throw "Missing installer script: $issFile"
        }

        $exeBaseName = "GutOralAxis-Desktop-Setup-$appVersion-win-x64"
        $exePath = Join-Path $outputRoot "$exeBaseName.exe"
        if (Test-Path -LiteralPath $exePath) {
            throw "Setup already exists, refusing to overwrite: $exePath"
        }

        Write-Host "Building .exe setup with Inno Setup (compression is slow, please wait)..."
        & $InnoSetupPath "/DSourceDir=$packagingSource" "/DAppVersion=$appVersion" "/DOutputDir=$outputRoot" "/DOutputBaseFilename=$exeBaseName" $issFile
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE."
        }
        if (-not (Test-Path -LiteralPath $exePath)) {
            throw "Inno Setup did not produce the expected file: $exePath"
        }
        $produced.Add($exePath)
    }

    # ---------------------------------------------------------------- .msi
    if ($requestedTargets -contains "Msi") {
        if (-not (Get-Command "wix" -ErrorAction SilentlyContinue)) {
            throw "The 'wix' command was not found. Run: dotnet tool install -g wix"
        }

        $wxsFile = Join-Path $packagingRoot "GutOralAxisDesktop.wxs"
        if (-not (Test-Path -LiteralPath $wxsFile)) {
            throw "Missing WiX source: $wxsFile"
        }

        $msiPath = Join-Path $outputRoot "GutOralAxis-Desktop-$appVersion-win-x64.msi"
        if (Test-Path -LiteralPath $msiPath) {
            throw "MSI already exists, refusing to overwrite: $msiPath"
        }

        Write-Host "Building .msi package with WiX v5..."
        Push-Location $packagingRoot
        try {
            & wix build $wxsFile -d "SourceDir=$packagingSource" -d "AppVersion=$appVersion" -ext WixToolset.UI.wixext -arch x64 -o $msiPath
            if ($LASTEXITCODE -ne 0) {
                throw "WiX build failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
        $produced.Add($msiPath)
    }
}
finally {
    # Always release the drive letter, including when a build threw.
    Remove-ShortSourceMapping -Drive $shortSourceDrive
}

# ---------------------------------------------------------------- checksums
# Include the matching ZIP so downloaders can verify any of the three formats.
$zipPath = Join-Path $outputRoot "$(Split-Path $ReleaseDirectory -Leaf).zip"
if (Test-Path -LiteralPath $zipPath) {
    $produced.Add($zipPath)
}

if ($produced.Count -gt 0) {
    $sumsPath = Join-Path $outputRoot "SHA256SUMS-$appVersion.txt"
    $lines = foreach ($file in $produced) {
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        "{0}  {1}" -f $hash, (Split-Path $file -Leaf)
    }
    Set-Content -LiteralPath $sumsPath -Value $lines -Encoding ascii
    $produced.Add($sumsPath)
}

Write-Host ""
[pscustomobject]@{
    status              = "success"
    application_version = $appVersion
    model_version       = $versions.model
    release_directory   = $ReleaseDirectory
    artifacts           = @($produced | ForEach-Object {
        [pscustomobject]@{
            file    = Split-Path $_ -Leaf
            size_mb = [math]::Round((Get-Item -LiteralPath $_).Length / 1MB, 1)
        }
    })
} | ConvertTo-Json -Depth 4
