[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateSet("auto", "online", "verified-local")]
    [string]$NuGetMode = "auto",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$AiEngineBundle = "",
    [switch]$EnableInternalOralAdenoma,
    [switch]$RunSmoke
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$desktopRoot = Join-Path $repositoryRoot "desktop"
$project = Join-Path $desktopRoot "src\GutOralAxis.Desktop\GutOralAxis.Desktop.csproj"
$versionManifest = Join-Path $desktopRoot "src\GutOralAxis.Desktop\version-manifest.json"
$versions = Get-Content -Raw -LiteralPath $versionManifest | ConvertFrom-Json
$buildId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$releaseName = "TCM-Desktop-$($versions.application)-$RuntimeIdentifier-$buildId"
$releaseRoot = Join-Path $repositoryRoot "artifacts\windows\$releaseName"
$archivePath = "$releaseRoot.zip"
if (Test-Path -LiteralPath $releaseRoot) {
    throw "Release directory already exists: $releaseRoot"
}
if (Test-Path -LiteralPath $archivePath) {
    throw "Release archive already exists: $archivePath"
}

Push-Location $repositoryRoot
try {
    & node "node_modules/typescript/bin/tsc" --project "frontend/tsconfig.json" --noEmit
    if ($LASTEXITCODE -ne 0) { throw "TypeScript typecheck failed." }
    & node --test "frontend/tests/transport.test.mjs"
    if ($LASTEXITCODE -ne 0) { throw "Frontend transport tests failed." }
    & node "node_modules/esbuild/bin/esbuild" "frontend/src/main.ts" --bundle --minify --format=iife --target=es2020 --legal-comments=none --outfile="static/generated/app.js"
    if ($LASTEXITCODE -ne 0) { throw "Frontend production build failed." }

    $webArguments = @("scripts/build_desktop_web.py")
    if ($EnableInternalOralAdenoma) {
        $webArguments += "--enable-internal-oral-adenoma"
    }
    & $Python @webArguments
    if ($LASTEXITCODE -ne 0) { throw "Desktop WebUI build failed." }

    if ([string]::IsNullOrWhiteSpace($AiEngineBundle)) {
        $engineOutput = Join-Path $repositoryRoot "artifacts\ai-engine\$buildId"
        $engineWork = Join-Path $repositoryRoot ".test-tmp\pyinstaller-$buildId"
        $env:APPDATA = Join-Path $repositoryRoot ".test-tmp\pyinstaller-appdata-$buildId"
        $env:PYTHONNOUSERSITE = "1"
        & $Python "scripts/build_ai_engine_bundle.py" --output-dir $engineOutput --work-dir $engineWork
        if ($LASTEXITCODE -ne 0) { throw "Standalone AI Engine build failed." }
        $AiEngineBundle = Join-Path $engineOutput "goa-ai-engine"
    }
    $AiEngineBundle = (Resolve-Path -LiteralPath $AiEngineBundle).Path
    if (-not (Test-Path -LiteralPath (Join-Path $AiEngineBundle "goa-ai-engine.exe") -PathType Leaf)) {
        throw "AI Engine bundle is invalid: $AiEngineBundle"
    }

    $localNuGet = Join-Path $desktopRoot ".local-nuget"
    if ($NuGetMode -eq "auto") {
        $NuGetMode = if (Test-Path (Join-Path $localNuGet "microsoft.windowsappsdk.2.3.1.nupkg")) {
            "verified-local"
        } else {
            "online"
        }
    }
    $nugetConfig = if ($NuGetMode -eq "verified-local") {
        Join-Path $desktopRoot "NuGet.Local.Config"
    } else {
        Join-Path $desktopRoot "NuGet.Config"
    }

    $dotnetState = Join-Path $repositoryRoot ".test-tmp\dotnet-desktop-build-$buildId"
    New-Item -ItemType Directory -Path $dotnetState -Force | Out-Null
    $env:APPDATA = Join-Path $dotnetState "appdata"
    $env:DOTNET_CLI_HOME = Join-Path $dotnetState "home"
    $env:NUGET_PACKAGES = Join-Path $desktopRoot ".packages"
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

    & dotnet restore $project --configfile $nugetConfig --runtime $RuntimeIdentifier
    if ($LASTEXITCODE -ne 0) { throw "Desktop NuGet restore failed." }
    & dotnet publish $project --configuration Release --runtime $RuntimeIdentifier --self-contained true --no-restore --output $releaseRoot -p:PublishReadyToRun=false
    if ($LASTEXITCODE -ne 0) { throw "Desktop publish failed." }

    $engineDestination = Join-Path $releaseRoot "Runtime\Engine"
    New-Item -ItemType Directory -Path $engineDestination -Force | Out-Null
    Copy-Item -Path (Join-Path $AiEngineBundle "*") -Destination $engineDestination -Recurse

    @{
        enable_internal_oral_adenoma = [bool]$EnableInternalOralAdenoma
        allow_development_engine_fallback = $false
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $releaseRoot "desktop-settings.json") -Encoding utf8

    $requiredReleaseFiles = @(
        "GutOralAxis.Desktop.exe",
        "GutOralAxis.Desktop.pri",
        "App.xbf",
        "MainWindow.xbf",
        "Assets\AppIcon.png",
        "Assets\AppIcon.ico",
        "WebUI\index.html",
        "WebUI\assets\app.css",
        "WebUI\assets\app.js",
        "Runtime\Engine\goa-ai-engine.exe",
        "Runtime\Engine\python-dependencies.json",
        "Runtime\Engine\runtime-integrity.json",
        "version-manifest.json",
        "desktop-settings.json"
    )
    foreach ($relativePath in $requiredReleaseFiles) {
        $requiredPath = Join-Path $releaseRoot $relativePath
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required release file is missing: $relativePath"
        }
    }

    & $Python "scripts/generate_release_manifest.py" --root $releaseRoot --output (Join-Path $releaseRoot "release-integrity.json") --version-manifest (Join-Path $releaseRoot "version-manifest.json")
    if ($LASTEXITCODE -ne 0) { throw "Release integrity manifest generation failed." }
    Compress-Archive -Path (Join-Path $releaseRoot "*") -DestinationPath $archivePath -CompressionLevel Optimal

    if ($RunSmoke) {
        & (Join-Path $PSScriptRoot "smoke_windows_desktop.ps1") -ApplicationDirectory $releaseRoot
        if ($LASTEXITCODE -ne 0) { throw "Packaged desktop smoke test failed." }
    }

    [pscustomobject]@{
        status = "success"
        release_directory = $releaseRoot
        portable_archive = $archivePath
        application_version = $versions.application
        model_version = $versions.model
        nuget_mode = $NuGetMode
    } | ConvertTo-Json
}
finally {
    Pop-Location
}
