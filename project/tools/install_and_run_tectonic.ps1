$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path project\tools | Out-Null
$zip = 'project\tools\tectonic.zip'
Write-Output "Discovering latest Tectonic release via GitHub API..."
$api = 'https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest'
try {
    $r = Invoke-RestMethod -Uri $api -UseBasicParsing -ErrorAction Stop
    $asset = $r.assets | Where-Object { $_.name -like 'tectonic-*-x86_64-pc-windows-msvc.zip' -or $_.name -eq 'tectonic-x86_64-pc-windows-msvc.zip' } | Select-Object -First 1
    if ($asset -and $asset.browser_download_url) {
        $downloadUrl = $asset.browser_download_url
        Write-Output "Found asset: $($asset.name)"
    } else {
        Write-Output "No matching asset found in release JSON, falling back to direct latest asset URL"
        $downloadUrl = 'https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-x86_64-pc-windows-msvc.zip'
    }
} catch {
    Write-Output "GitHub API query failed: $($_.Exception.Message)"
    $downloadUrl = 'https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-x86_64-pc-windows-msvc.zip'
}
Write-Output "Downloading Tectonic (small LaTeX engine) to $zip from $downloadUrl ..."
$downloaded = $false
for ($i=1; $i -le 3; $i++) {
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zip -UseBasicParsing -ErrorAction Stop
        $downloaded = $true
        break
    } catch {
        Write-Output "Attempt $i failed: $($_.Exception.Message)"
        Start-Sleep -Seconds 2
    }
}
if (-not $downloaded) { Write-Error "Failed to download Tectonic. Please download it manually and place the zip at $zip" ; exit 3 }
Write-Output "Extracting..."
Expand-Archive -Path $zip -DestinationPath project\tools\tectonic -Force
Write-Output "Searching for tectonic.exe..."
$tectonic = Get-ChildItem -Path project\tools\tectonic -Filter tectonic.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $tectonic) { Write-Error "TECTONIC_NOT_FOUND"; exit 2 }
$tectonicPath = $tectonic.FullName
Write-Output "Found tectonic at: $tectonicPath"
# Ensure output dir
New-Item -ItemType Directory -Force -Path results | Out-Null
Write-Output "Running tectonic to compile report/report.tex -> results/report_latex_tectonic.pdf"
& $tectonicPath (Resolve-Path project\report\report.tex).Path --outdir (Resolve-Path results).Path --print=warning
if ($LASTEXITCODE -eq 0) { Write-Output "Tectonic compile succeeded" } else { Write-Error "Tectonic compile failed with code $LASTEXITCODE" ; exit $LASTEXITCODE }