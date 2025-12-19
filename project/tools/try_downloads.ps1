$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force -Path project\tools | Out-Null
Write-Output "Querying GitHub API for tectonic latest release..."
try {
    $resp = Invoke-RestMethod -Uri 'https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest' -Headers @{ 'User-Agent'='ps' } -UseBasicParsing
} catch {
    Write-Output "GitHub API query failed: $($_.Exception.Message)"
    $resp = $null
}
if ($resp -and $resp.assets) {
    foreach ($a in $resp.assets) {
        if ($a.name -match 'windows|win|x86_64|msvc|pc' -and $a.name -match 'zip|zip.gz|zip.bz2') {
            $out = 'project\tools\tectonic_asset.zip'
            Write-Output "Attempting download: $($a.name) -> $out"
            try { Invoke-WebRequest -Uri $a.browser_download_url -OutFile $out -UseBasicParsing -TimeoutSec 120; Write-Output "Downloaded $out" ; exit 0 } catch { Write-Output "Failed asset download: $($_.Exception.Message)" }
        }
    }
}
# Try common Tectonic candidate URLs
$urls = @(
    'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-x86_64-pc-windows-msvc.zip',
    'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-x86_64-pc-windows-msvc.zip',
    'https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-x86_64-pc-windows-msvc.zip'
)
foreach ($u in $urls) {
    Write-Output "Trying $u"
    try { Invoke-WebRequest -Uri $u -OutFile 'project\tools\tectonic_asset.zip' -UseBasicParsing -TimeoutSec 120; Write-Output "Downloaded $u" ; exit 0 } catch { Write-Output "Failed: $($_.Exception.Message)" }
}
# Fallback: download MiKTeX basic installer from CTAN
$miktex = 'https://mirrors.ctan.org/systems/win32/miktex/setup/windows-x64/basic-miktex-24.1-x64.exe'
$outm = 'project\tools\basic-miktex-24.1-x64.exe'
Write-Output "Attempting MiKTeX download from CTAN: $miktex"
try { Invoke-WebRequest -Uri $miktex -OutFile $outm -UseBasicParsing -TimeoutSec 120; Write-Output "Downloaded MiKTeX installer to $outm" ; exit 0 } catch { Write-Output "Failed MiKTeX download: $($_.Exception.Message)" ; exit 2 }
