$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path project\tools | Out-Null
$zip = 'project\tools\miktex_portable.zip'
$urls = @(
    'https://mirrors.ctan.org/systems/win32/miktex/setup/miktex-portable.zip',
    'https://miktex.org/download/ctan/systems/win32/miktex/setup/miktex-portable.zip'
)
$downloaded = $false
foreach ($u in $urls) {
    Write-Output "Attempting download from $u ..."
    for ($i=1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing -ErrorAction Stop
            $downloaded = $true
            break
        } catch {
            Write-Output "Attempt $i failed for $u: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
        }
    }
    if ($downloaded) { break }
}
if (-not $downloaded) {
    Write-Error "Failed to download MiKTeX portable from known URLs. Please download manually and place at $zip" ; exit 3
}
Write-Output "Extracting..."
Expand-Archive -Path $zip -DestinationPath project\tools\miktex_portable -Force
Write-Output "Searching for pdflatex..."
$pdflatex = Get-ChildItem -Path project\tools\miktex_portable -Filter pdflatex.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pdflatex) { Write-Output $pdflatex.FullName } else { Write-Output "PDLATEX_NOT_FOUND" ; exit 2 }