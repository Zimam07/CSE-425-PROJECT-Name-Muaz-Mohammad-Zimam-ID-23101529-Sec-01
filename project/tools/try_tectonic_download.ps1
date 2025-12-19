$ErrorActionPreference='Stop'
$urls = @(
    'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-x86_64-pc-windows-msvc.zip',
    'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-x86_64-pc-windows-msvc.zip',
    'https://github.com/tectonic-typesetting/tectonic/releases/download/continuous/tectonic-x86_64-pc-windows-msvc.zip',
    'https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-x86_64-pc-windows-msvc.zip'
)
$dest = 'project\tools\tectonic_try.zip'
New-Item -ItemType Directory -Force -Path project\tools | Out-Null
foreach ($u in $urls) {
    Write-Output "Trying $u"
    try {
        Invoke-WebRequest -Uri $u -OutFile $dest -UseBasicParsing -TimeoutSec 120
        Write-Output "Downloaded $u to $dest"
        Expand-Archive -Path $dest -DestinationPath project\tools\tectonic -Force
        $exe = Get-ChildItem -Path project\tools\tectonic -Filter tectonic.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($exe) { Write-Output "Found tectonic at: $($exe.FullName)" ; exit 0 }
    } catch {
        Write-Output "Failed: $($_.Exception.Message)"
    }
}
Write-Error "No working tectonic binary found from candidates"; exit 2
