# Build helper for Windows PowerShell
param(
    [switch]$RunDemo,
    [switch]$Compile
)
Write-Output "Running build with RunDemo=$RunDemo, Compile=$Compile"
if ($RunDemo) {
    python .\project\tools\finalize_report.py --run-demo
}
if ($Compile) {
    python .\project\tools\finalize_report.py --compile
}
Write-Output 'Done'