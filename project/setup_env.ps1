# PowerShell setup script for Windows
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Environment set up. Activate via '.\.venv\Scripts\Activate.ps1'"