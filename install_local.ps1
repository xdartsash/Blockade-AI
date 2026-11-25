$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Separate environment: never replace packages in the user's global Python.
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create .venv; install Python 3.11-3.13 x64.' }
}
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& $pythonExe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed. Check Python version, network and the error above.' }
& $pythonExe -m pip install -r requirements.txt -c constraints.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependencies install failed' }
& $pythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts found' }
& $pythonExe -I tools/check_env.py
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
Write-Host 'Installation finished. Read START_RU.md; collect and label game frames first.'
