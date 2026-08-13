[CmdletBinding()]
param(
    [switch]$StartApp,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProofPath = Join-Path $ProjectRoot "state\supervisor\install-proof.json"

$PathBytes = [System.Text.Encoding]::UTF8.GetBytes($ProjectRoot.ToLowerInvariant())
$Hasher = [System.Security.Cryptography.SHA256]::Create()
try {
    $PathHash = ([System.BitConverter]::ToString($Hasher.ComputeHash($PathBytes))).Replace("-", "").Substring(0, 10).ToLowerInvariant()
} finally {
    $Hasher.Dispose()
}
$VenvRoot = Join-Path $env:LOCALAPPDATA "PocketLM\venvs\$PathHash"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

function Resolve-Python312 {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) { return @("py", "-3.12") }
    }
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if ($Python) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) { return @("python") }
    }
    throw "Python 3.12 or newer is required. Install it from python.org, then run this script again."
}

Set-Location $ProjectRoot
$PythonCommand = Resolve-Python312
if (-not (Test-Path $VenvPython)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvRoot) | Out-Null
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv $VenvRoot
    } else {
        & $PythonCommand[0] -m venv $VenvRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the PocketLM virtual environment." }
}

if (-not $SkipDependencyInstall) {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $VenvPython -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "PocketLM dependency installation failed." }
}

& $VenvPython -m pcketlm.app.chat_shell.supervisor_cli --output $ProofPath
if ($LASTEXITCODE -ne 0) { throw "PocketLM health check found a blocking compatibility issue." }

Write-Host "PocketLM installation check passed."
Write-Host "Proof: $ProofPath"
Write-Host "Environment: $VenvRoot"
Write-Host "Models are not included. Add a supported local model from the PocketLM Models screen."

if ($StartApp) {
    & $VenvPython -m pcketlm.app.web.main
}
