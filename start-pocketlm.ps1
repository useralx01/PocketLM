$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PathBytes = [System.Text.Encoding]::UTF8.GetBytes($ProjectRoot.ToLowerInvariant())
$Hasher = [System.Security.Cryptography.SHA256]::Create()
try {
    $PathHash = ([System.BitConverter]::ToString($Hasher.ComputeHash($PathBytes))).Replace("-", "").Substring(0, 10).ToLowerInvariant()
} finally {
    $Hasher.Dispose()
}
$Python = Join-Path $env:LOCALAPPDATA "PocketLM\venvs\$PathHash\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "PocketLM is not installed yet. Run .\install-pocketlm.ps1 first."
}
Set-Location $ProjectRoot
& $Python -m pcketlm.app.web.main
