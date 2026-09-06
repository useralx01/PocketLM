param(
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$TaskName = "PocketLM-PLM13-Watchdog"
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path -LiteralPath $RepoRoot).Path
$logDir = Join-Path $repoPath "state\auto_resume"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stopPath = Join-Path $logDir "plm13-watchdog.stop"
$pidPath = Join-Path $logDir "plm13-watchdog.pid"

Set-Content -LiteralPath $stopPath -Value ((Get-Date).ToString("o"))
Write-Output "Stop file written: $stopPath"

if (Test-Path -LiteralPath $pidPath) {
    $pidText = (Get-Content -Raw -LiteralPath $pidPath).Trim()
    $processId = 0
    if ([int]::TryParse($pidText, [ref]$processId)) {
        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            Write-Output "Stopped watchdog process id $processId"
        }
    }
}

& schtasks.exe /End /TN $TaskName *> $null
Write-Output "Watchdog stop requested. Disable startup with: schtasks /Change /TN $TaskName /DISABLE"
