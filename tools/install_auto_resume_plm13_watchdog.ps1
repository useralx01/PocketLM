param(
    [string]$TaskName = "PocketLM-PLM13-Watchdog",
    [string]$OldTaskName = "PocketLM-PLM13-AutoResume",
    [string]$RepoRoot = "C:\Users\isale\Documents\pcketlm",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path -LiteralPath $RepoRoot).Path
$scriptPath = Join-Path $repoPath "tools\auto_resume_plm13_watchdog.ps1"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Watchdog script not found: $scriptPath"
}

$logDir = Join-Path $repoPath "state\auto_resume"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& cmd.exe /c "schtasks /Query /TN `"$OldTaskName`" >nul 2>nul"
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /TN $OldTaskName /F | Write-Output
}

& cmd.exe /c "schtasks /Query /TN `"$TaskName`" >nul 2>nul"
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /TN $TaskName /F | Write-Output
}

$principal = [Security.Principal.WindowsIdentity]::GetCurrent()
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$registeredMode = "ScheduledTask-OnLogon"
& schtasks.exe /Create /TN $TaskName /SC ONLOGON /TR $action /F | Write-Output
if ($LASTEXITCODE -ne 0) {
    $registeredMode = "StartupFolder"
    $startupDir = [Environment]::GetFolderPath("Startup")
    if ([string]::IsNullOrWhiteSpace($startupDir)) {
        $startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
    }
    New-Item -ItemType Directory -Force -Path $startupDir | Out-Null
    $startupCmd = Join-Path $startupDir "PocketLM-PLM13-Watchdog.cmd"
    Set-Content -LiteralPath $startupCmd -Encoding ASCII -Value "@echo off`r`nstart `"PocketLM PLM13 Watchdog`" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`"`r`n"
}

Write-Output "Watchdog startup mode: $registeredMode"
if ($registeredMode -eq "ScheduledTask-OnLogon") {
    Write-Output "Watchdog task registered: $TaskName"
    Write-Output "Query: schtasks /Query /TN $TaskName /FO LIST /V"
    Write-Output "Run now: schtasks /Run /TN $TaskName"
    Write-Output "Disable startup: schtasks /Change /TN $TaskName /DISABLE"
    Write-Output "Enable startup: schtasks /Change /TN $TaskName /ENABLE"
} else {
    Write-Output "Startup shortcut: $startupCmd"
    Write-Output "Query: Get-Process powershell | Where-Object { `$_.Path -like '*powershell*' }"
    Write-Output "Disable startup: Remove-Item `"$startupCmd`""
    Write-Output "Enable startup: rerun tools\install_auto_resume_plm13_watchdog.ps1"
}
Write-Output "User: $($principal.Name)"
Write-Output "Script: $scriptPath"
Write-Output "Stop: powershell -NoProfile -ExecutionPolicy Bypass -File `"$repoPath\tools\stop_auto_resume_plm13_watchdog.ps1`""

if ($StartNow) {
    if ($registeredMode -eq "ScheduledTask-OnLogon") {
        & schtasks.exe /Run /TN $TaskName | Write-Output
        if ($LASTEXITCODE -ne 0) {
            throw "schtasks.exe /Run failed with exit code $LASTEXITCODE"
        }
    } else {
        Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", $scriptPath) -WindowStyle Hidden
        Write-Output "Watchdog process started directly."
    }
}
