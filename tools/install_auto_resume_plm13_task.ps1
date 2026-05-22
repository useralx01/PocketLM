param(
    [string]$TaskName = "PocketLM-PLM13-AutoResume",
    [string]$RepoRoot = "C:\Users\isale\Documents\pcketlm",
    [int]$IntervalHours = 2
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path -LiteralPath $RepoRoot).Path
$scriptPath = Join-Path $repoPath "tools\auto_resume_plm13.ps1"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Auto-resume script not found: $scriptPath"
}

$logDir = Join-Path $repoPath "state\auto_resume"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$principal = [Security.Principal.WindowsIdentity]::GetCurrent()
$startBoundary = (Get-Date).AddMinutes(5).ToString("yyyy-MM-ddTHH:mm:ss")
$taskXmlPath = Join-Path $logDir "$TaskName.xml"
$escapedScript = [Security.SecurityElement]::Escape($scriptPath)
$escapedRepo = [Security.SecurityElement]::Escape($repoPath)
$escapedAuthor = [Security.SecurityElement]::Escape($principal.Name)
$escapedSid = [Security.SecurityElement]::Escape($principal.User.Value)
$interval = "PT${IntervalHours}H"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>$escapedAuthor</Author>
    <Description>Auto-resume Codex on PLM-13 until the Linear issue is Done, Canceled, or blocked.</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>$startBoundary</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
      <Repetition>
        <Interval>$interval</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$escapedSid</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NoProfile -ExecutionPolicy Bypass -File "$escapedScript"</Arguments>
      <WorkingDirectory>$escapedRepo</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

[System.IO.File]::WriteAllText($taskXmlPath, $xml, [System.Text.Encoding]::Unicode)

$registeredMode = "S4U"
& schtasks.exe /Create /TN $TaskName /XML $taskXmlPath /F | Write-Output
if ($LASTEXITCODE -ne 0) {
    Write-Warning "S4U task registration failed with exit code $LASTEXITCODE. Falling back to current-user interactive registration."
    $registeredMode = "Interactive"
    $action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
    & schtasks.exe /Create /TN $TaskName /SC HOURLY /MO $IntervalHours /TR $action /F | Write-Output
    if ($LASTEXITCODE -ne 0) {
        throw "fallback schtasks.exe failed with exit code $LASTEXITCODE"
    }
}

Write-Output "Task registered: $TaskName ($registeredMode)"
Write-Output "Task XML: $taskXmlPath"
Write-Output "Query: schtasks /Query /TN $TaskName /FO LIST /V"
Write-Output "Run once: schtasks /Run /TN $TaskName"
Write-Output "Disable: schtasks /Change /TN $TaskName /DISABLE"
Write-Output "Enable: schtasks /Change /TN $TaskName /ENABLE"
