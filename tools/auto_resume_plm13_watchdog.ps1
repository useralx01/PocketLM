param(
    [switch]$DryRun,
    [string]$RepoRoot = "C:\Users\isale\Documents\pcketlm",
    [string]$CodexCommand = "C:\Users\isale\AppData\Roaming\npm\codex.cmd",
    [string]$IssueId = "PLM-13",
    [string]$StateOverride = "",
    [string]$LabelOverride = "",
    [int]$MaxLoops = 0
)

$ErrorActionPreference = "Stop"

function Write-WatchdogLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffK")
    $line = "[$stamp] $Message"
    Write-Output $line
    Add-Content -LiteralPath $script:LogPath -Value $line
}

function Get-LinearIssueState {
    param([string]$Identifier)

    if ($StateOverride) {
        $labels = @()
        if ($LabelOverride) {
            $labels = $LabelOverride.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        }
        return @{
            StateName = $StateOverride
            StateType = $StateOverride.ToLowerInvariant()
            Labels = $labels
            Source = "override"
        }
    }

    $token = [Environment]::GetEnvironmentVariable("LINEAR_API_KEY", "Process")
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = [Environment]::GetEnvironmentVariable("LINEAR_API_KEY", "User")
    }
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "LINEAR_API_KEY is not available in Process or User environment."
    }

    $query = @"
query IssueStatus(`$id: String!) {
  issue(id: `$id) {
    identifier
    state { name type }
    labels { nodes { name } }
  }
}
"@
    $payload = @{ query = $query; variables = @{ id = $Identifier } } | ConvertTo-Json -Depth 6
    $headers = @{
        Authorization = $token
        "Content-Type" = "application/json"
    }
    $response = Invoke-RestMethod -Uri "https://api.linear.app/graphql" -Method Post -Headers $headers -Body $payload -TimeoutSec 30
    if ($response.errors) {
        throw "Linear API returned errors: $($response.errors | ConvertTo-Json -Compress)"
    }
    if (-not $response.data.issue) {
        throw "Linear issue $Identifier was not found."
    }
    return @{
        StateName = [string]$response.data.issue.state.name
        StateType = [string]$response.data.issue.state.type
        Labels = @($response.data.issue.labels.nodes | ForEach-Object { [string]$_.name })
        Source = "linear"
    }
}

function Get-StopReasonForIssue {
    param([hashtable]$Issue)

    $stateName = ([string]$Issue.StateName).ToLowerInvariant()
    $stateType = ([string]$Issue.StateType).ToLowerInvariant()
    $labels = @($Issue.Labels | ForEach-Object { ([string]$_).ToLowerInvariant() })
    if ($stateType -in @("completed", "canceled") -or $stateName -in @("done", "canceled")) {
        return "issue state is $($Issue.StateName)/$($Issue.StateType)"
    }
    if ($labels -contains "blocked") {
        return "issue has blocked label"
    }
    return ""
}

function Resolve-CodexCommand {
    param([string]$Command)

    if (Test-Path -LiteralPath $Command) {
        return (Resolve-Path -LiteralPath $Command).Path
    }
    $resolved = Get-Command codex.cmd -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Source
    }
    $resolved = Get-Command codex -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Source
    }
    throw "Codex CLI was not found at $Command and is not on PATH."
}

$repoPath = (Resolve-Path -LiteralPath $RepoRoot).Path
$logDir = Join-Path $repoPath "state\auto_resume"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$script:LogPath = Join-Path $logDir ("plm13-watchdog-" + (Get-Date).ToString("yyyyMMdd-HHmmss") + ".log")
$pidPath = Join-Path $logDir "plm13-watchdog.pid"
$stopPath = Join-Path $logDir "plm13-watchdog.stop"
$lockPath = Join-Path $logDir "plm13-watchdog.lock"
$lockStream = $null

if (Test-Path -LiteralPath $lockPath) {
    $lockText = Get-Content -Raw -LiteralPath $lockPath -ErrorAction SilentlyContinue
    $stalePid = 0
    if ($lockText -match "pid=(\d+)" -and [int]::TryParse($Matches[1], [ref]$stalePid)) {
        if (-not (Get-Process -Id $stalePid -ErrorAction SilentlyContinue)) {
            Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    $lockStream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $writer = New-Object System.IO.StreamWriter($lockStream)
    $writer.WriteLine("pid=$PID")
    $writer.WriteLine("started=$((Get-Date).ToString('o'))")
    $writer.Flush()
    $lockStream.Position = 0
} catch [System.IO.IOException] {
    Write-Output "Another PLM-13 watchdog is already running."
    exit 0
}

try {
    Set-Content -LiteralPath $pidPath -Value $PID
    Remove-Item -LiteralPath $stopPath -Force -ErrorAction SilentlyContinue
    $codexPath = Resolve-CodexCommand -Command $CodexCommand
    $loopIndex = 0
    Write-WatchdogLog "PLM-13 watchdog starting. Repo=$repoPath DryRun=$DryRun MaxLoops=$MaxLoops Codex=$codexPath"

    while ($true) {
        $loopIndex += 1
        if (Test-Path -LiteralPath $stopPath) {
            Write-WatchdogLog "Stop file exists: $stopPath"
            exit 0
        }

        $issue = Get-LinearIssueState -Identifier $IssueId
        Write-WatchdogLog "Loop $loopIndex Linear safety: source=$($issue.Source) state=$($issue.StateName)/$($issue.StateType) labels=$($issue.Labels -join ',')"
        $stopReason = Get-StopReasonForIssue -Issue $issue
        if ($stopReason) {
            Write-WatchdogLog "Stop: $stopReason. Codex will not be launched."
            exit 0
        }

        $prompt = @"
Resume PLM-13 autonomously per the ticket's hard autonomy rule. Continue from where the last session left off.

First read PLM-13 in Linear and the pcketlm repo status. If PLM-13 is Done, Canceled, or has label blocked, exit cleanly. Otherwise continue the DeepSeek GPU paging/residency work. Keep evidence in LOG.md/DECISIONS.md/STATUS.md/TODO.md, run relevant tests, push commits, and post Linear comments with real numbers. Do not wait for operator confirmation unless the machine or external service makes progress impossible.
"@
        $codexArgs = @(
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", $repoPath,
            $prompt
        )
        Write-WatchdogLog "Loop $loopIndex Codex command: `"$codexPath`" exec --dangerously-bypass-approvals-and-sandbox -C `"$repoPath`" <resume-prompt>"
        if ($DryRun) {
            Write-WatchdogLog "Dry run: would launch Codex now."
        } else {
            $codexLogPath = Join-Path $logDir ("plm13-watchdog-codex-" + (Get-Date).ToString("yyyyMMdd-HHmmss") + ".log")
            Write-WatchdogLog "Loop $loopIndex launching Codex. Output=$codexLogPath"
            Push-Location $repoPath
            try {
                & $codexPath @codexArgs > $codexLogPath 2>&1
                $exitCode = $LASTEXITCODE
            } finally {
                Pop-Location
            }
            Write-WatchdogLog "Loop $loopIndex Codex exited with code $exitCode."
        }

        if ($MaxLoops -gt 0 -and $loopIndex -ge $MaxLoops) {
            Write-WatchdogLog "MaxLoops reached: $MaxLoops"
            exit 0
        }
    }
} catch {
    Write-WatchdogLog "ERROR: $($_.Exception.Message)"
    exit 1
} finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    if ($lockStream) {
        $lockStream.Dispose()
    }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
