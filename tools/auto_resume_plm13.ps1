param(
    [switch]$DryRun,
    [string]$RepoRoot = "C:\Users\isale\Documents\pcketlm",
    [string]$CodexCommand = "C:\Users\isale\AppData\Roaming\npm\codex.cmd",
    [string]$IssueId = "PLM-13",
    [string]$StateOverride = "",
    [string]$LabelOverride = ""
)

$ErrorActionPreference = "Stop"

function Write-LogLine {
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

function Test-ShouldStopForIssue {
    param([hashtable]$Issue)

    $stateNameRaw = ""
    if ($Issue.ContainsKey("StateName") -and $null -ne $Issue.StateName) {
        $stateNameRaw = [string]$Issue.StateName
    }
    $stateTypeRaw = ""
    if ($Issue.ContainsKey("StateType") -and $null -ne $Issue.StateType) {
        $stateTypeRaw = [string]$Issue.StateType
    }
    $stateName = $stateNameRaw.ToLowerInvariant()
    $stateType = $stateTypeRaw.ToLowerInvariant()
    $labels = @($Issue.Labels | ForEach-Object { $_.ToLowerInvariant() })
    if ($stateType -in @("completed", "canceled") -or $stateName -in @("done", "canceled")) {
        return "issue state is $($Issue.StateName)/$($Issue.StateType)"
    }
    if ($labels -contains "blocked") {
        return "issue has blocked label"
    }
    return ""
}

function Test-AutoResumeCodexProcessRunning {
    $marker = "Resume PLM-13 autonomously"
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'codex.cmd' OR Name = 'codex.ps1' OR Name = 'node.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($marker) }
    return @($processes).Count -gt 0
}

$repoPath = (Resolve-Path -LiteralPath $RepoRoot).Path
$logDir = Join-Path $repoPath "state\auto_resume"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$script:LogPath = Join-Path $logDir ("plm13-auto-resume-" + (Get-Date).ToString("yyyyMMdd-HHmmss") + ".log")
$lockPath = Join-Path $logDir "plm13-auto-resume.lock"
$lockStream = $null

try {
    $lockStream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $writer = New-Object System.IO.StreamWriter($lockStream)
    $writer.WriteLine("pid=$PID")
    $writer.WriteLine("started=$((Get-Date).ToString('o'))")
    $writer.Flush()
    $lockStream.Position = 0
} catch [System.IO.IOException] {
    Write-Output "Another PLM-13 auto-resume run is already holding the lock."
    exit 0
}

try {
    Write-LogLine "PLM-13 auto-resume starting. Repo=$repoPath DryRun=$DryRun"

    $issue = Get-LinearIssueState -Identifier $IssueId
    Write-LogLine "Linear safety check: source=$($issue.Source) state=$($issue.StateName)/$($issue.StateType) labels=$($issue.Labels -join ',')"
    $stopReason = Test-ShouldStopForIssue -Issue $issue
    if ($stopReason) {
        Write-LogLine "Stop: $stopReason. Codex will not be launched."
        exit 0
    }

    if (Test-AutoResumeCodexProcessRunning) {
        Write-LogLine "Stop: an auto-resume Codex process is already running."
        exit 0
    }

    if (-not (Test-Path -LiteralPath $CodexCommand)) {
        $resolvedCodex = Get-Command codex.cmd -ErrorAction SilentlyContinue
        if ($resolvedCodex) {
            $CodexCommand = $resolvedCodex.Source
        } else {
            throw "Codex CLI was not found at $CodexCommand and codex.cmd is not on PATH."
        }
    }

    $prompt = @"
Resume PLM-13 autonomously per the ticket's hard autonomy rule. Continue from where the last session left off.

Before doing work, read PLM-13 in Linear and the pcketlm repo status. If PLM-13 is Done, Canceled, or has label blocked, exit cleanly. Otherwise continue the DeepSeek GPU paging/residency engine work from branch plm-13-gpu-effective-speed. Keep evidence in LOG.md/DECISIONS.md/STATUS.md/TODO.md, run relevant tests, push commits, and post Linear comments with real numbers.
"@

    $codexArgs = @(
        "-a", "never",
        "exec",
        "--cd", $repoPath,
        "--sandbox", "danger-full-access",
        $prompt
    )
    Write-LogLine "Codex command: `"$CodexCommand`" $($codexArgs[0..6] -join ' ') <resume-prompt>"
    if ($DryRun) {
        Write-LogLine "Dry run: would launch Codex now."
        exit 0
    }

    $codexLogPath = Join-Path $logDir ("plm13-codex-" + (Get-Date).ToString("yyyyMMdd-HHmmss") + ".log")
    Write-LogLine "Launching Codex. Output=$codexLogPath"
    Push-Location $repoPath
    try {
        & $CodexCommand @codexArgs > $codexLogPath 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Write-LogLine "Codex exited with code $exitCode."
    exit $exitCode
} catch {
    Write-LogLine "ERROR: $($_.Exception.Message)"
    exit 1
} finally {
    if ($lockStream) {
        $lockStream.Dispose()
    }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
