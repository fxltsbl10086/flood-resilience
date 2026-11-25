param(
    [Parameter(Mandatory = $true)][string]$ResultDirectory
)

# Collect container state and logs without starting the solver or deleting files.
# After completion, the shared exporter checks native output time, status and grid.
$ErrorActionPreference = 'Stop'
$resultPath = (Resolve-Path -LiteralPath $ResultDirectory).Path
$statusPath = Join-Path $resultPath 'run_status.json'
$record = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
$stateText = & docker inspect $record.container_name --format '{{json .State}}'
if ($LASTEXITCODE -ne 0) { throw "" }
$state = $stateText | ConvertFrom-Json
[IO.File]::WriteAllText((Join-Path $resultPath 'container_state.json'),
    ($state | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
$logText = (& docker logs $record.container_name 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) { throw "" }
[IO.File]::WriteAllText((Join-Path $resultPath 'solver_console.log'), $logText, [Text.UTF8Encoding]::new($false))
if ($state.Running) {
    $record.status = 'running'
} elseif ($state.ExitCode -ne 0 -or $state.OOMKilled) {
    $record.status = 'solver_failed'
    $record.solver_executed = $true
} elseif ($record.status -notin @('completed', 'completed_diagnostic', 'completed_with_quality_notes', 'completed_requires_input_repair')) {
    $record.status = 'solver_finished'
    $record.solver_executed = $true
}
$record | Add-Member -NotePropertyName last_collected_utc -NotePropertyValue ((Get-Date).ToUniversalTime().ToString('o')) -Force
[IO.File]::WriteAllText($statusPath, ($record | ConvertTo-Json -Depth 14), [Text.UTF8Encoding]::new($false))
$lastProgress = ($logText -split "[\r\n]+" | Where-Object { $_ -match '\d+% complete' } | Select-Object -Last 1)
[ordered]@{
    status = $record.status
    running = $state.Running
    exit_code = $state.ExitCode
    oom_killed = $state.OOMKilled
    last_progress = $lastProgress
    result_directory = $resultPath
} | ConvertTo-Json
