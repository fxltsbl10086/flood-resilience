param(
    [Parameter(Mandatory = $true)][ValidateSet('Run', 'Collect')][string]$Action,
    [string]$ModelRoot,
    [string]$ResultRoot,
    [string]$ScenarioName,
    [string]$ResultDirectory,
    [string]$Image = 'deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release',
    [ValidateRange(1, 64)][int]$Threads = 8
)

# Receive separate arguments from Python and use UTF-8 for paths and logs.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
if ($Action -eq 'Run') {
    if (-not $ModelRoot -or -not $ResultRoot -or -not $ScenarioName) {
        throw ""
    }
    & (Join-Path $PSScriptRoot 'run_sfincs.ps1') -ModelRoot $ModelRoot -ResultRoot $ResultRoot -ScenarioName $ScenarioName -Image $Image -Threads $Threads
} else {
    if (-not $ResultDirectory) { throw "" }
    & (Join-Path $PSScriptRoot 'collect_sfincs_run.ps1') -ResultDirectory $ResultDirectory
}
