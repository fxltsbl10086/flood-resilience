param(
    [Parameter(Mandatory = $true)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][string]$ResultRoot,
    [Parameter(Mandatory = $true)][string]$ScenarioName,
    [string]$Image = 'deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release',
    [ValidateRange(1, 64)][int]$Threads = 8,
    # Zero keeps the Docker default; a positive value limits CPU quota without changing OpenMP or physics.
    [ValidateScript({ ($_ -eq 0) -or ($_ -ge 1 -and $_ -le 64) })][double]$CpuLimit = 0
)

# Use the installed image, mount inputs read-only and write to the separate result directory.
# Keep rainfall, drainage and boundary settings unchanged.
$ErrorActionPreference = 'Stop'
$cityRoot = (Resolve-Path -LiteralPath $ModelRoot).Path
$sourceScenarioRoot = Join-Path $cityRoot $ScenarioName
$resultCityRoot = [IO.Path]::GetFullPath($ResultRoot)
$scenarioRoot = Join-Path $resultCityRoot $ScenarioName
$runRoot = $scenarioRoot
$inputPath = Join-Path $sourceScenarioRoot 'sfincs.inp'
if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
    throw ""
}
if (Test-Path -LiteralPath (Join-Path $scenarioRoot 'sfincs_map.nc')) {
    throw ""
}

# Check referenced solver inputs and record their hashes.
$settings = @{}
foreach ($line in Get-Content -LiteralPath $inputPath -Encoding UTF8) {
    if ($line -match '^\s*(\w+)\s*=\s*(.+?)\s*$') {
        $settings[$Matches[1]] = $Matches[2]
    }
}
$inputFiles = @($inputPath)
$runtimeReferences = @{}
if ($settings['netamprfile'] -ne 'sfincs_cn_runoff.nc') {
    throw ""
}
foreach ($key in 'depfile', 'mskfile', 'indexfile', 'manningfile', 'qinffile', 'netamprfile') {
    if (-not $settings.ContainsKey($key)) { throw "" }
    $path = [IO.Path]::GetFullPath((Join-Path $sourceScenarioRoot $settings[$key]))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "" }
    if (-not $path.StartsWith($cityRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw ""
    }
    $relative = $path.Substring($cityRoot.Length).TrimStart('\').Replace('\', '/')
    $runtimeReferences[$key] = '/inputs/' + $relative
    $inputFiles += $path
}

New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$statusPath = Join-Path $runRoot 'run_status.json'
if (Test-Path -LiteralPath $statusPath) {
    $previous = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($previous.status -in @('submitting', 'submitted', 'running', 'completed', 'completed_with_quality_notes')) {
        throw ""
    }
}
$imageInfoText = & docker image inspect $Image
if ($LASTEXITCODE -ne 0) { throw "" }
$imageInfo = @($imageInfoText | ConvertFrom-Json)[0]
# Change only runtime file references; preserve the input template and physical settings.
$runtimeInputPath = Join-Path $runRoot 'sfincs.inp'
$runtimeLines = foreach ($line in Get-Content -LiteralPath $inputPath -Encoding UTF8) {
    if ($line -match '^\s*(\w+)\s*=') {
        $key = $Matches[1]
        if ($runtimeReferences.ContainsKey($key)) {
            ('{0,-21}= {1}' -f $key, $runtimeReferences[$key])
        } else { $line }
    } else { $line }
}
[IO.File]::WriteAllText($runtimeInputPath, ($runtimeLines -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
$inputFiles += $runtimeInputPath
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$containerName = ('sfincs-' + (Split-Path $cityRoot -Leaf) + '-' + $ScenarioName + '-' + $stamp).ToLowerInvariant()
# Use ASCII container names while retaining mounted directory names.
$containerName = [Regex]::Replace($containerName, '[^a-z0-9_.-]', '-')
$record = [ordered]@{
    status = 'submitting'
    scenario = $ScenarioName
    submitted_utc = (Get-Date).ToUniversalTime().ToString('o')
    image = $Image
    image_id = $imageInfo.Id
    image_repo_digests = $imageInfo.RepoDigests
    container_name = $containerName
    threads = $Threads
    cpu_limit = $CpuLimit
    cpu_wait_policy = $(if ($CpuLimit -gt 0) { 'PASSIVE' } else { $null })
    simulation_start = $settings['tstart']
    simulation_stop = $settings['tstop']
    scenario_directory = $scenarioRoot
    source_scenario_directory = $sourceScenarioRoot
    input_model_directory = $cityRoot
    runtime_configuration = $runtimeInputPath
    run_record_directory = $runRoot
    input_files = @($inputFiles | ForEach-Object {
        [ordered]@{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    solver_executed = $false
}
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $statusPath -Encoding utf8
$dockerArguments = @(
    'run', '--detach', '--pull', 'never', '--name', $containerName,
    '--mount', "type=bind,source=$cityRoot,target=/inputs,readonly",
    '--mount', "type=bind,source=$resultCityRoot,target=/data",
    '--workdir', "/data/$ScenarioName",
    '--env', "OMP_NUM_THREADS=$Threads"
)
if ($CpuLimit -gt 0) {
    # Use invariant number formatting and avoid busy-waiting under CPU limits.
    $cpuText = $CpuLimit.ToString([Globalization.CultureInfo]::InvariantCulture)
    $dockerArguments += @('--cpus', $cpuText, '--env', 'OMP_WAIT_POLICY=PASSIVE')
}
$dockerArguments += @($Image, 'sfincs')
$containerId = & docker @dockerArguments
if ($LASTEXITCODE -ne 0) {
    $record['status'] = 'submission_failed'
    $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $statusPath -Encoding utf8
    throw ""
}
$record['status'] = 'submitted'
$record['container_id'] = ($containerId | Out-String).Trim()
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $statusPath -Encoding utf8
Write-Output "Started: $containerName"
Write-Output "Run record: $statusPath"

