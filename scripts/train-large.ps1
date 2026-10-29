param([switch]$Background, [string]$AfterCarStatus, [string]$AfterCarExperiment)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
$readyPath = Join-Path $project 'datasets\large\readiness.json'
if (-not (Test-Path -LiteralPath $readyPath)) { throw 'Large dataset validation has not completed.' }
$ready = Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json
if ($ready.status -ne 'ready_for_training_approval') { throw 'Large dataset is not ready.' }
$hashes = @{
    'datasets/large/processed/training.json' = $ready.training_manifest_sha256
    'datasets/large/processed/lsdir-heldout.json' = $ready.heldout_manifest_sha256
    'datasets/processed/div2k.json' = $ready.original_div2k_manifest_sha256
}
foreach ($field in @('config_hashes','validated_source_hashes')) {
    foreach ($entry in $ready.$field.PSObject.Properties) { $hashes[$entry.Name] = $entry.Value }
}
foreach ($relative in $hashes.Keys) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $project $relative))
    if (-not $target.StartsWith($project + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unsafe readiness path.'
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$relative]) {
        throw ('Validated file changed; revalidate before training: ' + $relative)
    }
}
# Manual launch after explicit approval only. A reminder never invokes this script.
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($project) -and $_.CommandLine -match 'overnight\.py|superres\.train'
}
if ($existing) { throw 'An SR training job or supervisor is already running.' }
$arguments = @('scripts/overnight.py','--configs','configs/edsr_large_x2.yaml','configs/edsr_large_x4.yaml')
if ([bool]$AfterCarStatus -ne [bool]$AfterCarExperiment) {
    throw 'Provide both car status and expected experiment directory to prevent a stale completion from releasing the queue.'
}
if ($AfterCarStatus) {
    $arguments += @('--after-car-status', [System.IO.Path]::GetFullPath($AfterCarStatus),
                    '--after-car-experiment', [System.IO.Path]::GetFullPath($AfterCarExperiment))
}
if ($Background) {
    $null = New-Item -ItemType Directory -Path (Join-Path $project 'runs') -Force
    $quoted = $arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $child = Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $project -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $project 'runs\large-supervisor.stdout.log') `
        -RedirectStandardError (Join-Path $project 'runs\large-supervisor.stderr.log') -PassThru
    Set-Content -LiteralPath (Join-Path $project 'runs\large-supervisor.pid') -Value $child.Id
    Write-Output ('Started explicitly requested large SR supervisor: ' + $child.Id)
} else {
    Push-Location -LiteralPath $project
    try { & $python @arguments; if ($LASTEXITCODE -ne 0) { throw 'SR supervisor stopped; inspect logs.' } }
    finally { Pop-Location }
}
