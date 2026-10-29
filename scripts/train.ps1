param([string]$Config = "configs/edsr_x2.yaml", [string]$Resume = "", [switch]$Background)
$ErrorActionPreference = "Stop"
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $taskPython)) { throw 'Run scripts/setup.ps1 first' }
$env:PYTHONUNBUFFERED = '1'
$env:TORCH_HOME = Join-Path $taskRoot 'datasets\pretrained'
$taskArgs = @('-m','superres.train','--config',$Config)
if ($Resume) { $taskArgs += @('--resume',$Resume) }
if ($Background) {
    New-Item -ItemType Directory -Force -Path 'runs' | Out-Null
    $taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    # Paths with spaces need explicit quoting in Start-Process argument strings.
    $taskQuotedArgs = $taskArgs | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskQuotedArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "runs\console-$taskStamp.log" -RedirectStandardError "runs\errors-$taskStamp.log"
    Write-Output "Started PID $($taskProcess.Id). Logs: runs\console-$taskStamp.log"
} else {
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Training exited with an error; inspect the log and resume the last completed epoch' }
}

