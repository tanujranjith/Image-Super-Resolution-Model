param([switch]$Extract)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path 'datasets\downloads' | Out-Null
$taskArchives = @('DIV2K_train_HR.zip','DIV2K_valid_HR.zip','DIV2K_train_LR_bicubic_X2.zip','DIV2K_train_LR_bicubic_X4.zip','DIV2K_valid_LR_bicubic_X2.zip','DIV2K_valid_LR_bicubic_X4.zip')
foreach ($taskArchive in $taskArchives) {
    $taskTarget = Join-Path 'datasets\downloads' $taskArchive
    if (-not (Test-Path -LiteralPath $taskTarget)) {
        Write-Output "Downloading $taskArchive"
        Invoke-WebRequest -Uri ('https://data.vision.ee.ethz.ch/cvl/DIV2K/' + $taskArchive) -OutFile ($taskTarget + '.part') -TimeoutSec 1800
        Move-Item -LiteralPath ($taskTarget + '.part') -Destination $taskTarget
    }
}
if ($Extract) {
    & '.\.venv\Scripts\python.exe' -m superres.acquire --extract
    if ($LASTEXITCODE -ne 0) { throw 'Archive verification/extraction failed' }
    & '.\.venv\Scripts\python.exe' -m superres.prepare
    if ($LASTEXITCODE -ne 0) { throw 'Pair validation failed' }
}
