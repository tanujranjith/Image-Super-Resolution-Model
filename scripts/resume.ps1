param([string]$Checkpoint = "runs/edsr-x2-seed42/last.pt", [string]$Config = "configs/edsr_x2.yaml", [switch]$Background)
& "$PSScriptRoot\train.ps1" -Config $Config -Resume $Checkpoint -Background:$Background

