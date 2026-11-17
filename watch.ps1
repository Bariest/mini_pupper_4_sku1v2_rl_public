<#
.SYNOPSIS
    TensorBoard on a stage's training runs: http://localhost:6006 (Ctrl+C stops it; the training is not affected).

.DESCRIPTION
    What to look at (docs/PIPELINE.md, "Watching a run"):
      walk     Train/mean_reward rises; Episode_Termination/* (falls) drop after the clone's first iterations
      stairs   Curriculum/terrain_levels rises in every step; Episode_Termination/tilt and /body_contact fall
      getup    Episode_Reward/success climbs in `wide` and stays up in the steps after

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File watch.ps1 -Stage stairs
#>
param([ValidateSet("walk", "stairs", "getup")][string]$Stage = "getup", [int]$Port = 6006)
. "$PSScriptRoot\scripts\common.ps1"
Initialize-Environment
$dir = switch ($Stage) {
    "walk"   { Join-Path $Walk "runs\sku1v2_405_walk" }
    "stairs" { Join-Path $Walk "runs\sku1v2_405_stairs" }
    "getup"  { Join-Path $Recovery "runs\getup" }
}
if (-not (Test-Path $dir)) { throw "no runs yet in $dir" }
Write-Host "TensorBoard on $dir -> http://localhost:$Port"
& $PY -m tensorboard.main --logdir $dir --port $Port
