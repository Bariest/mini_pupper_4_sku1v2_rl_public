<#
.SYNOPSIS
    Before the long run (20-30 min): every stage on a few robots for a few iterations, end to end, then the four
    .mpxp exports. Nothing is kept (runs\check folders are deleted).

.DESCRIPTION
     1. the GPU and the environment
     2. walk: the SKU1 v2 model at 405 g, the firmware trot as the teacher, a small clone and 3 PPO iterations
     3. stairs: 2 iterations of step 1 (rough ground, 10-step history) from SKU2's walk
     4. walker: the shipped walker through the recovery runner's checks; the walk -> walker conversion
     5. get-up: one of SKU2's motions replayed on SKU1 v2; the whole recipe at 2 iterations per step; the get-up test
        and the walk / fall / get-up sequence on a few robots
     6. the .mpxp exports of the shipped policies, compared with the files on the robot (results\mpxp)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File check.ps1
#>
. "$PSScriptRoot\scripts\common.ps1"
Initialize-Environment
$check = "runs\check"
foreach ($d in "$Walk\$check", "$Recovery\$check") { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
New-Item -ItemType Directory -Force "$Walk\$check", "$Recovery\$check" | Out-Null

Write-Stage "1. the GPU and the environment"
Invoke-Py $Root @("-c", "import torch, mjlab, bam, mp4_recovery; assert torch.cuda.is_available(), 'no CUDA GPU'; print('GPU:', torch.cuda.get_device_name(0))") "the GPU"

Write-Stage "2. walk"
Invoke-Py $Walk @("tools\sku1v2_check.py", "--masses", $Masses) "the SKU1 v2 model (walk)"
Invoke-Py $Walk @("tools\sku1v2_walk_check.py") "the walk's teacher, clone and PPO"

Write-Stage "3. stairs (step 1, 2 iterations)"
Invoke-Py $Walk @("tools\patch_mujoco_warp.py", "--check") "the MuJoCo Warp patch (setup.ps1)"
Invoke-Py $Walk @("scripts\train_terrain.py", "--variant", "C", "--terrain", "rough", "--history", "10", "--envs", "64",
                  "--iters", "2", "--seed", "1", "--tag", "check", "--out", $check, "--masses", $Masses,
                  "--init", "policies\bam_t5ub06_stops_v1") "rough-ground training"

Write-Stage "4. walker"
Invoke-Py $Recovery @("tools\sku1v2_robot_check.py") "the SKU1 v2 model (recovery)"
Invoke-Py $Recovery @("-m", "mp4_recovery.deploy.runner", "check", "--policy", "policies\sku1v2_walker") "runner check of the walker"
Invoke-Py $Root @("tools\walk_to_walker.py", "walk\policies\bam_t5ub06_stops_v1", "recovery\$check\walker_base") "walk -> walker base"
Invoke-Py $Recovery @("-m", "mp4_recovery.export_policy", "--from-policy", "$check\walker_base", "$check\walker",
                      "--limits", "--stand-offset-abd", "12") "walker base -> walker"
Invoke-Py $Recovery @("-m", "mp4_recovery.deploy.runner", "check", "--policy", "$check\walker") "runner check of the converted walker"

Write-Stage "5. get-up (2 iterations per step, 256 robots)"
Invoke-Py $Recovery @("-m", "mp4_recovery.gpu.getup_search", "--limits", "--widen-limits-deg", "60", "--replay",
     "data\searched_motions\search_left.json", "--start", "left", "--robots", "16", "--seed", "12",
     "--json", "$check\replay_left.json") "SKU2's left get-up replayed on SKU1 v2"
Invoke-Script "$Recovery\scripts\recipes\getup.ps1" @{ Seed = 1; Out = "$check\getup"; Envs = 256; Iters = 2
    CloneEpochs = 2; CloneRobots = 32; CloneRounds = 1; Motions = "data/searched_motions"; Python = $PY }
$last = Get-LatestRun "$Recovery\$check\getup" "*_smooth4"
if (-not $last) { throw "the recipe left no smooth4 run" }
$ck = Get-LastCheckpoint $last.FullName
Invoke-Py $Recovery @("-m", "mp4_recovery.export_policy", $ck.FullName, "$check\getup_export") "export of the get-up"
Invoke-Py $Recovery @("-m", "mp4_recovery.deploy.runner", "check", "--policy", "$check\getup_export") "runner check of the get-up"
Invoke-Py $Recovery @("-m", "mp4_recovery.eval.getup_eval", "--policy", "$check\getup_export", "--seed", "12",
                      "--robots", "8", "--seconds", "2", "--starts", "left,back", "--json", "$check\getup_eval.json") "getup_eval"
Invoke-Py $Recovery @("-m", "mp4_recovery.eval.sequence", "--walker", "policies\sku1v2_walker", "--getup", "$check\getup_export",
                      "--robots", "4", "--seconds", "6", "--seed", "12", "--json", "$check\sequence.json") "the sequence"

Write-Stage "6. the .mpxp exports, against the files on the robot"
$wc = "walk\mp3_walk_policy\data\sku2_contract.json"; $wm = "walk\mp3_walk_policy\data\imu_mount.json"
$rc = "recovery\src\mp4_recovery\data\sku1v2_contract.json"; $rm = "recovery\src\mp4_recovery\data\imu_mount.json"
Invoke-Py $Root @("tools\export_mpxp.py", "results\policies\sku1v2_walker", "--contract", $rc, "--mount", $rm,
                  "--robot", "sku1", "--name", "sku1v2_recovery_walker", "--out", "recovery\$check\walker.mpxp") "the walker's .mpxp"
Invoke-Py $Root @("tools\export_mpxp.py", "results\policies\sku1v2_getup_s41", "--contract", $rc, "--mount", $rm,
                  "--robot", "sku1", "--name", "sku1v2_recovery_getup_s41", "--out", "recovery\$check\getup.mpxp") "the get-up's .mpxp"
Invoke-Py $Root @("tools\export_mpxp.py", "recovery\$check\getup_export", "--contract", $rc, "--mount", $rm,
                  "--robot", "sku1", "--name", "check_getup", "--out", "recovery\$check\getup_check.mpxp") "the check get-up's .mpxp"
Invoke-Py $Root @("tools\export_mpxp.py", "walk\policies\bam_t5ub06_stairs_narrow_v3", "--contract", $wc, "--mount", $wm,
                  "--robot", "sku2", "--name", "check_stairs", "--out", "walk\$check\stairs.mpxp") "a stairs .mpxp"
foreach ($p in @(@("recovery\$check\walker.mpxp", "results\mpxp\sku1v2_recovery.mpxp"),
                 @("recovery\$check\getup.mpxp", "results\mpxp\getup_sku1.mpxp"))) {
    $a = (Get-FileHash "$Root\$($p[0])").Hash; $b = (Get-FileHash "$Root\$($p[1])").Hash
    if ($a -ne $b) { throw "$($p[0]) differs from $($p[1]): the export does not reproduce the robot's file" }
    Write-Host "$($p[0]) = $($p[1]) (the robot's file, byte for byte)"
}

foreach ($d in "$Walk\$check", "$Recovery\$check") { Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue }
Write-Host ""
Write-Host "check OK. The long run:  powershell -ExecutionPolicy Bypass -File train.ps1" -ForegroundColor Green
