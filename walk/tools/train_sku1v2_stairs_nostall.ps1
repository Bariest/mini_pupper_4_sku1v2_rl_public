# SKU1 v2 stairs, step 8 ("no stall"): the step 6 policy (model_4897) trained 1500 more iterations so that it does
# not freeze halfway up. On the real robot (405 g) the step 6 policy climbed for 3 s, levelled its body and stood
# still with the forward command still on, until the run's end. Here:
#   - a stall is a failure, like a fall: commanded forward and the body not 2 cm further along its heading in 3 s
#     ends the episode (the curriculum demotes it);
#   - every failure (fall or stall) costs -5 (--terminated -250), so freezing is never safer than climbing;
#   - more robots on the narrow flights (35 % instead of 20 %), with treads 45-55 mm (the printed stairs: 50 mm);
#   - the base's centre of mass drawn -5 .. +12 mm along x around the mass file's (was +-5 mm): the real robot did not
#     tip back where the model does, so it is likely a little less rear-heavy than weighed.
# The rest as step 6 (risers 25-45 mm, the heading loop, the 405 g robot unless -Masses ""). Then the export
# (model_6396) and the tests on the printed staircases, next to step 6's export. Windows / PowerShell. From the
# repository's root:
#
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs_nostall.ps1               # train, export, test
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs_nostall.ps1 -EvalOnly     # export + test again
#   ... -Out runs\sku1v2_405_stairs -Old policies\sku1v2_405_stairs_narrow_4897 -Policy policies\sku1v2_405full_stairs_nostall -Tag nsfull_
#                     # on the chain of the full 405 g bundle (walk, then stairs)
#
# Run again after a stop: a finished step 8 is skipped, an interrupted one renamed *_interrupted and started over;
# tests already written are not run again.
param(
    [int]$Envs = 1024,
    [int]$Iters = 1500,
    [string]$Out = "runs\sku1v2_stairs_405",
    [string]$Policy = "policies\sku1v2_405_stairs_nostall",
    [int]$EvalEnvs = 256,
    [string]$Masses = "sku1v2_masses_405g.json",
    [string]$Old = "policies\sku1v2_405_stairs_narrow_4897",   # step 6's export: the policy step 8 is compared with
    [string]$Tag = "ns405_",                                   # the test files: runs\eval\sku1v2_<Tag>...
    [switch]$EvalOnly
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:MP3_ROBOT = "sku1v2"
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "disabled" }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not $env:BAM_DIR) { $env:BAM_DIR = (Resolve-Path "..\bam").Path }
if (-not $env:MINIPUPPERESP_DIR) { $env:MINIPUPPERESP_DIR = (Resolve-Path "..\minipupperesp").Path }
# the training Python: SKU1V2_PY (set by the repository's top-level scripts), else a bam\.venv-mjlab of its own
$PY = if ($env:SKU1V2_PY) { $env:SKU1V2_PY } else { Join-Path $env:BAM_DIR ".venv-mjlab\Scripts\python.exe" }
if (-not (Test-Path $PY)) { throw "no training Python at $PY (run setup.ps1 at the repository root)" }

& $PY -B tools\patch_mujoco_warp.py
$patch = @()
if ($LASTEXITCODE -ne 0) {
    Write-Warning "mujoco_warp could not be patched (another version?): training without the patch"
    $patch = @("--allow-unpatched")
}

function RunOf([int]$seed, [string]$tag) {
    if (-not (Test-Path $Out)) { return $null }
    Get-ChildItem $Out -Directory -Filter "*_s${seed}_${tag}" | Sort-Object Name | Select-Object -Last 1
}

$s6 = RunOf 6 "stairs_narrow_r45"
if (-not $s6) { throw "no step 6 run (*_s6_stairs_narrow_r45) in ${Out}: train the stairs (steps 1-6) first" }
$start = Join-Path $s6.FullName "model_4897.pt"
if (-not (Test-Path $start)) { throw "missing $start" }

$massArgs = @()
if ($Masses) {
    if (-not (Test-Path "mp3_walk_policy\data\$Masses")) { throw "no mass file mp3_walk_policy\data\$Masses" }
    $massArgs = @("--masses", $Masses)
}
$last = "model_$(4897 + $Iters - 1).pt"

if (-not $EvalOnly) {
    $r = RunOf 8 "stairs_nostall"
    if ($r -and (Test-Path (Join-Path $r.FullName $last))) {
        Write-Host "step 8: $($r.Name) done, skipped"
    } else {
        if ($r) {
            Write-Warning "step 8: $($r.Name) has no $last (interrupted): renamed $($r.Name)_interrupted, starting over"
            Rename-Item $r.FullName "$($r.Name)_interrupted"
        }
        Write-Host "step 8: no stalls, $Iters iterations from model_4897   $(Get-Date -Format 'HH:mm')"
        $a = @("--variant", "C", "--heading", "loop", "--foot-softness", "0.005", "0.02", "--friction", "0.4", "1.3",
               "--vx-max", "0.12", "--clearance", "apex", "--clearance-weight", "-2", "--stumble", "-0.5",
               "--progress", "2", "--envs", "$Envs", "--out", $Out,
               "--terrain", "stairs_narrow", "--stair-exit", "--tilt-limit", "1.0", "--heading-law", "pi",
               "--heading-wz", "0.4", "--max-init-level", "9", "--narrow-risers", "0.025", "0.045",
               "--narrow-treads", "0.045", "0.055", "--narrow-share", "0.35",
               "--stall-s", "3", "--stall-m", "0.02", "--terminated", "-250",
               "--base-com-x", "-0.005", "0.012") + $massArgs + $patch
        & $PY -B scripts\train_terrain.py @a --iters $Iters --resume $start --seed 8 --tag stairs_nostall
        if ($LASTEXITCODE -ne 0) { throw "step 8 failed (out of GPU memory: -Envs 512)" }
    }
}

$run = RunOf 8 "stairs_nostall"
if (-not $run) { throw "no step 8 run in $Out" }
$ck = Join-Path $run.FullName $last
if (-not (Test-Path $ck)) { throw "missing $ck" }
& $PY -B scripts\export_policy.py $ck $Policy
if ($LASTEXITCODE -ne 0) { throw "export of $ck failed" }

function Test-One([string]$pol, [string]$name, [string]$lay, [int]$seed) {
    $json = "runs\eval\sku1v2_${name}_${lay}_seed$seed.json"
    if (Test-Path $json) { return }
    & $PY -B tools\terrain_eval.py --policy $pol --terrain stairs_built --layout "data\stairs\$lay.json" `
        --width 0.20 --landing 0.25 --envs $EvalEnvs --seconds 30 --heading-loop --stop-at-goal `
        --tilt-limit 1.0 --seed $seed --json $json @massArgs
    if ($LASTEXITCODE -ne 0) { throw "the test of $pol failed" }
}
# both policies on the same robot, the same seeds (the same draws)
foreach ($seed in 1, 2) {
    foreach ($lay in @("printed_r25_t50_x4", "printed_r20_25_25")) {
        Test-One $Policy "${Tag}new" $lay $seed
        if (Test-Path $Old) { Test-One $Old "${Tag}4897" $lay $seed }
    }
}

# the summary: success / fall / stall (neither: still on the stairs after 30 s) per test, both seeds
$rows = @(
    @("step 6 (4897), 4 x 25 mm", "${Tag}4897_printed_r25_t50_x4"), @("step 8, 4 x 25 mm", "${Tag}new_printed_r25_t50_x4"),
    @("step 6 (4897), 20/25/25 mm", "${Tag}4897_printed_r20_25_25"), @("step 8, 20/25/25 mm", "${Tag}new_printed_r20_25_25"))
Write-Host ("the robot: " + $(if ($Masses) { $Masses } else { "sku1v2_masses.json (390 g, 195 / 195)" }))
Write-Host ""
foreach ($row in $rows) {
    $v = @()
    foreach ($seed in 1, 2) {
        $f = "runs\eval\sku1v2_$($row[1])_seed$seed.json"
        if (Test-Path $f) {
            $c = (Get-Content $f -Raw | ConvertFrom-Json).cells[0]
            $v += "success {0,5:N1}% fall {1,5:N1}% stall {2,4:N1}%" -f (100 * $c.success), (100 * $c.fall), (100 * $c.stall)
        }
    }
    Write-Host ("{0,-28} {1}" -f $row[0], ($v -join "  |  "))
}
Write-Host "done: $Policy ($ck; tests in runs\eval\sku1v2_${Tag}*.json)"
