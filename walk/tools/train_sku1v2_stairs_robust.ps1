# SKU1 v2 stairs, step 7: the step 6 policy (model_4897) trained 1000 more iterations on the same narrow flights
# (risers 25-45 mm), with
#   - the feet lifted higher: the swing apex wanted 32 mm over the step under and ahead of the foot (steps 1-6: 25 mm);
#   - robots unlike the model, drawn per robot at the start: each shin (knee to foot) +-2 mm, each foot's radius
#     x0.7-1.4 (3.4-6.9 mm; the model's is 4.9 mm), the servos' supply voltage x0.9-1.1 (weaker / stronger servos).
# The robot: the chain's own (data/sku1v2_masses.json, 390 g, 195 / 195) unless -Masses names another file (the 405 g
# chain: sku1v2_masses_405g.json). Then the export (model_5896) and the tests on that robot, for 4897 and the new
# policy: the printed staircases, and the 4-step one on robots unlike the model. Windows / PowerShell. From the
# repository's root:
#
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs_robust.ps1               # train, export, test
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs_robust.ps1 -EvalOnly     # export + test again
#   ... -Masses sku1v2_masses_405g.json -Out runs\sku1v2_stairs_405 -Old policies\sku1v2_405_stairs_narrow_4897 -Policy policies\sku1v2_405_stairs_robust_5896 -Tag w405_
#                     # step 7 on the chain trained on the weighed robot (train_sku1v2_stairs.ps1 -Masses ... -Tag w405_)
#
# Needs step 6's run (train_sku1v2_stairs.ps1) in runs\sku1v2_stairs. Run again after a stop: a finished step 7 is
# skipped, an interrupted one renamed *_interrupted and started over; tests already written are not run again.
param(
    [int]$Envs = 1024,
    [string]$Out = "runs\sku1v2_stairs",
    [string]$Policy = "policies\sku1v2_stairs_robust_5896",
    [int]$EvalEnvs = 256,
    [string]$Masses = "",
    [string]$Old = "policies\sku1v2_stairs_narrow_4897",    # step 6's export: the policy step 7 is compared with
    [string]$Tag = "s7_",                                   # the test files: runs\eval\sku1v2_<Tag>...
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
if (-not $s6) { throw "no step 6 run (*_s6_stairs_narrow_r45) in ${Out}: train_sku1v2_stairs.ps1 first" }
$start = Join-Path $s6.FullName "model_4897.pt"
if (-not (Test-Path $start)) { throw "missing $start" }

$massArgs = @()
if ($Masses) {
    if (-not (Test-Path "mp3_walk_policy\data\$Masses")) { throw "no mass file mp3_walk_policy\data\$Masses" }
    $massArgs = @("--masses", $Masses)
}
$robot = @("--leg-length-mm", "2", "--foot-radius-scale", "0.7", "1.4", "--servo-strength", "0.9", "1.1")
$last = "model_5896.pt"

if (-not $EvalOnly) {
    $r = RunOf 7 "stairs_robust"
    if ($r -and (Test-Path (Join-Path $r.FullName $last))) {
        Write-Host "step 7: $($r.Name) done, skipped"
    } else {
        if ($r) {
            Write-Warning "step 7: $($r.Name) has no $last (interrupted): renamed $($r.Name)_interrupted, starting over"
            Rename-Item $r.FullName "$($r.Name)_interrupted"
        }
        Write-Host "step 7: higher feet + robots unlike the model, 1000 iterations from model_4897   $(Get-Date -Format 'HH:mm')"
        $a = @("--variant", "C", "--heading", "loop", "--foot-softness", "0.005", "0.02", "--friction", "0.4", "1.3",
               "--vx-max", "0.12", "--clearance", "apex", "--clearance-weight", "-2", "--stumble", "-0.5",
               "--progress", "2", "--envs", "$Envs", "--out", $Out,
               "--terrain", "stairs_narrow", "--stair-exit", "--tilt-limit", "1.0", "--heading-law", "pi",
               "--heading-wz", "0.4", "--max-init-level", "9", "--narrow-risers", "0.025", "0.045",
               "--clearance-target", "0.032") + $massArgs + $robot + $patch
        & $PY -B scripts\train_terrain.py @a --iters 1000 --resume $start --seed 7 --tag stairs_robust
        if ($LASTEXITCODE -ne 0) { throw "step 7 failed (out of GPU memory: -Envs 512)" }
    }
}

$run = RunOf 7 "stairs_robust"
if (-not $run) { throw "no step 7 run in $Out" }
$ck = Join-Path $run.FullName $last
if (-not (Test-Path $ck)) { throw "missing $ck" }
& $PY -B scripts\export_policy.py $ck $Policy
if ($LASTEXITCODE -ne 0) { throw "export of $ck failed" }

function Test-One([string]$pol, [string]$name, [string]$lay, [int]$seed, [string[]]$extra) {
    $json = "runs\eval\sku1v2_${name}_${lay}_seed$seed.json"
    if (Test-Path $json) { return }
    & $PY -B tools\terrain_eval.py --policy $pol --terrain stairs_built --layout "data\stairs\$lay.json" `
        --width 0.20 --landing 0.25 --envs $EvalEnvs --seconds 30 --heading-loop --stop-at-goal `
        --tilt-limit 1.0 --seed $seed --json $json @massArgs @extra
    if ($LASTEXITCODE -ne 0) { throw "the test of $pol failed" }
}
$old = $Old
# every test on the same robot (the chain's or -Masses), both policies, the same seeds (the same draws)
foreach ($seed in 1, 2) {
    foreach ($lay in @("printed_r25_t50_x4", "printed_r20_25_25")) {
        Test-One $Policy "${Tag}5896" $lay $seed @()
        if (Test-Path $old) { Test-One $old "${Tag}4897" $lay $seed @() }
    }
    # robots unlike the model
    Test-One $Policy "${Tag}5896_varied" "printed_r25_t50_x4" $seed $robot
    if (Test-Path $old) { Test-One $old "${Tag}4897_varied" "printed_r25_t50_x4" $seed $robot }
}

# the summary: success per test, both seeds
$rows = @(
    @("4897, 4 x 25 mm", "${Tag}4897_printed_r25_t50_x4"), @("5896, 4 x 25 mm", "${Tag}5896_printed_r25_t50_x4"),
    @("4897, 20/25/25 mm", "${Tag}4897_printed_r20_25_25"), @("5896, 20/25/25 mm", "${Tag}5896_printed_r20_25_25"),
    @("4897, 4 x 25 mm, varied robots", "${Tag}4897_varied_printed_r25_t50_x4"),
    @("5896, 4 x 25 mm, varied robots", "${Tag}5896_varied_printed_r25_t50_x4"))
Write-Host ("the robot: " + $(if ($Masses) { $Masses } else { "sku1v2_masses.json (390 g, 195 / 195)" }))
Write-Host ""
foreach ($row in $rows) {
    $v = @()
    foreach ($seed in 1, 2) {
        $f = "runs\eval\sku1v2_$($row[1])_seed$seed.json"
        if (Test-Path $f) {
            $c = (Get-Content $f -Raw | ConvertFrom-Json).cells[0]
            $v += "success {0,5:N1}% fall {1,5:N1}%" -f (100 * $c.success), (100 * $c.fall)
        }
    }
    Write-Host ("{0,-34} {1}" -f $row[0], ($v -join "   |   "))
}
Write-Host "done: $Policy (tests in runs\eval\sku1v2_*.json)"
