# SKU1 v2 climbs stairs: the SKU2 stair policy's training chain (scripts/recipes/terrain_stairs.sh, the six steps
# behind policies/bam_t5ub06_terrain_h10_v2 and policies/bam_t5ub06_stairs_narrow_v3) run on SKU1's body at 390 g,
# 195 / 195 g on the front and rear feet (MP3_ROBOT=sku1v2, data/sku1v2_masses.json). Windows / PowerShell. From the
# repository's root:
#
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs.ps1                # all six steps, then export + test
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs.ps1 -LastStep 1    # step 1 only (rough ground)
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs.ps1 -EvalOnly      # export + test again
#   ... -Masses sku1v2_masses_405g.json -Out runs\sku1v2_stairs_405 -Policy policies\sku1v2_405_stairs_narrow -Tag w405_
#                                       # the whole chain on another mass file (the robot as weighed), its own runs,
#                                       # exports and test files (runs\eval\sku1v2_<Tag>...)
#
# The same options, iterations, seeds, checkpoints and order as the SKU2 recipe; only the robot differs. Step 1 starts,
# as SKU2's did, from the flat omni policy policies/bam_t5ub06_stops_v1 (the same legs, servos and firmware trot; the
# anchor fades out, so the walk is re-learnt on this body). Steps 2-6 each fine-tune the previous step's checkpoint.
# Run again after a stop: finished steps are skipped; an interrupted step's folder is renamed *_interrupted and the step
# starts over. 1200 + 5 x 1000 iterations at 1024 robots: SKU2's took about 4 h on an RTX 5080 (2.4 s per iteration).
param(
    [int]$LastStep = 6,
    [int]$Envs = 1024,
    [string]$Out = "runs\sku1v2_stairs",
    [string]$Policy = "policies\sku1v2_stairs_narrow",
    [int]$EvalEnvs = 256,
    [string]$Masses = "",
    [string]$Tag = "",
    [string]$Init = "",          # step 1's warm start (a policies\<name> flat omni walk); default: SKU2's bam_t5ub06_stops_v1
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
if (-not (Test-Path "$env:MINIPUPPERESP_DIR\sim\model\pupper.xml")) { throw "no SKU1 model at $env:MINIPUPPERESP_DIR\sim\model\pupper.xml" }

# mujoco_warp's heightfield warning floods the console on stairs and rough ground: remove it (idempotent)
& $PY -B tools\patch_mujoco_warp.py
$patch = @()
if ($LASTEXITCODE -ne 0) {
    Write-Warning "mujoco_warp could not be patched (another version?): training without the patch"
    $patch = @("--allow-unpatched")
}

$common = @("--variant", "C", "--heading", "loop", "--foot-softness", "0.005", "0.02", "--friction", "0.4", "1.3",
            "--vx-max", "0.12", "--clearance", "apex", "--clearance-weight", "-2", "--stumble", "-0.5",
            "--progress", "2", "--envs", "$Envs", "--out", $Out) + $patch
$massArgs = @()
if ($Masses) {
    if (-not (Test-Path "mp3_walk_policy\data\$Masses")) { throw "no mass file mp3_walk_policy\data\$Masses" }
    $massArgs = @("--masses", $Masses)
    $common += $massArgs
    Write-Host "the robot: mp3_walk_policy\data\$Masses"
}
$narrow = @("--terrain", "stairs_narrow", "--stair-exit", "--tilt-limit", "1.0", "--heading-law", "pi",
            "--heading-wz", "0.4")

function RunOf([int]$seed, [string]$tag) {   # the newest run folder with this seed and tag
    if (-not (Test-Path $Out)) { return $null }
    Get-ChildItem $Out -Directory -Filter "*_s${seed}_${tag}" | Sort-Object Name | Select-Object -Last 1
}
function Ckpt([int]$seed, [string]$tag, [string]$file) {
    $r = RunOf $seed $tag
    if (-not $r) { throw "no run *_s${seed}_${tag} in $Out" }
    $p = Join-Path $r.FullName $file
    if (-not (Test-Path $p)) { throw "missing $p" }
    return $p
}
function Step([int]$n, [int]$seed, [string]$tag, [string]$last, [string]$resume, [string[]]$opts) {
    if ($n -gt $LastStep) { return }
    $r = RunOf $seed $tag
    if ($r) {
        if (Test-Path (Join-Path $r.FullName $last)) { Write-Host "step ${n}: $($r.Name) done, skipped"; return }
        Write-Warning "step ${n}: $($r.Name) has no $last (interrupted): renamed $($r.Name)_interrupted, starting over"
        Rename-Item $r.FullName "$($r.Name)_interrupted"
    }
    $a = $common + $opts
    if ($resume) { $a += @("--resume", $resume) }
    Write-Host "step ${n}: $tag   $(Get-Date -Format 'HH:mm')"
    & $PY -B scripts\train_terrain.py @a --seed $seed --tag $tag
    if ($LASTEXITCODE -ne 0) { throw "step $n failed (out of GPU memory: -Envs 512)" }
}

if (-not $EvalOnly) {
    $s1 = @("--terrain", "rough", "--history", "10", "--iters", "1200")
    if ($Init) {
        if (-not (Test-Path "$Init\policy.npz")) { throw "no policy $Init (-Init)" }
        $s1 += @("--init", $Init)
        Write-Host "step 1 starts from $Init"
    }
    Step 1 4 "rough" "model_1199.pt" "" $s1
    if ($LastStep -ge 2) {
        Step 2 5 "stairs" "model_1699.pt" (Ckpt 4 "rough" "model_700.pt") @("--terrain", "stairs", "--iters", "1000")
    }
    if ($LastStep -ge 3) {
        Step 3 5 "stairs_exit" "model_2499.pt" (Ckpt 5 "stairs" "model_1500.pt") `
            @("--terrain", "stairs", "--stair-exit", "--iters", "1000")
    }
    if ($LastStep -ge 4) {
        Step 4 5 "stairs_narrow" "model_2899.pt" (Ckpt 5 "stairs_exit" "model_1900.pt") ($narrow + @("--iters", "1000"))
    }
    if ($LastStep -ge 5) {
        Step 5 6 "stairs_narrow_r40" "model_3898.pt" (Ckpt 5 "stairs_narrow" "model_2899.pt") `
            ($narrow + @("--max-init-level", "9", "--narrow-risers", "0.020", "0.040", "--iters", "1000"))
    }
    if ($LastStep -ge 6) {
        Step 6 6 "stairs_narrow_r45" "model_4897.pt" (Ckpt 6 "stairs_narrow_r40" "model_3898.pt") `
            ($narrow + @("--max-init-level", "9", "--narrow-risers", "0.025", "0.045", "--iters", "1000"))
    }
}

function Test-Stairs([string]$pol, [string]$name) {
    # the README's test: the printed staircases, 2 seeds x $EvalEnvs robots, 30 s, heading loop, stop at the goal
    foreach ($lay in @("printed_r25_t50_x4", "printed_r20_25_25")) {
        foreach ($seed in 1, 2) {
            $json = "runs\eval\sku1v2_${Tag}${name}_${lay}_seed$seed.json"
            if ($Tag -and (Test-Path $json)) { continue }    # a tagged chain: tests already written are kept
            & $PY -B tools\terrain_eval.py --policy $pol --terrain stairs_built --layout "data\stairs\$lay.json" `
                --width 0.20 --landing 0.25 --envs $EvalEnvs --seconds 30 --heading-loop --stop-at-goal `
                --tilt-limit 1.0 --seed $seed --json $json @massArgs
            if ($LASTEXITCODE -ne 0) { throw "the test of $pol failed" }
        }
    }
}

if ($LastStep -ge 6) {
    # export as SKU2's was (step 6, iteration 4500) and the step's last checkpoint, then test both
    $run = RunOf 6 "stairs_narrow_r45"
    if (-not $run) { throw "no step 6 run in $Out" }
    foreach ($it in 4500, 4897) {
        $ck = Join-Path $run.FullName "model_$it.pt"
        if (-not (Test-Path $ck)) { Write-Warning "no $ck"; continue }
        $pol = "${Policy}_$it"
        & $PY -B scripts\export_policy.py $ck $pol
        if ($LASTEXITCODE -ne 0) { throw "export of $ck failed" }
        Test-Stairs $pol "stairs_$it"
    }
    # the reference: SKU2's shipped stair policy, untouched, on this robot
    Test-Stairs "policies\bam_t5ub06_stairs_narrow_v3" "sku2policy"
    # another mass file: the 390 g chain's policy on this robot too, if it is here
    if ($Masses -and (Test-Path "policies\sku1v2_stairs_narrow_4897") -and ($Policy -ne "policies\sku1v2_stairs_narrow")) {
        Test-Stairs "policies\sku1v2_stairs_narrow_4897" "390g_4897"
    }
    Write-Host "done: ${Policy}_4500 and ${Policy}_4897, tests in runs\eval\sku1v2_${Tag}*.json"
} elseif ($LastStep -ge 1 -and -not $EvalOnly) {
    Write-Host "done up to step $LastStep (runs in $Out)"
}
