# SKU1 v2 learns to walk: the recipe of policies/sku1_368_stops_s2 (SKU1 at 368 g, "seed 2", trained in omniRL) and of
# SKU2's policies/bam_t5ub06_stops_v1, on SKU1's body at 390 g, 195 / 195 g on the front and rear feet
# (MP3_ROBOT=sku1v2, data/sku1v2_masses.json). Omnidirectional: forward, back, sideways, turning in place, curves,
# and standing still on a zero command. Windows / PowerShell. From the repository's root:
#
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_walk.ps1                 # clone, seeds 1 and 2, eval, export
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_walk.ps1 -Seeds 3        # one more seed
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_walk.ps1 -EvalOnly       # eval + export again
#
# 1. Clone: the firmware trot (minipupperesp's own gait and IK) on take-like schedules with stops, 4096 robots,
#    action noise 0.06, 150 epochs (scripts/distill.py --omni --stops), with its gates (runs\sku1v2_walk\clone\gate.json).
# 2. PPO per seed: 1300 iterations at 512 robots, anchored to the clone (beta 20), std fixed at 0.1, 50 critic-only
#    iterations, lr 1e-4 (scripts/train.py --stops), exactly sku1_368_stops_s2's options.
# 3. The omni gates (scripts/evaluate_omni.py: eight directions, turns, curves, stops, randomised robots, the joystick)
#    and the export to policies\sku1v2_stops_s<seed>.
# Run again after a stop: finished parts are skipped; an interrupted seed's folder is renamed *_interrupted.
param(
    [int[]]$Seeds = @(1, 2),
    [int]$Envs = 512,
    [int]$Iters = 1300,
    [string]$Out = "runs\sku1v2_walk",
    [string]$Policy = "policies\sku1v2_stops",     # the exports: <Policy>_s<seed>
    [switch]$EvalOnly
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:MP3_ROBOT = "sku1v2"
if ($env:MP3_MASSES) { Write-Host "the robot: mp3_walk_policy\data\$($env:MP3_MASSES)" }
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "disabled" }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not $env:BAM_DIR) { $env:BAM_DIR = (Resolve-Path "..\bam").Path }
if (-not $env:MINIPUPPERESP_DIR) { $env:MINIPUPPERESP_DIR = (Resolve-Path "..\minipupperesp").Path }
# the training Python: SKU1V2_PY (set by the repository's top-level scripts), else a bam\.venv-mjlab of its own
$PY = if ($env:SKU1V2_PY) { $env:SKU1V2_PY } else { Join-Path $env:BAM_DIR ".venv-mjlab\Scripts\python.exe" }
if (-not (Test-Path $PY)) { throw "no training Python at $PY (run setup.ps1 at the repository root)" }
foreach ($f in "sim\model\pupper.xml", "main\stanford_gait.c", "main\stanford_kinematics.c") {
    if (-not (Test-Path "$env:MINIPUPPERESP_DIR\$f")) { throw "missing $env:MINIPUPPERESP_DIR\$f" }
}
$last = "model_$($Iters - 1).pt"

# 1. the clone
$clone = "$Out\clone\clone.pt"
if (Test-Path $clone) {
    Write-Host "clone: $clone done, skipped"
} elseif (-not $EvalOnly) {
    Write-Host "clone: the firmware trot with stops, 4096 robots   $(Get-Date -Format 'HH:mm')"
    & $PY -B scripts\distill.py --arm bam_t5ub06 --omni --stops --envs 4096 --epochs 150 --noise 0.06 --out "$Out\clone"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $clone)) { throw "the clone failed" }
}

function RunOf([int]$seed) {   # the newest run folder of this seed
    if (-not (Test-Path "$Out\ppo")) { return $null }
    Get-ChildItem "$Out\ppo" -Directory -Filter "*_stops_s${seed}_sku1v2" | Sort-Object Name | Select-Object -Last 1
}

# 2. PPO, one seed after the other
if (-not $EvalOnly) {
    foreach ($seed in $Seeds) {
        $r = RunOf $seed
        if ($r) {
            if (Test-Path (Join-Path $r.FullName $last)) { Write-Host "seed ${seed}: $($r.Name) done, skipped"; continue }
            Write-Warning "seed ${seed}: $($r.Name) has no $last (interrupted): renamed $($r.Name)_interrupted, starting over"
            Rename-Item $r.FullName "$($r.Name)_interrupted"
        }
        Write-Host "seed ${seed}: PPO $Iters iterations, $Envs robots   $(Get-Date -Format 'HH:mm')"
        & $PY -B scripts\train.py --arm bam_t5ub06 --stops --clone $clone --seed $seed --beta 20 --fixed-std `
            --critic-warmup 50 --lr0 1e-4 --iters $Iters --envs $Envs --out "$Out\ppo" --tag sku1v2
        if ($LASTEXITCODE -ne 0) { throw "seed $seed failed (out of GPU memory: -Envs 256)" }
    }
}

# 3. the gates and the export
$summary = @()
foreach ($seed in $Seeds) {
    $r = RunOf $seed
    if (-not $r) { Write-Warning "seed ${seed}: no run"; continue }
    $ck = Join-Path $r.FullName $last
    if (-not (Test-Path $ck)) { Write-Warning "seed ${seed}: no $last"; continue }
    $ev = Join-Path $r.FullName "eval_omni_$([IO.Path]::GetFileNameWithoutExtension($last)).json"
    if (-not (Test-Path $ev)) {
        Write-Host "seed ${seed}: the omni gates   $(Get-Date -Format 'HH:mm')"
        & $PY -B scripts\evaluate_omni.py --ckpt $ck
        if ($LASTEXITCODE -ne 0) { throw "the evaluation of $ck failed" }
    }
    $pol = "${Policy}_s$seed"
    & $PY -B scripts\export_policy.py $ck $pol
    if ($LASTEXITCODE -ne 0) { throw "export of $ck failed" }
    $res = Get-Content $ev -Raw | ConvertFrom-Json
    $summary += "seed ${seed}: ALL PASS $($res.pass)   joystick falls $([math]::Round($res.joystick.fall_frac, 3))   -> $pol"
}
Write-Host ""
$summary | ForEach-Object { Write-Host $_ }
Write-Host "the gates of each seed: $Out\ppo\<run>\eval_omni_*.json"
