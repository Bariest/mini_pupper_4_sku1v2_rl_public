# SKU1 v2 stairs v3: an existing stairs policy (a step 6 checkpoint, model_4897) trained on further on SKU1 v2 as
# weighed (405 g, 177 / 223 g), so that it does not stand still on the stairs:
#   - a stall is a failure like a fall: told to walk forward and the body not 2 cm further in 3 s ends the try; on the
#     narrow flights "further" is along the stairs (the heading it started with), so turning away does not count;
#   - turning more than 0.6 rad (34 deg) from the stairs' direction on a narrow flight is a failure too (a policy
#     that climbs diagonally walks off a 200 mm wide staircase);
#   - every failure (fall, stall, turned away) costs the same (-5);
#   - on stairs the body is no longer pulled level (only its roll is penalised): a level body on this short,
#     rear-heavy robot ends with straight rear legs, frozen;
#   - risers 15-30 mm (it could not get past the easiest level of 25-45 mm), treads 45-55 mm like the printed stairs,
#     35 % of the robots on narrow flights, the centre of mass drawn -5 .. +12 mm along x around the mass file's.
# Then the export (model_<4897 + Iters - 1>) and the tests on the printed staircases, next to the starting policy.
# Windows / PowerShell. From the repository's root:
#
#   powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs_v3.ps1 -From runs\init\A_405first\<run> -Name A_405first
#   ... -EvalOnly          export + tests again
#
# Run again after a stop: a finished run is skipped, an interrupted one renamed *_interrupted and started over; tests
# already written are not run again.
param(
    [Parameter(Mandatory = $true)][string]$From,     # a run folder with model_4897.pt and config.json
    [Parameter(Mandatory = $true)][string]$Name,     # A_405first, B_390, ...: the run tag, policy and test names
    [int]$Envs = 1024,
    [int]$Iters = 2000,
    [string]$Out = "runs\sku1v2_405_v3",
    [string]$Masses = "sku1v2_masses_405g.json",
    [int]$Seed = 9,
    [int]$EvalEnvs = 256,
    [switch]$EvalOnly,
    [switch]$NoTest                                  # train and export only (check.ps1)
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

$start = Join-Path $From "model_4897.pt"
if (-not (Test-Path $start)) { throw "missing $start" }
if (-not (Test-Path "mp3_walk_policy\data\$Masses")) { throw "no mass file mp3_walk_policy\data\$Masses" }
$massArgs = @("--masses", $Masses)
$tag = "v3_$Name"
$last = "model_$(4897 + $Iters - 1).pt"
$startPol = "policies\start_${Name}_4897"
$newPol = "policies\sku1v2_405_v3_${Name}"

function RunOf {
    if (-not (Test-Path $Out)) { return $null }
    Get-ChildItem $Out -Directory -Filter "*_s${Seed}_${tag}" | Sort-Object Name | Select-Object -Last 1
}

if (-not $EvalOnly) {
    $r = RunOf
    if ($r -and (Test-Path (Join-Path $r.FullName $last))) {
        Write-Host "${Name}: $($r.Name) done, skipped"
    } else {
        if ($r) {
            Write-Warning "${Name}: $($r.Name) has no $last (interrupted): renamed $($r.Name)_interrupted, starting over"
            Rename-Item $r.FullName "$($r.Name)_interrupted"
        }
        Write-Host "${Name}: $Iters iterations from $start   $(Get-Date -Format 'HH:mm')"
        $a = @("--variant", "C", "--heading", "loop", "--foot-softness", "0.005", "0.02", "--friction", "0.4", "1.3",
               "--vx-max", "0.12", "--clearance", "apex", "--clearance-weight", "-2", "--stumble", "-0.5",
               "--progress", "2", "--envs", "$Envs", "--out", $Out,
               "--terrain", "stairs_narrow", "--stair-exit", "--tilt-limit", "1.0", "--heading-law", "pi",
               "--heading-wz", "0.4", "--max-init-level", "9", "--narrow-risers", "0.015", "0.030",
               "--narrow-treads", "0.045", "0.055", "--narrow-share", "0.35",
               "--stall-s", "3", "--stall-m", "0.02", "--stall-stairs-axis", "--heading-fail", "0.6",
               "--upright-roll-only-on-stairs", "--terminated", "-250",
               "--base-com-x", "-0.005", "0.012") + $massArgs + $patch
        & $PY -B scripts\train_terrain.py @a --iters $Iters --resume $start --seed $Seed --tag $tag
        if ($LASTEXITCODE -ne 0) { throw "$Name failed (out of GPU memory: -Envs 512)" }
    }
}

$run = RunOf
if (-not $run) { throw "no $tag run in $Out" }
$ck = Join-Path $run.FullName $last
if (-not (Test-Path $ck)) { throw "missing $ck" }
& $PY -B scripts\export_policy.py $ck $newPol
if ($LASTEXITCODE -ne 0) { throw "export of $ck failed" }
& $PY -B scripts\export_policy.py $start $startPol
if ($LASTEXITCODE -ne 0) { throw "export of $start failed" }
if ($NoTest) { Write-Host "exported $newPol"; exit 0 }

function Test-One([string]$pol, [string]$name, [string]$lay, [int]$s) {
    $json = "runs\eval\sku1v2_${name}_${lay}_seed$s.json"
    if (Test-Path $json) { return }
    & $PY -B tools\terrain_eval.py --policy $pol --terrain stairs_built --layout "data\stairs\$lay.json" `
        --width 0.20 --landing 0.25 --envs $EvalEnvs --seconds 30 --heading-loop --stop-at-goal `
        --tilt-limit 1.0 --seed $s --json $json @massArgs
    if ($LASTEXITCODE -ne 0) { throw "the test of $pol failed" }
}
foreach ($s in 1, 2) {
    foreach ($lay in @("printed_r25_t50_x4", "printed_r20_25_25")) {
        Test-One $startPol "v3start_$Name" $lay $s
        Test-One $newPol "v3_$Name" $lay $s
    }
}
Write-Host "done: $newPol ($ck)"
