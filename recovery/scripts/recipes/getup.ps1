# The get-up recipe (scripts/recipes/getup.sh) for Windows / PowerShell, step for step: a supervised clone of the searched
# motions, then seven PPO steps. From the repository root:
#
#   powershell -ExecutionPolicy Bypass -File scripts\recipes\getup.ps1 [-Seed 41] [-Out runs\getup\s41]
#
# MP4R_ROBOT picks the robot (sku2 or sku1v2) and with it the searched motions (data\searched_motions or
# data\searched_motions_sku1v2). Each step fine-tunes the previous step's last checkpoint; a finished step (its .done file)
# is skipped when the script runs again, an interrupted one starts over. -Envs: robots per PPO step (2048, as the recipe;
# fewer with less GPU memory). -Iters / -CloneEpochs / -CloneRobots / -CloneRounds / -Motions: only for a quick check
# (check.ps1).
param([int]$Seed = 41, [string]$Out = "", [int]$Envs = 2048, [int]$Iters = 0, [int]$CloneEpochs = 300,
      [int]$CloneRobots = 256, [int]$CloneRounds = 4, [string]$Motions = "", [string]$Python = ".venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not $Out) { $Out = "runs\getup\s$Seed" }
$env:WANDB_MODE = "disabled"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path $Python)) { throw "no Python at $Python (setup.ps1)" }
$robot = if ($env:MP4R_ROBOT) { $env:MP4R_ROBOT.ToLower() } else { "sku2" }
$S = if ($Motions) { $Motions } elseif ($robot -eq "sku1v2") { "data/searched_motions_sku1v2" } else { "data/searched_motions" }
foreach ($m in "search_left", "search_nose", "search_right_a", "search_right_b", "search_back_a", "search_back_b") {
    if (-not (Test-Path "$S/$m.json")) { throw "missing $S/$m.json (for sku1v2: tools\sku1v2_motions.py first)" }
}
New-Item -ItemType Directory -Force $Out | Out-Null

$COMMON = @("--limits", "--getup-phase", "4", "--getup-band", "90", "180", "180", "--getup-episode-s", "8",
    "--std-bounds", "0.03", "0.3", "--init-std", "0.03", "--entropy-coef", "0", "--getup-success-gyro", "0",
    "--getup-end-on-success", "--getup-midfall", "0.2",
    "--getup-mix", "0.10", "0.10", "0.30", "0.20", "0.20", "0.10", "0.10", "0.10", "0.30", "0.20", "0.20", "0.10",
    "--actor-hidden", "512", "256", "128", "--critic-hidden", "512", "256", "128", "--obs-norm",
    "--obs-norm-freeze", "100", "--lr0", "1e-5", "--seed", "$Seed")
$WIDE = @("--widen-limits-deg", "60")       # trained inside the limits, it never learns to flip from the back
$COSTS = @("--getup-action-rate=-0.1", "--getup-action-rate-tilted", "1.0", "--getup-joint-vel=-5e-4")
$RAMP = @("--getup-filter", "0.75", "--getup-filter-start", "0.5", "--getup-filter-ramp", "200")
$FILTER = @("--getup-filter", "0.75", "--getup-filter-start", "0.75", "--getup-filter-ramp", "0")
$LR = @("--lr-max", "5e-5")
$SMOOTH_A = @("--caps-temporal", "1", "--caps-spatial", "20", "--caps-obs-noise", "0.05")
$SMOOTH_B = @("--caps-temporal", "2", "--caps-spatial", "40", "--caps-obs-noise", "0.05")

function Py([string[]]$a) {
    & $Python -B @a
    if ($LASTEXITCODE -ne 0) { throw "failed: python $($a -join ' ')" }
}
function LastCkpt([string]$name) {   # the highest-numbered checkpoint of the newest run with this name
    $run = Get-ChildItem $Out -Directory -Filter "*_$name" -ErrorAction SilentlyContinue | Sort-Object Name |
        Select-Object -Last 1
    if (-not $run) { return $null }
    $ck = Get-ChildItem $run.FullName -Filter "model_*.pt" | Sort-Object { [int]($_.BaseName -replace "model_", "") } |
        Select-Object -Last 1
    if ($ck) { return $ck.FullName } else { return $null }
}
$script:PREV = $null
# (PowerShell names ignore case: the step's own count is $n, so that $script:Iters, the script's -Iters, is not hidden)
function Step([string]$name, [int]$n, [int]$warm, [string[]]$opts) {
    if ($script:Iters -gt 0) { $n = $script:Iters }
    if (-not (LastCkpt $name) -or -not (Test-Path "$Out\$name.done")) {
        Write-Host "--- $name : $n iterations   $(Get-Date -Format 'HH:mm')"
        Py (@("-m", "mp4_recovery.gpu.train") + $COMMON + @("--critic-warmup", "$warm", "--envs", "$Envs",
            "--iters", "$n", "--out", $Out, "--tag", $name) + $opts)
        New-Item -ItemType File -Force "$Out\$name.done" | Out-Null
    } else { Write-Host "--- $name : done, skipped" }
    $script:PREV = LastCkpt $name
}

if (-not (Test-Path "$Out\clone\model_bc.pt")) {
    Write-Host "--- the clone of the searched motions   $(Get-Date -Format 'HH:mm')"
    Py (@("-m", "mp4_recovery.gpu.train") + $COMMON + $WIDE + @("--critic-warmup", "50", "--envs", "64", "--iters", "1",
        "--out", "$Out\cfg", "--tag", "cfg"))
    $cfg = Get-ChildItem "$Out\cfg" -Directory -Filter "*_cfg" | Sort-Object Name | Select-Object -Last 1
    Py @("-m", "mp4_recovery.gpu.getup_bc", "--seq", "left=$S/search_left.json", "--seq", "nose=$S/search_nose.json",
        "--seq", "right=$S/search_right_a.json", "--seq", "right=$S/search_right_b.json",
        "--seq", "back=$S/search_back_a.json", "--seq", "back=$S/search_back_b.json",
        "--config", "$($cfg.FullName)\config.json", "--hidden", "512", "256", "128", "--obs-norm",
        "--prev-action-noise", "0.3", "--robots", "$CloneRobots", "--rounds", "$CloneRounds", "--epochs", "$CloneEpochs",
        "--eval-extra", "midfall", "--eval-robots", "64", "--seed", "$Seed", "--out", "$Out\clone")
}
Step "wide" 1500 50 (@("--init-actor", "$Out\clone\model_bc.pt") + $WIDE)
Step "limits" 1000 50 (@("--resume", $script:PREV) + $COSTS)
Step "filter" 253 30 (@("--resume", $script:PREV) + $COSTS + $RAMP + $LR)
Step "smooth1" 400 30 (@("--resume", $script:PREV) + $COSTS + $FILTER + $LR + $SMOOTH_A)
Step "smooth2" 200 30 (@("--resume", $script:PREV) + $COSTS + $FILTER + $LR + $SMOOTH_A)
Step "smooth3" 600 30 (@("--resume", $script:PREV) + $COSTS + $FILTER + $LR + $SMOOTH_B)
Step "smooth4" 600 30 (@("--resume", $script:PREV) + $COSTS + $FILTER + $LR + $SMOOTH_B)
Write-Host "LAST $script:PREV"
