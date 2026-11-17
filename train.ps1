<#
.SYNOPSIS
    Train every SKU1 v2 policy with SKU2's recipes, test them in simulation and export the .mpxp files the robot runs.

.DESCRIPTION
    Five stages, in this order (docs/PIPELINE.md has every option of every step):

      walk     The flat omnidirectional walk (SKU2's bam_t5ub06_stops_v1 recipe): a supervised clone of the firmware
               trot with stops, then anchored PPO, 1300 iterations per seed; the omni gates; the export.
      stairs   SKU2's rough-ground and stairs chain (walk\scripts\recipes\terrain_stairs.sh): six steps, 6200
               iterations, each fine-tuning the one before, starting from this robot's own walk; the printed-staircase
               tests; exports at iterations 4500 and 4897.
      walker   SKU2's walker recipe (recovery\scripts\recipes\walker.sh): the walk with the joint bounds and the
               abductions' stand 12 deg wider; shoved in simulation against the walk without the wider stance.
      getup    SKU2's get-up recipe (recovery\scripts\recipes\getup.ps1): the searched motions replayed (and searched
               again where SKU2's fail) on SKU1 v2, a supervised clone, seven PPO steps; the get-up from seven starting
               poses, the recovery repository's thresholds, the walk / shove / fall / get-up / walk sequence.
      export   The four .mpxp files for mpv2onlyRL in out\, a manifest and a report.

    Each stage skips what is already finished, so running the script again after a stop continues where it stopped.
    The choices one stage hands to the next (the walk seed, the stairs checkpoint, the get-up seed) are in
    out\choices.json; each is made from the tests unless given.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File train.ps1                          # everything (about 1 day on an RTX 4090)
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File train.ps1 -Stage walk              # one stage
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File train.ps1 -Stage getup -GetupSeeds 41,42 -ReuseMotions
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File train.ps1 -Stage export -WalkSeed 2 # export with a chosen walk
#>
param(
    [ValidateSet("all", "walk", "stairs", "walker", "getup", "export")]
    [string[]]$Stage = @("all"),
    [int[]]$WalkSeeds = @(1, 2),          # PPO's outcome depends on the seed; the better one is kept
    [int]$WalkSeed = 0,                   # 0: the seed that passes the omni gates with the fewest joystick falls
    [int]$WalkEnvs = 512,                 # the recipe's; 256 with less GPU memory
    [int]$StairsEnvs = 1024,              # the recipe's; 512 with less GPU memory
    [ValidateSet("own", "sku2")]
    [string]$StairsInit = "own",          # step 1's warm start: this robot's walk (own) or SKU2's walk (sku2)
    [int]$StairsIter = 0,                 # 0: the stairs export (4500 or 4897) with the higher success
    [int[]]$GetupSeeds = @(41),           # the recipe's seed; more seeds, ~12 h each
    [int]$GetupSeed = 0,                  # 0: the seed that passes the thresholds with the highest mean success
    [int]$GetupEnvs = 2048,               # the recipe's; 1024 with less GPU memory
    [switch]$ReuseMotions,                # start the get-up from results\'s searched motions (skips 0-3 h)
    [switch]$EvalOnly                     # no training: the tests, the choices and the export again
)
. "$PSScriptRoot\scripts\common.ps1"
Initialize-Environment
$all = $Stage -contains "all"
function Want([string]$s) { return $all -or ($Stage -contains $s) }
$started = Get-Date

# ---------------------------------------------------------------------------------------------------------- walk
$WalkOut = "runs\sku1v2_405_walk"                 # under walk\
$WalkPolicy = "policies\sku1v2_405_walk"          # walk\policies\sku1v2_405_walk_s<seed>
$WalkIters = 1300

function Select-WalkSeed {
    if ($WalkSeed -gt 0) { return $WalkSeed }
    $best = $null
    foreach ($s in $WalkSeeds) {
        $run = Get-LatestRun (Join-Path $Walk "$WalkOut\ppo") "*_stops_s${s}_sku1v2"
        if (-not $run) { continue }
        $ev = Read-Json (Join-Path $run.FullName "eval_omni_model_$($WalkIters - 1).json")
        if (-not $ev -or -not (Test-Path (Join-Path $Walk "${WalkPolicy}_s$s\policy.npz"))) { continue }
        $score = [pscustomobject]@{ Seed = $s; Pass = [bool]$ev.pass; Falls = [double]$ev.joystick.fall_frac }
        Write-Host ("walk seed {0}: omni gates {1}, joystick falls {2:N3}" -f $s, $(if ($score.Pass) { "PASS" } else { "fail" }), $score.Falls)
        if (-not $best -or ($score.Pass -and -not $best.Pass) -or
            ($score.Pass -eq $best.Pass -and $score.Falls -lt $best.Falls)) { $best = $score }
    }
    if (-not $best) {
        $prev = (Get-Choices).walk_seed
        if ($prev) { return [int]$prev }
        throw "no finished walk (train.ps1 -Stage walk first, or -WalkSeed)"
    }
    if (-not $best.Pass) { Write-Warning "no walk seed passes every omni gate: seed $($best.Seed) has the fewest falls" }
    return $best.Seed
}

if (Want "walk") {
    Write-Stage "1/5 walk: SKU2's flat omni walk recipe on SKU1 v2, seeds $($WalkSeeds -join ', ')"
    Invoke-Script "$Walk\tools\train_sku1v2_walk.ps1" @{
        Seeds = $WalkSeeds; Envs = $WalkEnvs; Iters = $WalkIters; Out = $WalkOut; Policy = $WalkPolicy
        EvalOnly = [bool]$EvalOnly }
    $ws = Select-WalkSeed
    Set-Choice "walk_seed" $ws
    Write-Host "the walk: walk\${WalkPolicy}_s$ws" -ForegroundColor Green
}

# -------------------------------------------------------------------------------------------------------- stairs
$StairsOut = "runs\sku1v2_405_stairs"
$StairsPolicy = "policies\sku1v2_405_stairs_narrow"
$StairsTag = "w405_"

function Select-StairsIter {
    if ($StairsIter -gt 0) { return $StairsIter }
    $best = $null
    foreach ($it in 4500, 4897) {
        if (-not (Test-Path (Join-Path $Walk "${StairsPolicy}_$it\policy.npz"))) { continue }
        $v = @()
        foreach ($seed in 1, 2) {
            $j = Read-Json (Join-Path $Walk "runs\eval\sku1v2_${StairsTag}stairs_${it}_printed_r25_t50_x4_seed$seed.json")
            if ($j) { $v += [double]$j.cells[0].success }
        }
        if (-not $v) { continue }
        $m = ($v | Measure-Object -Average).Average
        Write-Host ("stairs {0}: success {1:P1} on the printed 4 x 25 mm staircase" -f $it, $m)
        if (-not $best -or $m -gt $best.Success) { $best = [pscustomobject]@{ Iter = $it; Success = $m } }
    }
    if (-not $best) {
        $prev = (Get-Choices).stairs_iter
        if ($prev) { return [int]$prev }
        throw "no finished stairs policy (train.ps1 -Stage stairs first, or -StairsIter)"
    }
    return $best.Iter
}

if (Want "stairs") {
    $p = @{ Envs = $StairsEnvs; Masses = $Masses; Out = $StairsOut; Policy = $StairsPolicy; Tag = $StairsTag
            EvalOnly = [bool]$EvalOnly }
    if ($StairsInit -eq "own") {
        $ws = Select-WalkSeed
        $p.Init = "${WalkPolicy}_s$ws"
        Write-Stage "2/5 stairs: SKU2's six-step rough-ground and stairs chain, from this robot's walk (seed $ws)"
    } else {
        Write-Stage "2/5 stairs: SKU2's six-step rough-ground and stairs chain, from SKU2's walk"
    }
    Invoke-Script "$Walk\tools\train_sku1v2_stairs.ps1" $p
    $si = Select-StairsIter
    Set-Choice "stairs_iter" $si
    Write-Host "the stairs walk: walk\${StairsPolicy}_$si" -ForegroundColor Green
}

# -------------------------------------------------------------------------------------------------------- walker
if (Want "walker") {
    $ws = Select-WalkSeed
    Write-Stage "3/5 walker: SKU2's walker recipe on the walk of seed $ws (bounds, stance 12 deg wider)"
    $src = "walk\${WalkPolicy}_s$ws"
    if (-not (Test-Path "$Root\$src\policy.npz")) { throw "no $src (train.ps1 -Stage walk first)" }
    Invoke-Py $Root @("tools\walk_to_walker.py", $src, "recovery\policies\sku1v2_walker_base") "the walker base"
    Invoke-Py $Recovery @("-m", "mp4_recovery.export_policy", "--from-policy", "policies\sku1v2_walker_base",
                          "policies\sku1v2_walker", "--limits", "--stand-offset-abd", "12") "the walker"
    Invoke-Py $Recovery @("-m", "mp4_recovery.export_policy", "--from-policy", "policies\sku1v2_walker_base",
                          "policies\sku1v2_walker_stance0", "--limits") "the walker without the wider stance"
    Invoke-Py $Recovery @("-m", "mp4_recovery.deploy.runner", "check", "--policy", "policies\sku1v2_walker") "runner check of the walker"
    # the shove tests belong to one walk: start them over when the walk changed
    $stamp = Join-Path $Recovery "runs\eval\walker_source.txt"
    New-Item -ItemType Directory -Force (Split-Path $stamp) | Out-Null
    $hash = (Get-FileHash "$Root\$src\policy.npz" -Algorithm SHA256).Hash
    if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -ne $hash)) {
        Remove-Item "$Recovery\runs\eval\push_sku1v2_walker*.json", "$Recovery\runs\eval\sequence_*.json" -ErrorAction SilentlyContinue
    }
    Set-Content -Encoding ascii $stamp $hash
    foreach ($w in "sku1v2_walker_stance0", "sku1v2_walker") {
        $j = "runs\eval\push_$w.json"
        if (-not (Test-Path "$Recovery\$j")) {
            Invoke-Py $Recovery @("-m", "mp4_recovery.eval.push_eval", "--policy", "policies\$w", "--seed", "11",
                                  "--schedule", "0.5", "1.5", "2", "4", "--json", $j) "push_eval $w"
        }
    }
    Set-Choice "walker_from" $src
}

# --------------------------------------------------------------------------------------------------------- getup
function Select-GetupSeed {
    if ($GetupSeed -gt 0) { return $GetupSeed }
    $best = $null
    foreach ($s in $GetupSeeds) {
        if (-not (Test-Path "$Recovery\policies\sku1v2_getup_s$s\policy.npz")) { continue }
        $t = Read-Json "$Recovery\runs\eval\thresholds_s$s.json"
        if (-not $t) { continue }
        $succ = @($t.starts.PSObject.Properties | Where-Object { $null -ne $_.Value.checks.success } |
                  ForEach-Object { [double]$_.Value.checks.success.value })    # (crouch has no success check)
        $m = ($succ | Measure-Object -Average).Average
        $pass = $t.verdict -eq "pass"
        Write-Host ("get-up seed {0}: thresholds {1}, mean success {2:P1}" -f $s, $t.verdict, $m)
        if (-not $best -or ($pass -and -not $best.Pass) -or ($pass -eq $best.Pass -and $m -gt $best.Success)) {
            $best = [pscustomobject]@{ Seed = $s; Pass = $pass; Success = $m }
        }
    }
    if (-not $best) {
        $prev = (Get-Choices).getup_seed
        if ($prev) { return [int]$prev }
        throw "no finished get-up (train.ps1 -Stage getup first, or -GetupSeed)"
    }
    if (-not $best.Pass) { Write-Warning "no get-up seed passes the thresholds: seed $($best.Seed) is the best" }
    return $best.Seed
}

if (Want "getup") {
    Write-Stage "4/5 get-up: SKU2's get-up recipe on SKU1 v2, seeds $($GetupSeeds -join ', ')"
    if (-not (Test-Path "$Recovery\policies\sku1v2_walker\policy.npz")) { throw "no walker (train.ps1 -Stage walker first)" }
    if ($ReuseMotions) {
        $dst = "$Recovery\data\searched_motions_sku1v2"
        New-Item -ItemType Directory -Force $dst, "$Recovery\runs\motions" | Out-Null
        Copy-Item "$Root\results\recovery\searched_motions_sku1v2\search_*.json" $dst
        Copy-Item "$Root\results\recovery\searched_motions_sku1v2\summary.json" "$Recovery\runs\motions\summary.json"
        Write-Host "the searched motions: results\recovery\searched_motions_sku1v2 (the robot's current get-up's)"
    }
    if (-not $EvalOnly) {
        Write-Host "--- the motions: SKU2's six get-ups replayed on SKU1 v2, searched again where they fail"
        Invoke-Py $Recovery @("tools\sku1v2_motions.py", "--threshold", "0.75") "the searched motions"
    }
    foreach ($seed in $GetupSeeds) {
        $out = "runs\getup\s$seed"
        if (-not $EvalOnly) {
            Write-Host "--- the recipe, seed $seed (clone, wide, limits, filter, smooth1-4)"
            Invoke-Script "$Recovery\scripts\recipes\getup.ps1" @{ Seed = $seed; Out = $out; Envs = $GetupEnvs; Python = $PY }
        }
        $last = Get-LatestRun "$Recovery\$out" "*_smooth4"
        if (-not $last) { Write-Warning "seed ${seed}: no smooth4 run in recovery\$out"; continue }
        $ck = Get-LastCheckpoint $last.FullName
        $pol = "policies\sku1v2_getup_s$seed"
        Write-Host "--- seed ${seed}: export $($ck.Name) and test"
        Invoke-Py $Recovery @("-m", "mp4_recovery.export_policy", $ck.FullName, $pol) "export"
        Invoke-Py $Recovery @("-m", "mp4_recovery.deploy.runner", "check", "--policy", $pol) "runner check of the get-up"
        $ge = "runs\eval\getup_s${seed}_s12.json"
        if (-not (Test-Path "$Recovery\$ge")) {
            Invoke-Py $Recovery @("-m", "mp4_recovery.eval.getup_eval", "--policy", $pol, "--seed", "12", "--robots", "256",
                "--seconds", "8", "--starts", "left,right,nose,back,midfall,crouch,random", "--motion", "--json", $ge) "getup_eval"
        }
        Push-Location $Recovery
        & $PY -B -m mp4_recovery.eval.thresholds $ge --json "runs\eval\thresholds_s$seed.json"   # exit 1 = a failed threshold
        Pop-Location
        $sq = "runs\eval\sequence_s${seed}_s12.json"
        if (-not (Test-Path "$Recovery\$sq")) {
            Invoke-Py $Recovery @("-m", "mp4_recovery.eval.sequence", "--walker", "policies\sku1v2_walker", "--getup", $pol,
                "--schedule", "0.5", "1.5", "2", "4", "--robots", "256", "--seconds", "60", "--seed", "12",
                "--getup-phase-restart", "1.0", "--json", $sq) "the sequence"
        }
    }
    $gs = Select-GetupSeed
    Set-Choice "getup_seed" $gs
    Write-Host "the get-up: recovery\policies\sku1v2_getup_s$gs" -ForegroundColor Green
}

# -------------------------------------------------------------------------------------------------------- export
if (Want "export") {
    Write-Stage "5/5 export: the .mpxp files for mpv2onlyRL"
    $wc = "walk\mp3_walk_policy\data\sku2_contract.json"       # SKU2's signs and stand, this robot's calibration
    $wm = "walk\mp3_walk_policy\data\imu_mount.json"
    $rc = "recovery\src\mp4_recovery\data\sku1v2_contract.json"
    $rm = "recovery\src\mp4_recovery\data\imu_mount.json"
    $items = @()
    try { $ws = Select-WalkSeed } catch { $ws = $null; Write-Warning $_ }
    if ($ws) { Set-Choice "walk_seed" $ws; $items += ,@("default.mpxp", "walk\${WalkPolicy}_s$ws", $wc, $wm, "sku1v2_405_walk_s$ws") }
    try { $si = Select-StairsIter } catch { $si = $null; Write-Warning $_ }
    if ($si) { Set-Choice "stairs_iter" $si; $items += ,@("sku1v2_stairs.mpxp", "walk\${StairsPolicy}_$si", $wc, $wm, "sku1v2_405_stairs_$si") }
    if (Test-Path "$Recovery\policies\sku1v2_walker\policy.npz") {
        $items += ,@("sku1v2_recovery.mpxp", "recovery\policies\sku1v2_walker", $rc, $rm, "sku1v2_recovery_walker")
    }
    try { $gs = Select-GetupSeed } catch { $gs = $null; Write-Warning $_ }
    if ($gs) { Set-Choice "getup_seed" $gs; $items += ,@("getup_sku1.mpxp", "recovery\policies\sku1v2_getup_s$gs", $rc, $rm, "sku1v2_recovery_getup_s$gs") }
    $manifest = [ordered]@{ robot = "sku1v2 (firmware: sku1)"; masses = $Masses; made = (Get-Date -Format "s"); files = [ordered]@{} }
    foreach ($i in $items) {
        $file, $src, $contract, $mount, $name = $i
        Invoke-Py $Root @("tools\export_mpxp.py", $src, "--contract", $contract, "--mount", $mount, "--robot", "sku1",
                          "--name", $name, "--out", "out\$file") "the export of $src"
        $manifest.files[$file] = [ordered]@{ name = $name; source = $src.Replace("\", "/")
            sha256 = (Get-FileHash "$OutDir\$file" -Algorithm SHA256).Hash.ToLower() }
    }
    $manifest.choices = Get-Choices
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding ascii "$OutDir\MANIFEST.json"
}

# -------------------------------------------------------------------------------------------------------- report
Write-Stage "the report"
Invoke-Py $Root @("tools\report.py", "--write", "out\REPORT.md") "the report"
Write-Host ""
Write-Host ("done in {0:N1} h. The robot's files: out\*.mpxp (deploy.ps1 copies them into mpv2onlyRL)" -f ((Get-Date) - $started).TotalHours) -ForegroundColor Green
