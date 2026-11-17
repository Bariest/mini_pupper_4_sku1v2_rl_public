# Shared by the top-level scripts (setup, check, train, watch, deploy): the repository's paths, the training
# environment and a few helpers. Dot-source it:   . "$PSScriptRoot\scripts\common.ps1"
$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $PSScriptRoot
$Walk      = Join-Path $Root "walk"          # RLmp4: the flat omni walk and the rough-ground / stairs chain
$Recovery  = Join-Path $Root "recovery"      # mini_pupper_4_recovery: the walker and the get-up
$ThirdParty = Join-Path $Root "third_party"
$OutDir    = Join-Path $Root "out"           # the .mpxp files for the robot, the manifest and the report

# The robot: SKU1 v2 as weighed, 405 g, 177 / 223 g on the front / rear feet (docs/ROBOT.md)
$Masses = "sku1v2_masses_405g.json"

# The one Python environment (setup.ps1). SKU1V2_PY overrides it (another environment, CI).
$PY = if ($env:SKU1V2_PY) { $env:SKU1V2_PY } else { Join-Path $Root ".venv\Scripts\python.exe" }

function Initialize-Environment {
    if (-not (Test-Path $PY)) { throw "no Python at $($PY) (run setup.ps1 first)" }
    $env:SKU1V2_PY = $PY                                               # walk\tools\*.ps1 use it
    $env:BAM_DIR = Join-Path $ThirdParty "bam"                         # the MD01 servo model (BAM)
    $env:MINIPUPPERESP_DIR = Join-Path $ThirdParty "minipupperesp"     # SKU1's body and the firmware gait
    $env:MP3_ROBOT = "sku1v2"                                          # walk/: SKU1's body instead of SKU2's
    $env:MP3_MASSES = $Masses                                          # walk/: every task on the 405 g robot
    $env:MP4R_ROBOT = "sku1v2"                                         # recovery/: the same robot
    if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "disabled" }
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue           # a ROS install's PYTHONPATH breaks Python 3.12
    New-Item -ItemType Directory -Force $OutDir | Out-Null
}

function Write-Stage([string]$text) {
    Write-Host ""
    Write-Host ("=== {0}   {1}" -f $text, (Get-Date -Format "yyyy-MM-dd HH:mm")) -ForegroundColor Cyan
}

# Run Python in a directory; throw with $what when it fails
function Invoke-Py([string]$Dir, [string[]]$Arguments, [string]$What) {
    Push-Location $Dir
    try {
        & $PY -B @Arguments
        if ($LASTEXITCODE -ne 0) { throw "failed: $What (exit code $LASTEXITCODE)" }
    } finally { Pop-Location }
}

# Run one of the sub-repositories' PowerShell scripts; they change directory, so come back after
function Invoke-Script([string]$Path, [hashtable]$Params) {
    Push-Location $Root
    try { & $Path @Params } finally { Pop-Location }
}

function Read-Json([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try { return Get-Content $Path -Raw | ConvertFrom-Json } catch { return $null }
}

# The choices one stage hands to the next (which walk seed, which stairs checkpoint, which get-up seed): out\choices.json
$ChoicesFile = Join-Path $OutDir "choices.json"
function Get-Choices {
    $c = Read-Json $ChoicesFile
    if (-not $c) { $c = [pscustomobject]@{} }
    return $c
}
function Set-Choice([string]$Key, $Value) {
    $c = Get-Choices
    $c | Add-Member -NotePropertyName $Key -NotePropertyValue $Value -Force
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    $c | ConvertTo-Json -Depth 6 | Set-Content -Encoding ascii $ChoicesFile
}

# The newest run folder whose name matches, or $null
function Get-LatestRun([string]$Dir, [string]$Filter) {
    if (-not (Test-Path $Dir)) { return $null }
    Get-ChildItem $Dir -Directory -Filter $Filter | Where-Object { $_.Name -notlike "*_interrupted" } |
        Sort-Object Name | Select-Object -Last 1
}

# The highest-numbered model_<n>.pt in a run folder, or $null
function Get-LastCheckpoint([string]$RunDir) {
    Get-ChildItem $RunDir -Filter "model_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace "model_", "") } | Select-Object -Last 1
}
