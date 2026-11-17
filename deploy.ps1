<#
.SYNOPSIS
    Copy the exported .mpxp files (out\) into an mpv2onlyRL checkout, keeping the files they replace.

.DESCRIPTION
    mpv2onlyRL builds in every policy file it finds in main\rl\policies\ (main\CMakeLists.txt):
      default.mpxp           the walk the robot boots with     ("Walk - SKU1 v2 - 405 g")
      sku1v2_recovery.mpxp   the walker with the get-up        ("Walk + get-up - SKU1 v2")
      getup_sku1.mpxp        the get-up (Get up, "Always gets back up")
      sku1v2_stairs.mpxp     the stairs walk                   ("Stairs - SKU1 v2")
    The files replaced are moved to out\replaced_<date>\ first. Then, in the firmware folder, with "This robot"
    (APP_POLICY_ROBOT) set to sku1:   idf.py reconfigure   and   idf.py build flash

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy.ps1 -Firmware C:\esp\projects\mpv2onlyRL
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy.ps1 -Firmware C:\esp\projects\mpv2onlyRL -Only getup_sku1.mpxp
#>
param([Parameter(Mandatory = $true)][string]$Firmware, [string[]]$Only = @())
. "$PSScriptRoot\scripts\common.ps1"
$dst = Join-Path $Firmware "main\rl\policies"
if (-not (Test-Path $dst)) { throw "no $($dst) (-Firmware is the mpv2onlyRL folder)" }
$files = Get-ChildItem $OutDir -Filter "*.mpxp" -ErrorAction SilentlyContinue
if ($Only) { $files = $files | Where-Object { $Only -contains $_.Name } }
if (-not $files) { throw "nothing to deploy in $OutDir (train.ps1 -Stage export)" }
$keep = Join-Path $OutDir ("replaced_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($f in $files) {
    $target = Join-Path $dst $f.Name
    if (Test-Path $target) {
        if ((Get-FileHash $target).Hash -eq (Get-FileHash $f.FullName).Hash) { Write-Host "$($f.Name): already there"; continue }
        New-Item -ItemType Directory -Force $keep | Out-Null
        Move-Item $target (Join-Path $keep $f.Name)
    }
    Copy-Item $f.FullName $target
    Write-Host "$($f.Name) -> $target" -ForegroundColor Green
}
if (Test-Path $keep) { Write-Host "the replaced files: $keep" }
$sdk = Join-Path $Firmware "sdkconfig"
if ((Test-Path $sdk) -and -not (Select-String -Path $sdk -Pattern '^CONFIG_APP_POLICY_ROBOT="sku1"' -Quiet)) {
    Write-Warning "$sdk does not say CONFIG_APP_POLICY_ROBOT=""sku1"": set This robot = sku1 (idf.py menuconfig) so these policies use the robot's calibration"
}
Write-Host ""
Write-Host "next, in $($Firmware):  idf.py reconfigure ; idf.py build flash"
