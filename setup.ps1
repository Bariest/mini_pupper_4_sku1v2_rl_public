<#
.SYNOPSIS
    Once per training PC: the Python environment for every stage (.venv in this folder).

.DESCRIPTION
    Python 3.12 with PyTorch 2.14 for CUDA 13, MuJoCo 3.7, MuJoCo Warp, mjlab 1.3 and RSL-RL at the versions both
    training repositories were validated with (recovery\requirements\constraints.txt), the recovery package
    (recovery\, editable) and BAM, the servo model (third_party\bam, editable). Then the MuJoCo Warp patch that removes
    a GPU warning print flooding rough-ground and stairs training (the physics is unchanged).

    Needs the internet (about 5.5 GB of wheels, 20-30 min the first time) and an NVIDIA driver 580 or newer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -TorchBackend cu128     # an older driver (untested pin set)
#>
param([string]$TorchBackend = "cu130")
. "$PSScriptRoot\scripts\common.ps1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if (Test-Path "$env:USERPROFILE\.local\bin\uv.exe") {
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    } else {
        Write-Host "installing uv (the Python installer, https://docs.astral.sh/uv/)..."
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
}

Set-Location $Root
Write-Stage "the environment: .venv (Python 3.12)"
uv venv --python 3.12 --allow-existing .venv
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

Write-Stage "the packages (torch backend $TorchBackend)"
uv pip install --python .venv\Scripts\python.exe `
    -c recovery\requirements\constraints.txt -r recovery\requirements\gpu.txt --torch-backend=$TorchBackend `
    -e recovery -e "third_party\bam[mjlab]" trimesh matplotlib
if ($LASTEXITCODE -ne 0) { throw "the install failed" }

Write-Stage "the MuJoCo Warp patch"
& .venv\Scripts\python.exe -m mp4_recovery.gpu.patch_warp
if ($LASTEXITCODE -ne 0) { throw "the MuJoCo Warp patch failed (another mujoco-warp version?)" }

Write-Stage "the result"
& .venv\Scripts\python.exe -c @"
import torch, mujoco, mujoco_warp, mjlab, rsl_rl, bam, mp4_recovery
ok = torch.cuda.is_available()
print('torch', torch.__version__, '| mujoco', mujoco.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if ok else 'NOT FOUND (check the NVIDIA driver: 580 or newer)')
"@
if ($LASTEXITCODE -ne 0) { throw "the environment does not import" }
Write-Host ""
Write-Host "setup done. Next:  powershell -ExecutionPolicy Bypass -File check.ps1" -ForegroundColor Green
