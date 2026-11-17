# mini_pupper_4_sku1v2_rl

Reinforcement-learning training for the **Mini Pupper 4 SKU1 v2** (405 g), using the same recipes that produced the SKU2
policies. One repository trains every policy the robot runs. The results are `.mpxp` files that the ESP32-S3 firmware
([mpv2onlyRL](#deploying-to-the-robot)) runs on the robot itself, with no Raspberry Pi and no PC in the loop.

| Policy | What it does | SKU2's recipe it follows | Robot file |
|---|---|---|---|
| **Walk** | omnidirectional walking on flat ground: forward, sideways, turning, stopping | `bam_t5ub06_stops_v1`: a clone of the firmware trot, then anchored PPO | `default.mpxp` |
| **Stairs** | blind stair climbing (IMU and joints only), heading held by a PI loop | `bam_t5ub06_stairs_narrow_v3`: six chained steps, rough ground → narrow flights | `sku1v2_stairs.mpxp` |
| **Walker** | the walk with joint bounds and a 12° wider stance, handing over to the get-up | mini_pupper_4_recovery's `walker.sh` | `sku1v2_recovery.mpxp` |
| **Get-up** | rolls over and stands up from the back, the sides, nose down, mid-fall | mini_pupper_4_recovery's `getup` recipe: searched motions, a clone, seven PPO steps | `getup_sku1.mpxp` |

Only the robot changes from SKU2's training: SKU1's body, 405 g, 177 / 223 g on the front / rear feet, and this robot's
calibration. SKU1 v2 shares SKU2's legs, MD01 servo modules, IMU and servo mapping, so the servo model, the
observations, the actions and every training option stay the same. [docs/ROBOT.md](docs/ROBOT.md) lists exactly
what differs.

## Requirements

- Windows 10/11 with PowerShell. The pipeline scripts are PowerShell. The Linux shell recipes of the two training
  code bases are in `walk/scripts/recipes/` and `recovery/scripts/recipes/`.
- An NVIDIA GPU with at least 8 GB of memory and driver 580 or newer (CUDA 13). It was developed on an RTX 4090.
- About 15 GB of disk space: 5.5 GB for the Python environment, and the rest for runs and checkpoints.
- Internet access for `setup.ps1`. It installs [uv](https://docs.astral.sh/uv/) if missing, which then installs Python 3.12.

## Quick start

```powershell
git clone <this repository> mini_pupper_4_sku1v2_rl
cd mini_pupper_4_sku1v2_rl
powershell -ExecutionPolicy Bypass -File setup.ps1     # once: .venv with PyTorch, MuJoCo Warp, mjlab (20-30 min)
powershell -ExecutionPolicy Bypass -File check.ps1     # every stage for a few iterations (20-30 min): "check OK"
powershell -ExecutionPolicy Bypass -File train.ps1     # everything: about a day on an RTX 4090
```

`train.ps1` runs five stages in order and can be stopped and restarted at any time, because finished work is skipped.
Watch a stage while it trains with `watch.ps1 -Stage walk|stairs|getup` (TensorBoard on http://localhost:6006).

| Stage | Command | Time (RTX 4090, approximate) | Output |
|---|---|---|---|
| 1. walk | `train.ps1 -Stage walk` | 2-4 h (two seeds) | `walk/policies/sku1v2_405_walk_s<seed>` |
| 2. stairs | `train.ps1 -Stage stairs` | 4-5 h | `walk/policies/sku1v2_405_stairs_narrow_<iter>` |
| 3. walker | `train.ps1 -Stage walker` | under 30 min | `recovery/policies/sku1v2_walker` |
| 4. get-up | `train.ps1 -Stage getup` | 12-15 h per seed | `recovery/policies/sku1v2_getup_s<seed>` |
| 5. export | `train.ps1 -Stage export` | 1 min | `out/*.mpxp`, `out/MANIFEST.json`, `out/REPORT.md` |

Stages 2 and 3 start from the walk that stage 1 picked, so run stage 1 first. Stage 4 needs the walker from stage 3
(for the walk / fall / get-up sequence test). When a stage trains more than one seed, the seed it keeps is chosen from
the simulation tests and recorded in `out/choices.json`. `-WalkSeed`, `-StairsIter` and `-GetupSeed` override the
choice. Every option is listed in `Get-Help .\train.ps1 -Detailed`.

Useful options:

```powershell
train.ps1 -Stage getup -ReuseMotions          # start the get-up from the robot's current searched motions (skips 0-3 h)
train.ps1 -Stage getup -GetupSeeds 41,42      # two get-up seeds; the better one is exported
train.ps1 -Stage stairs -StairsInit sku2      # start the stairs chain from SKU2's walk, as the earlier SKU1 v2 runs did
train.ps1 -WalkEnvs 256 -StairsEnvs 512 -GetupEnvs 1024    # less GPU memory
train.ps1 -EvalOnly                           # no training: the tests, the choices and the export again
```

## Deploying to the robot

```powershell
powershell -ExecutionPolicy Bypass -File deploy.ps1 -Firmware C:\esp\projects\mpv2onlyRL
```

This copies `out/*.mpxp` into the firmware's `main/rl/policies/`, moving any files it replaces to `out/replaced_<date>/`.
Then build the firmware with **This robot** (`APP_POLICY_ROBOT`) set to `sku1`:

```powershell
cd C:\esp\projects\mpv2onlyRL
idf.py reconfigure
idf.py build flash
```

Before the first get-up on the real robot, check the calibration and test on a soft surface. The joint limits are
estimates (see [docs/ROBOT.md](docs/ROBOT.md#joint-limits)). [docs/DEPLOY.md](docs/DEPLOY.md) covers the file names, the
labels in the app, memory, and how to go back.

## What is on the robot today

`results/` holds the policies that run on the robot now, with their simulation results, as a reference for new
training runs. The walk and stairs files there were trained by earlier, separate SKU1 v2 runs of these recipes. See
[results/README.md](results/README.md).

| | Simulation result |
|---|---|
| Get-up, seed 41 | 99-100 % from every start (left, right, nose, back, mid-fall, crouch, random); 0.9-3.1 s median; the recovery thresholds **pass** |
| Walker shoved (0.5-1.5 m/s every 2-4 s) | falls 38-59 % with the 12° wider stance, against 77-91 % without |
| Walk / shove / fall / get-up / walk, 60 s | topple rate 0.12; walking again within 8 s: 0.52 |

## Repository layout

```
setup.ps1  check.ps1  train.ps1  watch.ps1  deploy.ps1   the pipeline (start here)
scripts/common.ps1          paths, environment variables and helpers shared by the scripts
tools/
  export_mpxp.py            policy.npz -> .mpxp, the robot's format (the same file as mpv2onlyRL/tools/policy)
  walk_to_walker.py         a walk from walk/ -> the recovery repository's walker base
  report.py                 every stage's tests in one table (out/REPORT.md)
walk/                       RLmp4 (mp3_walk_policy): walk, rough ground and stairs training, SKU1 v2 port
recovery/                   mini_pupper_4_recovery: walker and get-up training, SKU1 v2 port
third_party/bam/            BAM, the MD01 servo model (Rhoban, Apache-2.0; MangDang's MD01 parameters)
third_party/minipupperesp/  SKU1's simulation model and the firmware gait sources (the walk's teacher)
results/                    the policies on the robot now, with their tests
docs/                       PIPELINE.md, ROBOT.md, DEPLOY.md
out/                        (made by train.ps1) the robot's files, the manifest and the report
```

`walk/` and `recovery/` are full copies of the two training code bases. Their own READMEs describe them as they are for
SKU2. The SKU1 v2 additions are `MP3_ROBOT=sku1v2` and `MP3_MASSES` in `walk/` (its `docs/SKU1V2_WALK.md` and
`docs/SKU1V2_STAIRS.md`), and `MP4R_ROBOT=sku1v2` in `recovery/` (its `docs/SKU1V2.md`). The pipeline scripts set these
variables, so you don't need to.

## Documentation

| | |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | every stage step by step: options, iterations, checkpoints, tests, the choices, what to watch |
| [docs/ROBOT.md](docs/ROBOT.md) | the SKU1 v2 model against SKU2's: body, masses, calibration, joint limits, what is estimated |
| [docs/DEPLOY.md](docs/DEPLOY.md) | the `.mpxp` files on the robot: names, app labels, memory, calibration, going back |
| [results/README.md](results/README.md) | the policies on the robot now and their provenance |

## License

Apache License 2.0 ([LICENSE](LICENSE)). The vendored code keeps its own license and notices; see [NOTICE](NOTICE).
