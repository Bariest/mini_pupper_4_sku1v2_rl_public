# SKU1 v2: the stair policy on SKU1's body

`policies/bam_t5ub06_stairs_narrow_v3` climbs stairs on SKU2. SKU1 v2 is the same robot with SKU1's body: the same
legs, MD01 modules, IMU, firmware and servo mapping, but SKU1's trunk, at **390 g, 195 g on the front feet and
195 g on the rear feet** (SKU2: 478 g, 197 / 258 g). `MP3_ROBOT=sku1v2` builds that robot everywhere:

| | SKU2 (default) | SKU1 v2 (`MP3_ROBOT=sku1v2`) |
|---|---|---|
| model | mpx-sim `model/sku_2/pupper.xml` | minipupperesp `sim/model/pupper.xml` (from `sku_1_urdf_fixed`) |
| masses (terrain task) | `data/sku2_masses_478g.json` | `data/sku1v2_masses.json` (390 g, 50 % on the rear feet) |
| contract, actuator model, IMU | `data/sku2_contract.json`, `bam_t5ub06`, `imu_mount.json` | the same files |

The contract's stand puts SKU1's model at the same neutral pose (hip axes 70 mm above the ground). The masses file
comes from `tools/robot_masses.py` (each module 66.82 g and each leg 6.5 g as SKU2, the base the rest, its centre of
mass x solved for the 50 % rear share):

```powershell
$env:MP3_ROBOT = "sku1v2"
& $PY -B tools\robot_masses.py --robot sku1v2 --total-g 390 --front-g 195 --rear-g 195
& $PY -B tools\sku1v2_check.py        # 390.0 g, rear 50.0 %, hip axis 70.0 mm, "robot OK"
```

Each run's `config.json` records `"robot"` (also under `task.sources`), and an exported `policy.npz` carries
`"robot"` in its meta. `--resume` refuses a checkpoint from another robot; `tools/terrain_eval.py` warns when a policy
is evaluated on a robot it was not trained on.

## Training (Windows, PowerShell)

`tools/train_sku1v2_stairs.ps1` is `scripts/recipes/terrain_stairs.sh` for this robot: the same six steps, options,
seeds, iterations and resume checkpoints. Step 1 starts from `policies/bam_t5ub06_stops_v1`, as SKU2's did.

| Step | Terrain | Iterations | |
|---|---|---|---|
| 1 | `rough`, 10-step history | 1200 (from `bam_t5ub06_stops_v1`) | the rough-ground walk |
| 2 | `stairs` | 1000 (from step 1, iteration 700) | |
| 3 | `stairs`, `--stair-exit` | 1000 (from step 2, iteration 1500) | |
| 4 | `stairs_narrow`, tilt limit 1.0, the robot's heading law | 1000 (from step 3, iteration 1900) | |
| 5 | as 4, narrow risers 20-40 mm, every difficulty level at the start | 1000 (from step 4, iteration 2899) | |
| 6 | as 5, narrow risers 25-45 mm | 1000 (from step 5, iteration 3898) | exported at 4500 and 4897 |

```powershell
powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs.ps1               # all of it, then export + test
powershell -ExecutionPolicy Bypass -File tools\train_sku1v2_stairs.ps1 -LastStep 1   # the rough-ground walk only
```

Runs go to `runs/sku1v2_stairs/`, the policies to `policies/sku1v2_stairs_narrow_4500` and `_4897`, the tests to
`runs/eval/sku1v2_*.json`: the printed staircases (`data/stairs/`: four 25 mm risers on 50 mm treads, and the 20 / 25 /
25 mm one SKU2 climbed on the robot), 200 mm wide, two seeds of 256 robots, as in the README, for both exports and for
SKU2's shipped policy run unchanged on SKU1 v2 (the reference).

`scripts/train_terrain.py --device cpu` and `tools/terrain_eval.py --device cpu` run without a GPU (slowly), for checks.
