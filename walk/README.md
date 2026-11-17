> **In this repository:** a copy of RLmp4 (mp3_walk_policy), as released for SKU2, with its SKU1 v2 additions: `MP3_ROBOT=sku1v2` and `MP3_MASSES` ([docs/SKU1V2_WALK.md](docs/SKU1V2_WALK.md), [docs/SKU1V2_STAIRS.md](docs/SKU1V2_STAIRS.md)). The top-level scripts (`../train.ps1`) set these variables and run it; see [../README.md](../README.md) and [../docs/PIPELINE.md](../docs/PIPELINE.md). Its own instructions below still work for SKU2.

# mp3_walk_policy

Walking policies for Mini Pupper 3 **SKU2** (the long-body variant with a Raspberry Pi 5), trained in **mjlab**, and
the numpy runner that executes them on the Pi. Each leg is driven by one MD01, MP3's three-motor servo module
(abduction, hip, knee).

The repository trains four kinds of policy:
- **forward walking** on flat ground (a command vx);
- **omnidirectional walking** on flat ground (a joystick command vx, vy, wz);
- **rough ground** (blind: the IMU and the joint angles only, no terrain sensing);
- **stairs** (also blind), fine-tuned from the rough-ground policy.

Each policy is trained against one actuator model, chosen with `--arm`:
- `stock`: mpx-sim's position-servo model (profile `sim`) with its 5 ms latency;
- `bam_published`, `bam_t5ub06`: the MD01 servo models identified with BAM. `bam_published` uses the parameters in
  `bam/bam/params/md01`; `bam_t5ub06` is a refit of them with a lower rotor inertia, which tracks SKU2's joints more
  closely (`mp3_walk_policy/data/bam_t5ub06_id_*_m6.json`). The hip uses the parameters identified on the module's
  hip output (`id_2`); the knee and abduction use the knee output's (`id_3`). The rough-ground and stair policies use
  `bam_t5ub06`.

## Requirements

- Linux x86_64 (training and evaluation; Windows notes for the flat and omni workflow are in
  [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)).
- An NVIDIA GPU, Turing or newer, with a driver for CUDA 13 (version 580 or later). One training run at 1024
  environments uses about 2 GB of GPU memory; one evaluation about 0.6 GB.
- [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (`uv` installs it).
- gcc, to build the firmware gait (the teacher tables and the firmware trot on the robot).
- Sibling checkouts: all of them side by side in one folder with this repository, as in the install commands below
  (or point `MPX_SIM_DIR`, `BAM_DIR`, `MINIPUPPERESP_DIR`, `MP3_LINK_DIR` at them):

| Checkout | Commit | Provides | Needed for |
|---|---|---|---|
| `../mpx-sim` (github.com/mangdang-bam/mpx-sim) | `740b690` | the SKU2 model and the stock servo profile | everything |
| `../bam` (github.com/mangdang-bam/bam, branch `mangdang`) | `13cea06` | the BAM package and the MD01 parameters | everything |
| `../minipupperesp` (github.com/mangdang-bam/minipupperesp, branch `testboard`) | `35adb8b` | the firmware gait sources | the teacher tables, the firmware trot on the robot, some tests |
| `../mp3-link` (github.com/mangdang-bam/mp3-link) | `9dea2ea` | SKU2's ESP32 firmware and the Pi package `mp3link`, with a fake ESP32 for the runner tests | the robot, `tests/robot` |

**ROS users:** a ROS install that puts its Python packages on `PYTHONPATH` breaks the Python 3.12 environment.
Every command below clears it with `env -u PYTHONPATH`.

## Install

```bash
mkdir mp3 && cd mp3
git clone -b khk_goat https://github.com/mangdang-bam/mp3_walk_policy.git
git clone https://github.com/mangdang-bam/mpx-sim.git && git -C mpx-sim checkout 740b690
git clone -b mangdang https://github.com/mangdang-bam/bam.git && git -C bam checkout 13cea06
git clone -b testboard https://github.com/mangdang-bam/minipupperesp.git && git -C minipupperesp checkout 35adb8b
git clone https://github.com/mangdang-bam/mp3-link.git && git -C mp3-link checkout 9dea2ea   # for tests/robot and the robot

cd mp3_walk_policy
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements/gpu.txt -e "../bam[mjlab]"
PY="env -u PYTHONPATH WANDB_MODE=disabled .venv/bin/python -B"    # add MUJOCO_GL=egl to render headless
$PY tools/patch_mujoco_warp.py                                     # once per install, see below
$PY -m pytest -q tests                                             # about 4 min the first time (Warp kernels compile)
```

`requirements/gpu.txt` pins every package of the tested environment (mjlab 1.3.0, mujoco 3.7.0, mujoco-warp 3.7.0.1,
warp-lang 1.12.1, rsl-rl-lib 5.0.1, torch 2.14.0 with CUDA 13): about 5.5 GB of wheels, 25–30 min on first install.
`--index-strategy unsafe-best-match` lets uv take torch and the CUDA wheels from the PyTorch index and the rest from
PyPI. Run every command from this repository's root; `PY` is a shell variable, so define it again in a new shell.

The tests should end with `102 passed, 2 skipped, 1 xfailed`: one test needs omni recordings that are not in mpx-sim
at this commit, and one compares a teacher table built with another compiler. The repeated warnings about "pending
keyframes" and non-default `<option>` fields come from mjlab building the model and are harmless.

**The mujoco-warp patch.** On rough ground, mujoco-warp prints a warning from the GPU for every heightfield contact
it has no room for; in training that is millions of lines and a run about 7 times slower. `tools/patch_mujoco_warp.py`
removes the print (the contact is still skipped, so the physics is unchanged); `--check` reports the state.
`scripts/train_terrain.py` refuses to start without it (`--allow-unpatched` overrides). Reinstalling mujoco-warp
undoes it.

**A first end-to-end check** (a few minutes): a short rough-ground training from the shipped omni policy, its export,
and an evaluation of the shipped stair policy:

```bash
$PY scripts/train_terrain.py --variant C --terrain rough --history 10 --envs 256 --iters 20 --seed 1 --tag check
$PY scripts/export_policy.py runs/terrain/*_check/model_19.pt policies/check      # the run folder is <date>_C_bam_t5ub06_s1_check
$PY tools/terrain_eval.py --policy policies/bam_t5ub06_stairs_narrow_v3 --terrain stairs_built \
    --layout data/stairs/printed_r25_t50_x4.json --width 0.20 --landing 0.25 --envs 64 --seconds 30 \
    --heading-loop --stop-at-goal --tilt-limit 1.0 --seed 1 --json runs/eval/check.json
```

**The runner's tests** need only numpy, pyserial and pytest, in their own environment:

```bash
uv venv --python 3.12 .venv-pi && uv pip install --python .venv-pi/bin/python numpy pyserial pytest
env -u PYTHONPATH .venv-pi/bin/python -B -m pytest -q tests/robot      # the runner against a fake ESP32: 44 passed
```

## Policies

Each folder in `policies/` holds `policy.npz` (weights and metadata) and the training `config.json` (every option,
and for the rough-ground and stair policies the whole training chain); the flat ones also hold their evaluation
results and the runner equivalence check. `data/streams/` holds the flat policies' 0.10 m/s action streams and the
firmware trot's (for `runner replay`).

| Policy | What it does |
|---|---|
| `bam_t5ub06_stairs_narrow_v3` | climbs stairs about the size of the robot: trained on flights 200 mm wide with 25–45 mm risers; 10-step observation history; heading held by the runner's heading loop |
| `bam_t5ub06_terrain_h10_v2` | walks forward over rough ground (relief, slopes, boxes and steps up to 30 mm); 10-step observation history |
| `bam_t5ub06_stops_v1` | omnidirectional walking on flat ground, including stopping and starting; the warm start of the rough-ground and stair training |
| `bam_t5ub06_omni_v0` | omnidirectional walking on flat ground |
| `<arm>_v3n2_s*`, `<arm>_beta5_s*` (run tag `b5n2`), `stock_v3_s4` | forward walking on flat ground (below); `v3n2` = β 20 with the `n2` noise, `v3` = β 20 with `n1`, `beta5` = β 5 with `n2`, `s<k>` = seed |

The `_v2` / `_v3` suffixes are internal iteration numbers; only these versions are released.

In simulation, with the shipped evaluation tools (commands under "Evaluation"; 512 trials each, two seeds of 256):

| Policy | Course | Success |
|---|---|---|
| `bam_t5ub06_stairs_narrow_v3` | a 3D-printed test staircase (`data/stairs/`): four 25 mm risers on 50 mm treads, 200 mm wide | 94.3 % (all four feet on the landing, no fall, every stance foot on the treads) |
| `bam_t5ub06_terrain_h10_v2` | a model of a 3D-printed rough test tile (a 1 mm heightmap) between walls: 0.65 m without a fall | 100 % |

On a real SKU2, `bam_t5ub06_stairs_narrow_v3` climbed a 3-step version of that staircase (risers 20, 25 and
25 mm) in 3 of 5 attempts; both misses stepped off a side.

## Training

### Forward walking (flat ground)

The method comes from the Mini Pupper 2 walking policy:
1. **Teacher:** the firmware's own trot (minipupperesp's gait and IK, compiled on the host), tabulated at the policy's
   50 Hz for 0.05–0.20 m/s.
2. **Behaviour cloning** of the teacher in the chosen actuator model.
3. **PPO from the clone,** with an imitation term (weight β) that keeps the policy's mean action close to the clone's.

The shipped policies use β 20 (or β 5, which lets the policy move further from the teacher's trot), exploration std
fixed at 0.10, 50 critic-only iterations first and a learning rate from 1e-4 (adaptive), set by the `train.py` command
below; 1300 iterations × 512 environments.

**Observation noise** (uniform, ramped in over iterations 150–450): `n2` (default, `mp3_walk_policy/data/noise_n2.json`)
is SKU2's measured sensor noise plus the broadband part of the extra variation seen on the robot while walking; `n1`
is an older, larger profile. Each file's `method` field describes how it was built.

```bash
$PY scripts/distill.py --arm stock --out runs/clone/stock                       # teacher data, clone, clone check
$PY scripts/train.py --arm stock --clone runs/clone/stock/clone.pt --seed 1 --beta 20 --fixed-std \
    --critic-warmup 50 --lr0 1e-4 --tag v3n2                                    # --beta 5 for the beta5 policies
$PY scripts/evaluate.py --ckpt runs/ppo/<run>/model_1299.pt                     # the checks below
$PY scripts/export_policy.py runs/ppo/<run>/model_1299.pt policies/<name>       # numpy export
$PY tools/runner_equivalence.py --policy policies/<name> --mount swap_xy \
    --json policies/<name>/runner_equivalence.json                              # runner = simulator on the same inputs
$PY tools/action_stream.py --policy policies/<name> --out data/streams/<name>_0p10.npz   # for `runner replay`
```

The defaults of `distill.py` and `train.py` are not the recipe: pass the options shown. `scripts/evaluate_runs.sh`
runs the evaluation, export and videos for several runs. Eight runs in parallel take 2–3 h on an RTX 5080.

### Omnidirectional walking (flat ground)

Step by step (train, watch, deploy on the Pi, on Windows): [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

An omnidirectional policy follows a joystick command (vx, vy, wz) in any direction: travel speed 0.05–0.12 m/s at any
angle, yaw rate up to ±0.6 rad/s. The recipe is the same three steps:
- **Teacher:** the same firmware trot (`stanford_gait_step(vx, vy, wz)`) on a (vx, vy, wz) grid
  (`data/teacher_omni.npz`, `teacher.build_omni_table`), interpolated trilinearly.
- **Task** (`env_cfg.omni_options`):
  - the observed command is the whole (vx, vy, wz);
  - the action scale is 0.32 / 0.45 / 0.42 rad, because strafing swings the abduction about 16°;
  - the rewards that assumed +x travel follow the commanded direction instead: heading against the integrated yaw
    rate, drift across the commanded path, and swing progress along each foot's commanded velocity;
  - in training, commands are resampled every 4–6 s.
- **Clone:** 4096 environments, 150 epochs, perturbation 0.06.
- **Stops** (`--stops`, the recipe of `bam_t5ub06_stops_v1`): the training commands also stop and restart through the
  firmware's leg phases.

```bash
$PY tools/omni_teacher.py --arm stock                                        # the teacher, open loop, per command
$PY scripts/distill.py --arm stock --omni --envs 4096 --epochs 150 --noise 0.06 --out runs/clone/stock_omni
$PY scripts/train.py --arm stock --omni --clone runs/clone/stock_omni/clone.pt --seed 1 --beta 20 --fixed-std \
    --critic-warmup 50 --lr0 1e-4 --tag v3n2
$PY scripts/evaluate_omni.py --ckpt runs/ppo/<run>/model_1299.pt
```

`scripts/evaluate_omni.py` judges a policy on eight directions at 0.10 m/s, the speed range's corners, turning in
place and two curves (`gates.OMNI_EVAL_COMMANDS`). Each command must pass these gates:
- speed along the command within 20 %, and across it under 20 % of the command;
- yaw rate within 20 % of a commanded turn, or the heading held within 10°;
- clearance ≥ 8 mm, RMS pitch and roll < 3°, the firmware rhythm, no falls.

It also checks robustness on randomised plants under the training noise, twice the `n2` noise, and a random joystick
changing every 2 s.

### Rough ground and stairs

`scripts/train_terrain.py` trains on heightfield and box terrain (`mp3_walk_policy/terrain.py`,
`mp3_walk_policy/terrain_task.py`) with a curriculum: each environment moves up a difficulty level when it crosses
its tile and down when it fails. The policy stays blind; the critic also sees a height scan around the robot.

- **Warm start** (`--variant C`): the omnidirectional policy `bam_t5ub06_stops_v1`, its output layer rescaled to the
  terrain action scale (0.25 / 0.5 / 0.5 rad), anchored to it at first (β 5, fading to 0), 50 critic-only iterations.
- **Commands:** forward 0.05–0.12 m/s; the yaw rate comes from a heading loop (the same PI law the robot's runner
  uses, `mp3_walk_policy/heading.py`), so the policy learns to hold a heading.
- **Rewards** beyond the flat task's: forward progress, swing-apex foot clearance over the terrain under the foot,
  a stumble penalty (a foot pushed mostly sideways) and a thigh and shin contact penalty.
- **Randomisation:** foot softness and friction, masses, the policy-to-servo delay, the observation noise.
- **Terrain presets** (`--terrain`): `rough` (low-poly relief, relief on a raised plinth, slopes up to 20°, scattered
  boxes and low stairs of 5–30 mm), `stairs` (`rough` with jittered staircases at 50 % of the spawns: risers 5–60 mm
  with difficulty, ±30 % per step, treads 50–200 mm, up and down), `stairs_narrow` (`stairs` with narrow flights
  200 mm wide and drops at both sides). `--stair-exit` counts a staircase as crossed only when
  the robot walks off its top; `--tilt-limit 1.0` (rad) allows the pitch of steep flights before an episode ends as a
  fall.

The shipped policies come from one chain of six fine-tuning steps, each starting from the previous step's checkpoint.
`scripts/recipes/terrain_stairs.sh` runs it with the exact options of each step (also recorded in each policy's
`config.json` under `training_chain`):

| Step | Terrain | Iterations | Exported |
|---|---|---|---|
| 1 | `rough`, 10-step history | 1200 (from `bam_t5ub06_stops_v1`) | `bam_t5ub06_terrain_h10_v2` (iteration 700) |
| 2 | `stairs` | 1000 (from step 1, iteration 700) | |
| 3 | `stairs`, `--stair-exit` | 1000 (from step 2, iteration 1500) | |
| 4 | `stairs_narrow`, tilt limit 1.0, the robot's heading law | 1000 (from step 3, iteration 1900) | |
| 5 | as 4, narrow risers 20–40 mm, environments starting on every difficulty level | 1000 (from step 4, iteration 2899) | |
| 6 | as 5, narrow risers 25–45 mm | 1000 (from step 5, iteration 3898) | `bam_t5ub06_stairs_narrow_v3` (iteration 4500) |

```bash
sh scripts/recipes/terrain_stairs.sh 1      # step 1 only: the rough-ground policy
sh scripts/recipes/terrain_stairs.sh        # all six steps
$PY scripts/export_policy.py runs/recipes/<step 6 run>/model_4500.pt policies/my_stairs
```

The script uses `.venv/bin/python` (`PYTHON=` overrides) and writes to `runs/recipes/`. Run again, it skips finished
steps; delete an interrupted step's run folder before rerunning. Ctrl-C stops it. The recipe trains a comparable
policy, not the same weights: GPU training is not deterministic. **The defaults of
`train_terrain.py` are not the recipe**: use the script, or copy a step's options from it.

- **Timing (RTX 5080):** one run at 1024 environments takes about 2.4 s per iteration alone (the six steps about
  4 h); eight runs side by side take 14–19 s per iteration each and fill the GPU's 16 GB. A resumed run at iteration s for N iterations saves
  checkpoints every 50 iterations and ends at `model_(s+N-1)`.
- **Logging:** TensorBoard by default; `--logger wandb --wandb-entity <you>` logs to Weights & Biases (project
  `mp3_walk_policy`).

## Evaluation

### Flat ground

`scripts/evaluate.py` (`mp3_walk_policy/gates.py`) tests a policy in simulation at 0.05 / 0.10 / 0.15 / 0.20 m/s:
- **Clean** (nominal model, no noise): no falls, speed within 20 % of the command, foot clearance ≥ 8 mm, RMS pitch
  < 3°, and a gait frequency within 0.3 Hz of the firmware trot's 2.78 Hz.
- **Robust** (0.10 m/s, 256 randomised models with the training noise): no falls, the same gait frequency; with `n2`,
  at most 1 % falls under twice the noise.
- **Open loop:** the clean 0.10 m/s actions replayed blind on the randomised models keep ≥ 75 % of the closed loop's
  distance and ≥ 0.06 m/s.

### Rough ground and stairs

`tools/terrain_eval.py` scores a policy on randomised robots (friction, masses, policy-to-servo delay, observation noise) and
writes per-trial outcomes to JSON, so policies compare trial by trial on the same seeds. Pass the options in full:
its defaults are a generic course.

```bash
# rough ground: the rough test tile between walls; success = 0.65 m without a fall
$PY tools/terrain_eval.py --policy policies/bam_t5ub06_terrain_h10_v2 --heading-loop --terrain square_walls \
    --envs 256 --goal 0.65 --seconds 20 --stop-at-goal --seed 11 --json runs/eval/square.json

# the test staircase, 200 mm wide with drops at the sides; the start jittered ±1 cm and ±2°
$PY tools/terrain_eval.py --policy policies/bam_t5ub06_stairs_narrow_v3 --terrain stairs_built \
    --layout data/stairs/printed_r25_t50_x4.json --width 0.20 --landing 0.25 --start-noise 0.01 2 \
    --envs 256 --seconds 30 --heading-loop --stop-at-goal --foot-solref 0.01 1 --tilt-limit 1.0 \
    --seed 11 --json runs/eval/stairs.json

# a grid of staircases: risers 10–70 mm × treads 50–200 mm
$PY tools/terrain_eval.py --policy policies/bam_t5ub06_stairs_narrow_v3 --terrain stairs_climb --envs 1040 \
    --seconds 20 --vx 0.10 --heading-loop --stop-at-goal --foot-solref 0.01 1 --tilt-limit 1.0 --seed 11 \
    --json runs/eval/stair_grid.json
$PY tools/stair_capability.py --out runs/eval/capability name=runs/eval/stair_grid.json   # the success border
```

- On a narrow staircase read `success_width` (every stance foot stayed on the treads), not `success`.
- `--layout` describes any staircase as `{"steps_mm": [[riser, tread], ...]}`; `data/stairs/` holds the test staircase
  and a 3-step version on a 15 mm start plate.
- `--heading-loop` closes the runner's heading loop around the policy (cap `--heading-wz-max`, 0.4 rad/s, as trained).
- `--yaw-kick T0:T1:TORQUE:DUR` adds yaw torque pulses on the body, a stand-in for a foot catching a stair edge.
- `--video x.mp4 --envs 1 --clean` renders one trial (`MUJOCO_GL=egl`); `tools/edge_metrics.py`,
  `tools/joint_stats.py` and `tools/tracking_stats.py` report foot strikes at an edge, per-joint excursions and the
  simulated servo currents.
- Look at the terrain: `tools/terrain_preview.py --out x.png [--preset rough]`, `tools/terrain_gallery.py`.

## Export

`scripts/export_policy.py <run>/model_<k>.pt policies/<name>` writes `policy.npz` (the network as numpy arrays, its
metadata, and fixture inputs and outputs checked against torch to 1e-5) and copies the run's `config.json`.
`runner check` (below) runs on any machine with numpy, e.g. in the runner's test environment from "Install":

```bash
env -u PYTHONPATH .venv-pi/bin/python -B -m mp3_walk_policy.runner check --policy policies/<name>
```

## Robot interface

`mp3_walk_policy/contract.py` (with `mp3_walk_policy/data/sku2_contract.json`) defines what the robot and the
simulator must agree on:
- **Joints:** servo ids 1–12 (FR, FL, RR, RL × abduction, hip, knee). The policy works in radians from the standing
  pose: sign × radians(servo angle), the angle minipupperesp's `servo_write` takes without the calibration offset; the
  knee angle is the shin angle. The simulation names the abduction, hip and knee joints `*_hip`, `*_thigh`, `*_calf`.
- **Observation (45 per step):** gyro (body frame, rad/s, bias removed), gravity direction (−acc/|acc|, (0, 0, −1)
  upright), command (vx, vy, wz), joint positions, joint velocities (median of 3, then a 4-sample slope), the previous
  applied target (in action units). Policies with history stack the last 10 steps (450 inputs).
- **Action (12), 50 Hz:** clipped to [−1, 1], scaled per joint (abduction / hip / knee: forward 0.10 / 0.40 / 0.42 rad,
  omni 0.32 / 0.45 / 0.42, rough ground and stairs 0.25 / 0.5 / 0.5), limited to ±20° (abduction) and ±60° (others)
  and to 20° per step, then converted to the firmware's servo units.
- **Delay:** each actuator model's own latency plus a policy-to-servo delay drawn from 8–19 ms per episode (10 ms
  measured on SKU2).
- **IMU mounting:** `swap_xy` on SKU2 (`mp3_walk_policy/data/imu_mount.json`). For another robot,
  `tools/tilt_mount.py` describes the tilt test that measures it.
- **Calibration:** `offsets_deg`, `stand_dd` and `stand_ticks` in the contract are the reference robot's servo
  calibration (`python3 -m mp3link offsets --json` reads a robot's). The policies do not depend on them; the runner
  warns when the robot's offsets differ from the contract's. Earlier calibrations are listed under
  `calibrations_superseded`, so policies exported with them still pass `runner check`.

## On the robot

The ESP32 must run mp3-link's firmware. The Pi needs numpy, pyserial and `mp3link` (mp3-link's `pi/` package, which
`tools/deploy_pi.sh` copies to `~/mp3link_calib`). After every power-up, write the servo configuration the policies
were trained for (mode 1, kp 80, kd 400, kff 0.001, max_pwm 0.99; RAM only): `walk` does it, or
`python3 -m mp3link config` with those values.

```bash
PI_HOST=<pi> tools/deploy_pi.sh policies/<name> [...] --run        # copies the runner, tools and policies to ~/walk
# on the Pi, from ~/walk with PYTHONPATH=~/walk:~/mp3link_calib:
python3 -B -m mp3_walk_policy.runner check --policy policies/<name>
python3 -B -m mp3_walk_policy.runner run --policy policies/<name> --vx 0.10 --seconds 10 [--dry] --log logs/<name>
python3 -B -m mp3_walk_policy.runner run --policy policies/bam_t5ub06_stairs_narrow_v3 --heading-loop --vx 0.10 \
    --seconds 20 --tilt-cut 0 --log logs/stairs                    # stairs: the heading loop on, the tilt cut off
python3 -B -m mp3_walk_policy.runner replay --policy policies/<name> --actions data/streams/<name>_0p10.npz \
    --seconds 5 --hold-before 1 --log logs/<name>                  # the stream open loop, robot held in the air
python3 -B -m mp3_walk_policy.runner gait --vx 0.10 --heading-loop --seconds 20   # the firmware's own trot, no policy
```

`--dry` runs everything with the servos idle. A run holds the standing pose 10 s before and 5 s after.

- **By hand:** `ssh -t <pi> ~/walk/tools/walk <policy>` writes the servo configuration and walks the policy (`--help`
  for the options). Enter ramps back to the standing pose, holds 3 s and idles the servos (the body drops about
  7 cm); Ctrl-C idles them at once.
- **Logged runs ("takes") from a workstation:** `PI_HOST=<pi> tools/pi_take.sh <name> runner run ...` runs the runner detached
  on the Pi, so a dropped connection does not stop it, and reports when it ends; `--soft-stop <name>` ends it
  gracefully and `--stop <name>` at once (options in the script header). Logs go to the Pi's `~/walk/logs/<YYYYMMDD>/`.
- **Delay:** `python3 tools/chain_delay.py logs/<YYYYMMDD>/<name>` measures the policy-to-servo delay of a run's log.
- **Safety:** the runner refuses to start without a valid IMU, the expected servo configuration or a measured IMU
  mounting, over a latched fault, or above 45 °C. It cuts the servos on a tilt above 35° (`--tilt-cut`; edge strikes
  on stairs read as large tilts, so stair runs turn it off and a hand must be ready to catch the robot). It returns
  to the standing pose and stops on an IMU dropout, a non-finite action, lost or late frames, a servo at 55 °C, or
  Enter. After a current-guard trip, `--clear` re-arms.

## Repository layout

| Path | Contents |
|---|---|
| `mp3_walk_policy/` | the robot model and contract, tasks (flat, omni, terrain), actuator models, teacher, export, the Pi runner |
| `scripts/` | cloning, training, evaluation, export, rendering, live viewers; `recipes/` |
| `tools/` | terrain evaluation and metrics, the mujoco-warp patch, Pi deployment and logged runs, robot-log tools |
| `policies/` | the exported policies |
| `data/` | action streams for `runner replay`, staircase layouts |
| `tests/` | the simulation and contract tests, and `tests/robot` (the runner against a fake ESP32) |
| `requirements/` | the pinned GPU environment |

## Licence

Apache License 2.0 (`LICENSE`); credits in `NOTICE` (BAM by Rhoban, the Stanford Pupper gait, mjlab, MuJoCo).
