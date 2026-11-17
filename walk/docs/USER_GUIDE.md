# User guide: train, watch and deploy a walk policy

This guide covers the omni walk policy on Windows. Rough-ground and stair training are covered in the README (Linux). The forward policy uses the same steps
without `--omni`. The commands are PowerShell, run from the repository root. Linux equivalents are in the README.

```powershell
cd path\to\mp3_walk_policy
$env:WANDB_MODE = "disabled"
$PY = "..\bam\.venv-mjlab\Scripts\python.exe"
```

A policy is trained for one actuator model ("arm"): `stock` (mpx-sim's servo, faster to train) or `bam_t5ub06` (the
MD01 model fitted to the real joints). Every command below takes `--arm`.

## 1. Set up (once)

The training environment is `..\bam\.venv-mjlab`. If it is missing:

```powershell
cd ..\bam
uv venv --python 3.12 .venv-mjlab
uv pip install --python .venv-mjlab\Scripts\python.exe --torch-backend=auto -e ".[mjlab]" pytest imageio imageio-ffmpeg
cd ..\mp3_walk_policy
& $PY -B -m pytest -q tests --ignore=tests/robot          # should pass (about a minute)
```

`--torch-backend=auto` matters: PyPI's Windows torch has no CUDA. The firmware gait (the teacher) compiles itself with
MSVC the first time it is needed. The first environment build also compiles Warp kernels (about a minute, cached).

## 2. Train

Training has three steps. Each one's output is the next one's input.

### 2a. The clone (about 8 min)

Behaviour cloning of the firmware trot. It ends with a gate: the clone must match the teacher on 16 commands, on the
nominal plant and on randomised plants.

```powershell
& $PY -B scripts\distill.py --arm stock --omni --envs 4096 --epochs 150 --noise 0.06 --out runs\clone\stock_omni
```

The last lines must read `nominal PASS` and `randomised PASS`. The clone is `runs\clone\stock_omni\clone.pt`, and the
gate's numbers are in `gate.json` next to it.

**With stops** (the recommended omni recipe; it learns to stop and restart like the recorded takes): add `--stops`.
The data are random take-like schedules (walks in any direction, turns in place up to 1.2 rad/s, stops), labelled by the
firmware playing each one through its own leg phases. The gate adds the recorded takes. About 12 min for BAM.

```powershell
& $PY -B scripts\distill.py --arm bam_t5ub06 --stops --envs 4096 --epochs 150 --noise 0.06 --out runs\clone\bam_t5ub06_stops
```

Then train with `--stops` in place of `--omni` (below). `evaluate_omni.py` plays the takes for every policy. A stop
passes when every foot is down (within 3 mm) and the body is still. `tools\play_takes.py --ckpt <run> --n 8 --randomise`
plays only the takes.

### 2b. Anchored PPO (about 1 h for stock, longer for BAM)

```powershell
& $PY -B scripts\train.py --arm stock --omni --clone runs\clone\stock_omni\clone.pt --seed 1 `
    --beta 20 --fixed-std --critic-warmup 50 --lr0 1e-4 --tag v3n2
```

The run goes to `runs\ppo\<date>_<time>_stock_omni_s1_v3n2\`, with a checkpoint `model_<N>.pt` every 25 iterations up
to `model_1299.pt`. The console prints each iteration and an ETA. To keep the terminal free, start it in the
background with its output in a log:

```powershell
Start-Process -NoNewWindow -FilePath $PY -RedirectStandardOutput runs\ppo\stock_omni_s1.log `
    -RedirectStandardError runs\ppo\stock_omni_s1.err -ArgumentList "-B","scripts\train.py","--arm","stock","--omni",`
    "--clone","runs\clone\stock_omni\clone.pt","--seed","1","--beta","20","--fixed-std","--critic-warmup","50",`
    "--lr0","1e-4","--tag","v3n2"
Get-Content runs\ppo\stock_omni_s1.log -Wait | Select-String "Learning iteration|ETA"     # follow progress
```

Two runs at once share the 8 GB GPU and both slow down. Stock alone runs at about 2.9 s per iteration.

### 2c. Evaluate (about 10 min)

```powershell
& $PY -B scripts\evaluate_omni.py --ckpt runs\ppo\<run>\model_1299.pt
```

It prints one line per command with the speed along and across the command, the yaw rate, the heading drift, the
rhythm, clearance, pitch and roll, and the gates it failed. Then it prints the robustness checks (randomised plants under noise,
twice the noise, and a random joystick) and `ALL PASS True/False`. The results are saved as
`runs\ppo\<run>\eval_omni_model_1299.json`.

## 3. Watch training

### Curves (TensorBoard)

```powershell
..\bam\.venv-mjlab\Scripts\tensorboard.exe --logdir runs\ppo
```

Open <http://localhost:6006>. The useful curves:

| Curve | What it tells you |
|---|---|
| `Train/mean_reward` | Overall progress; should rise and level off |
| `Train/mean_episode_length` | 500 = no falls (10 s episodes); dips mean falls |
| `Episode_Termination/tilt`, `body_contact` | Which way it falls |
| `Episode_Reward/track_lin_vel`, `track_ang_vel` | How well it follows the joystick |
| `Episode_Reward/yaw`, `lateral_position` | Heading drift and drift off the commanded path (less negative is better) |
| `Metrics/twist/error_vel_xy`, `error_vel_yaw` | Command tracking error |
| `Loss/anchor_mse` | How far the policy has moved from the clone |
| `Curriculum/noise_ramp` | The observation-noise ramp (0 to 1 over iterations 150–450) |

### The robot, live

`scripts\watch.py` plays any checkpoint while training continues. Given a run directory, it picks the newest
checkpoint.

```powershell
& $PY -B scripts\watch.py --ckpt runs\ppo\<run>                          # MuJoCo window, tours 16 commands
& $PY -B scripts\watch.py --ckpt runs\ppo\<run> --cmd 0,0.10,0           # one command: strafe left 0.10 m/s
& $PY -B scripts\watch.py --ckpt runs\ppo\<run> --viewer viser           # browser, with joystick sliders
& $PY -B scripts\watch.py --ckpt runs\clone\stock_omni\clone.pt --arm stock --omni     # the clone
```

- **Window** (default): on the tour, each command is held 4 s (`--hold`), and the console prints the command as it
  changes.
- **Browser** (`--viewer viser`): open <http://localhost:8080>. Under "Twist", tick "Enable" and drag the vx / vy / wz
  sliders to drive the robot like a joystick.
- `--noisy` and `--randomise` show it under the training noise and a randomised plant.

The command convention is vx forward, vy left (m/s), and wz turning left (rad/s). Training covers speeds of 0.05–0.12 m/s in
any direction and |wz| ≤ 0.6.

### A video

```powershell
& $PY -B scripts\export_policy.py runs\ppo\<run>\model_1299.pt policies\stock_omni_s1
& $PY -B scripts\render.py --ckpt policies\stock_omni_s1 --vx 0.07 --vy 0.07 --wz 0 --seconds 10 --out runs\video\diag.mp4
```

## 4. Deploy on the Raspberry Pi

### What the robot needs

- The ESP32 runs mp3-link's firmware, and the legs are calibrated (mp3-link README, "Calibrating the legs").
- Nothing gets installed on the Pi. The Pi needs `python3` with numpy and pyserial, and two folders:
  - `~/mp3link_calib`: mp3-link's `pi/` package (the `mp3link` link to the ESP32);
  - `~/walk`: this repository's runner, contract, IMU mounting and policies.
- `mp3_walk_policy\data\imu_mount.json` is the IMU mounting measured on our SKU2. For another robot, measure it
  first (README, "The contract", IMU mounting).

### Export the policy

Export only a policy that passed `evaluate_omni.py`:

```powershell
& $PY -B scripts\export_policy.py runs\ppo\<run>\model_1299.pt policies\stock_omni_s1
```

`policies\stock_omni_s1\` gets `policy.npz` (the weights, the action scale and `"omni": true`), the run's
`config.json` and its evaluations.

### Copy to the Pi

`tools/deploy_pi.sh` needs `rsync`. On Linux or macOS: `PI_HOST=pi@<pi> tools/deploy_pi.sh policies/stock_omni_s1 --run`.
This Windows machine has no `rsync`, so copy with `scp` (Windows' built-in OpenSSH):

```powershell
$PI  = "pi@<pi-address>"
$POL = "stock_omni_s1"
ssh $PI "mkdir -p ~/walk/mp3_walk_policy/data ~/walk/policies/$POL ~/walk/tools ~/walk/data ~/walk/logs ~/mp3link_calib"
scp mp3_walk_policy\__init__.py mp3_walk_policy\contract.py mp3_walk_policy\export.py mp3_walk_policy\runner.py "${PI}:walk/mp3_walk_policy/"
scp mp3_walk_policy\data\sku2_contract.json mp3_walk_policy\data\imu_mount.json "${PI}:walk/mp3_walk_policy/data/"
scp policies\$POL\policy.npz "${PI}:walk/policies/$POL/"
scp tools\tilt_mount.py "${PI}:walk/tools/"
scp -r data\streams "${PI}:walk/data/"
scp -r ..\mp3-link\pi\mp3link "${PI}:mp3link_calib/"           # once, or when mp3-link changes
```

### Run it (on the Pi, over `ssh -t $PI`)

Every session, write the identified servo configuration first:

```bash
cd ~/mp3link_calib && python3 -m mp3link config
```

Then go step by step. `R` is shorthand for the runner command:

```bash
cd ~/walk
R="env PYTHONPATH=$HOME/walk:$HOME/mp3link_calib python3 -B -m mp3_walk_policy.runner"

$R check --policy policies/stock_omni_s1
#   fixtures ok, contract ok, the inference time

$R run --policy policies/stock_omni_s1 --vx 0 --vy 0.08 --wz 0 --seconds 5 --dry --log logs/dry
#   never arms: real STATE, IMU and timing, the servos stay idle

$R run --policy policies/stock_omni_s1 --vx 0.08 --seconds 5 --hold-before 1 --log logs/air
#   robot on a stand, feet in the air

$R run --policy policies/stock_omni_s1 --vx 0 --vy 0.08 --wz 0 --seconds 10 --log logs/floor_strafe
#   on the floor: holds the stand 10 s, walks 10 s, holds 5 s
```

- Pass the command with `--vx`, `--vy` (m/s, + = left) and `--wz` (rad/s, + = turn left). It is held for the whole run.
  The runner has no live joystick input yet.
- The runner refuses `--vy` / `--wz` for a forward-only policy, and it refuses to arm without `IMU ok`, the servo
  configuration, or the measured IMU mounting.
- **Stopping:**
  - Enter ramps to the stand and disarms.
  - Ctrl-C disarms at once.
  - It also disarms by itself on a tilt above 35°, an IMU dropout, a late step or 55 °C.
- **Logs:** `logs/<take>.npz` (every frame), `.policy.npz` (each step's observation and action) and `.stand.json`.
  Copy them back with `scp "${PI}:walk/logs/*" logs\`.

Start slow (0.05–0.08 m/s) and on a clear floor. Try one direction at a time before diagonals and turns.
