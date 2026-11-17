# The pipeline

`train.ps1` runs five stages. Each one is SKU2's recipe as it was released, with the robot switched to SKU1 v2 by
environment variables (`scripts/common.ps1`):

| Variable | Value | Read by | Effect |
|---|---|---|---|
| `MP3_ROBOT` | `sku1v2` | `walk/mp3_walk_policy/paths.py` | SKU1's body (`third_party/minipupperesp/sim/model/pupper.xml`) instead of SKU2's |
| `MP3_MASSES` | `sku1v2_masses_405g.json` | the same | every walk task on the robot as weighed: 405 g, 177 / 223 g front / rear |
| `MP4R_ROBOT` | `sku1v2` | `recovery/src/mp4_recovery/paths.py` | the recovery repository on SKU1 v2: model, masses, contract, limits, motions |
| `BAM_DIR`, `MINIPUPPERESP_DIR` | `third_party/...` | `walk/` | the servo model and the firmware gait sources |
| `SKU1V2_PY` | `.venv\Scripts\python.exe` | `walk/tools/*.ps1` | the one training environment |

Nothing else differs from SKU2's training: the same servo model (`bam_t5ub06`, the MD01 fitted with BAM), the same
sensing-to-command delay (8-19 ms), noise, domain randomisation, rewards, network sizes, iterations, seeds and
checkpoints.

Every stage can be run again. Finished work is skipped. An interrupted training step's folder is renamed
`*_interrupted` and the step starts over.

## 1. Walk

`walk/tools/train_sku1v2_walk.ps1` follows the recipe of SKU2's `bam_t5ub06_stops_v1` (and SKU1's `sku1_368_stops_s2`).
`walk/docs/SKU1V2_WALK.md` has the details.

| Step | What | Options |
|---|---|---|
| Teacher | minipupperesp's own trot and inverse kinematics (`main/stanford_gait.c`, `stanford_kinematics.c`), on take-like command schedules with stops | the prebuilt `walk/mp3_walk_policy/data/libgait_win64_<hash>.dll` on Windows, so no compiler is needed |
| Clone | supervised copy of the teacher, 4096 robots | `distill.py --arm bam_t5ub06 --omni --stops --epochs 150 --noise 0.06` |
| PPO | per seed, anchored to the clone | `train.py --arm bam_t5ub06 --stops --beta 20 --fixed-std --critic-warmup 50 --lr0 1e-4`, 1300 iterations, 512 robots |
| Gates | eight directions, turns, curves, stops, randomised robots, the joystick | `evaluate_omni.py` → `eval_omni_model_1299.json` |
| Export | | `export_policy.py` → `walk/policies/sku1v2_405_walk_s<seed>` |

**The choice:** of the seeds trained (`-WalkSeeds`, default 1 and 2), the one that passes every gate with the fewest
joystick falls (`-WalkSeed` overrides). PPO's result depends on the seed, which is why the recipe trains two.

Commands are omnidirectional up to 0.12 m/s forward, 0.12 m/s sideways and 0.6 rad/s turning.

## 2. Stairs

`walk/tools/train_sku1v2_stairs.ps1` runs SKU2's chain (`walk/scripts/recipes/terrain_stairs.sh`), the steps behind
`bam_t5ub06_terrain_h10_v2` and `bam_t5ub06_stairs_narrow_v3`. Each step fine-tunes the step before. All steps use
1024 robots, a 10-step observation history and a yaw command from a heading loop.

| Step | Seed | Terrain | Iterations | Starts from |
|---|---|---|---|---|
| 1 `rough` | 4 | rough ground (relief, slopes, boxes, steps up to 30 mm) | 1200 | the walk (`-StairsInit own`) or SKU2's walk (`sku2`) |
| 2 `stairs` | 5 | stairs | 1000 | step 1, iteration 700 |
| 3 `stairs_exit` | 5 | stairs, counted only when it walks off the top | 1000 | step 2, iteration 1500 |
| 4 `stairs_narrow` | 5 | narrow flights 200 mm wide (drops at both sides) | 1000 | step 3, iteration 1900 |
| 5 `stairs_narrow_r40` | 6 | narrow, risers 20-40 mm | 1000 | step 4, iteration 2899 |
| 6 `stairs_narrow_r45` | 6 | narrow, risers 25-45 mm | 1000 | step 5, iteration 3898 |

Shared options: `--variant C --heading loop --foot-softness 0.005 0.02 --friction 0.4 1.3 --vx-max 0.12 --clearance apex
--clearance-weight -2 --stumble -0.5 --progress 2`. The narrow steps add `--stair-exit --tilt-limit 1.0 --heading-law pi
--heading-wz 0.4`.

**`-StairsInit own`** (the default) starts step 1 from this robot's own walk, as SKU2's chain started from SKU2's own
walk. The anchor to the starting walk fades out during step 1, so the gait is relearnt on rough ground either way.
`sku2` starts from SKU2's `bam_t5ub06_stops_v1`, as the first SKU1 v2 stairs runs did.

**Tests:** the two printed test staircases (`walk/data/stairs/`): 4 × 25 mm risers on 50 mm treads, and 3 steps of
20 / 25 / 25 mm. Both are 200 mm wide with a 250 mm landing. 2 seeds × 256 robots, 30 s, heading loop, stop at the goal.
Success means all four feet on the landing, no fall, and every stance foot on the treads. The tests run on exports at
iterations 4500 (as SKU2's) and 4897, with SKU2's policy unchanged as the reference.

**The choice:** the export with the higher success on the 4 × 25 mm staircase (`-StairsIter` overrides).

**Further steps** (not SKU2's recipe; they were made for SKU1 v2 after robot tests, and are run by hand from `walk/`):

| Script | What |
|---|---|
| `train_sku1v2_stairs_nostall.ps1` | 1500 more iterations where standing still on the stairs counts as a failure (the step-6 policy froze halfway on the real robot) |
| `train_sku1v2_stairs_robust.ps1` | feet lifted higher (32 mm) and robots unlike the model (shins ±2 mm, feet ×0.7-1.4, servos ±10 %) |
| `train_sku1v2_stairs_v3.ps1` | stalls and turning away count as failures, no pull to a level body, risers 15-30 mm |

Each script's header has its parameters. Pass `-Out runs\sku1v2_405_stairs` and the stage's policy names to continue
this pipeline's chain.

## 3. Walker

SKU2's walker is not trained in the recovery repository. It is a finished omni walk with the joint bounds and a stance
12° wider at the abductions written into its metadata (`recovery/scripts/recipes/walker.sh`). Here:

1. `tools/walk_to_walker.py`: the walk from stage 1 into the recovery repository's format (`policies/sku1v2_walker_base`).
   The network, action scale and fixtures are copied unchanged. The metadata names the recovery contract. The script
   refuses to convert if the walk's contract and the recovery contract differ in signs, stand or calibration.
2. `export_policy --from-policy ... --limits --stand-offset-abd 12` → `policies/sku1v2_walker` (and `--limits` alone →
   `sku1v2_walker_stance0`, for comparison).
3. The recovery runner's checks (`deploy.runner check`).
4. **Shove test** (`eval.push_eval`): pushes of 0.5-1.5 m/s every 2-4 s for 20 s, standing and walking. The share of
   robots that fall is compared with and without the wider stance.

## 4. Get-up

SKU2's recipe (`recovery/scripts/recipes/getup.ps1`, the same as `getup.sh`). `recovery/docs/TRAINING.md` explains each
step.

| Step | Iterations | What changes |
|---|---|---|
| Motions | - | SKU2's six searched open-loop get-ups replayed on SKU1 v2. A motion that stands fewer than 75 % of the robots up is searched again on SKU1 v2 from itself (`tools/sku1v2_motions.py`, 0-3 h). `-ReuseMotions` starts from the robot's current motions instead |
| Clone | 300 epochs, 4 rounds | supervised copy of the motions into one network (512-256-128) |
| `wide` | 1500 | PPO, hip and knee targets allowed 60° past their track ends |
| `limits` | 1000 | the real limits; action-rate and joint-velocity costs |
| `filter` | 253 | target filter ramped from α 0.5 to 0.75 |
| `smooth1`, `smooth2` | 400, 200 | filter fixed at 0.75; smoothness loss, temporal 1, spatial 20 |
| `smooth3`, `smooth4` | 600, 600 | smoothness loss, temporal 2, spatial 40 |

All steps use 2048 robots (3.2 GB of GPU memory) and 8 s episodes. Starts are crouch, partial roll, side, nose or tail
down, back and random, with 20 % of them mid-fall.

**Tests**, per seed:

- `eval.getup_eval`: 256 robots from each of seven starts (left, right, nose, back, mid-fall, crouch, random), 8 s,
  giving success and the time to a held stand.
- `eval.thresholds`: the recovery repository's pass / fail limits on those numbers.
- `eval.sequence`: walk, shoves of 0.5-1.5 m/s, fall, get-up and walk again, for 256 robots over 60 s. It reports the
  topple rate, the share walking again within 8 s, and robots down for over 15 s.

**The choice:** of the seeds trained (`-GetupSeeds`, default 41), the one that passes the thresholds with the highest
mean success (`-GetupSeed` overrides).

## 5. Export

`tools/export_mpxp.py` (the same file as `mpv2onlyRL/tools/policy/export_mpxp.py`) packs each policy with its contract
(servo signs, calibration, stand, action scale), the IMU mounting and 32 fixtures. The fixtures let the robot check its
own forward pass at load. The recovery policies also carry their decoder block (target filter, stance offset, joint
bounds, the runner's rules).

| File | From | Contract |
|---|---|---|
| `out/default.mpxp` | `walk/policies/sku1v2_405_walk_s<seed>` | `walk/mp3_walk_policy/data/sku2_contract.json` |
| `out/sku1v2_stairs.mpxp` | `walk/policies/sku1v2_405_stairs_narrow_<iter>` | the same |
| `out/sku1v2_recovery.mpxp` | `recovery/policies/sku1v2_walker` | `recovery/src/mp4_recovery/data/sku1v2_contract.json` |
| `out/getup_sku1.mpxp` | `recovery/policies/sku1v2_getup_s<seed>` | the same |

Both contracts hold SKU2's servo signs and stand with this robot's calibration (offsets −9, 7, −15, 6, −34, −33,
−3.61, 25, 17, −6, 8, 4). `check.ps1` exports the robot's current walker and get-up again and compares them byte for
byte with `results/mpxp/`.

`out/MANIFEST.json` records each file's source, name and sha256, along with the choices. `out/REPORT.md` holds every
stage's tests (`tools/report.py`).

## Watching a run

`watch.ps1 -Stage walk|stairs|getup` opens TensorBoard on http://localhost:6006.

| Stage | Look at | Healthy |
|---|---|---|
| walk | `Train/mean_reward`, `Episode_Termination/*` | reward rises; falls drop after the first iterations |
| stairs | `Curriculum/terrain_levels` | rises within every step |
| stairs | `Episode_Termination/tilt`, `/body_contact` | fall over each step |
| get-up | `Episode_Reward/success` | climbs in `wide` and stays up in the steps after |

## Less GPU memory

`-WalkEnvs 256 -StairsEnvs 512 -GetupEnvs 1024` halves each stage's robots. Training then takes somewhat longer and the
results may differ slightly from the recipe's. Don't train two stages on the same GPU at once.
