# SKU1 v2 walks

The flat omnidirectional walk of `policies/sku1_368_stops_s2` (SKU1 at 368 g, trained in omniRL, seed 2) and of SKU2's
`policies/bam_t5ub06_stops_v1`, trained on SKU1 v2: SKU1's body (minipupperesp's `sim/model/pupper.xml`), 390 g,
195 / 195 g on the front and rear feet (`MP3_ROBOT=sku1v2`, `data/sku1v2_masses.json`, as in `SKU1V2_STAIRS.md`).

| | |
|---|---|
| Teacher | minipupperesp's own trot and IK (`firmware_gait.py`: `main/stanford_gait.c`, `main/stanford_kinematics.c`), on take-like schedules with stops |
| Clone | `scripts/distill.py --arm bam_t5ub06 --omni --stops --envs 4096 --epochs 150 --noise 0.06` |
| PPO | `scripts/train.py --arm bam_t5ub06 --stops --clone <clone> --seed <k> --beta 20 --fixed-std --critic-warmup 50 --lr0 1e-4` (1300 iterations, 512 robots) |
| Gates | `scripts/evaluate_omni.py --ckpt <run>/model_1299.pt` |
| Export | `scripts/export_policy.py <run>/model_1299.pt policies/sku1v2_stops_s<k>` (meta `robot: sku1v2`) |

`tools/train_sku1v2_walk.ps1` runs all of it for seeds 1 and 2; `tools/sku1v2_walk_check.py` is the few-minute check
before. Two seeds because PPO's outcome varies with the seed: SKU1's seed 2 was the better of its two.

The teacher's gait library is compiled from minipupperesp's C sources. On Windows,
`mp3_walk_policy/data/libgait_win64_<hash>.dll` (mingw-w64 gcc, `-O2 -ffp-contract=off`, the same flags as the Linux
and Pi builds) is used when its hash matches the sources (read with LF line endings), so no compiler is needed;
otherwise MSVC Build Tools build it.

On the robot (mpv2 RL firmware): `tools/policy/export_mpxp.py policies/sku1v2_stops_s<k> --contract <your SKU1
contract> --mount ... --out main/rl/policies/default.mpxp`. The firmware knows the robots sku1 and sku2; the export
writes an `sku1v2` policy as `sku1`.
