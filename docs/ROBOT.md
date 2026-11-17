# The robot: SKU1 v2 against SKU2

SKU1 v2 is the Mini Pupper 4 with SKU1's trunk and SKU2's legs. It shares SKU2's MD01 three-motor servo modules, IMU,
servo mapping and ESP32-S3 controller. Training therefore keeps SKU2's servo model, observations, actions and options,
and changes only the body.

| | SKU2 | SKU1 v2 (this repository) | Where |
|---|---|---|---|
| Body model | mpx-sim `sku_2/pupper.xml` | minipupperesp `sim/model/pupper.xml` (from `sku_1_urdf_fixed`, sha256 `411de14dfe18`) | `third_party/minipupperesp/sim/model/`, `recovery/src/mp4_recovery/robot/assets/sku_1/` |
| Mass | 478 g, 197 / 258 g on the front / rear feet | **405 g, 177 / 223 g** (55.75 % on the rear feet), weighed with shoes and battery | `sku1v2_masses_405g.json` (the same file in `walk/` and `recovery/`) |
| Parts | a module 66.82 g, a leg 6.5 g | the same parts; the base takes the rest, its centre of mass solved for the rear share | `walk/tools/robot_masses.py` |
| Start height | 72.5 mm | 70.1 mm (the hips sit 2.4 mm higher on the base origin); hip axes 70 mm above the ground at the stand | |
| Servo model | `bam_t5ub06` (MD01 fitted with BAM) | the same | `third_party/bam`, `walk/mp3_walk_policy/data/bam_t5ub06_id_*.json` |
| Sensing → command | 8-19 ms | the same | |
| Contract (signs, stand) | SKU2's | SKU2's | |
| Calibration | SKU2's robot's | this robot's: offsets −9, 7, −15, 6, −34, −33, −3.61, 25, 17, −6, 8, 4 | `walk/.../sku2_contract.json`, `recovery/.../sku1v2_contract.json` |
| Joint limits | measured on SKU2 | **estimated**: SKU2's measurement, with this calibration's track ends | `recovery/src/mp4_recovery/data/sku1v2_joint_limits.json` |
| Hip envelope | measured on SKU2 | **estimated**: SKU2's | `recovery/src/mp4_recovery/data/sku1v2_hip_envelope.json` |

`check.ps1` builds both models and prints the result: "built 405.0 g, rear feet 55.75 %".

## Joint limits

The abduction stops and the hip envelope come from the chassis beside the hips. SKU1's and SKU2's models have base
boxes of the same width (40 mm half-width) reaching about as far past the hip axes (23.5 and 22.7 mm), and the hip
brackets and legs are the same parts. SKU2's measured ranges are therefore the best estimate until this robot's are
measured. The track ends (where each servo's sensor ends) are this robot's, from its calibration.

To measure them, move each limp leg by hand to each stop and record the positions (`recovery/docs/ROBOT.md`, "Check the
joint limits"). Put the measured file in place of `sku1v2_joint_limits.json`, update
`limits.LIMITS_SHA256["sku1v2"]` in `recovery/src/mp4_recovery/limits.py`, and train the get-up again
(`train.ps1 -Stage walker,getup,export`).

## Calibration

The policies are trained with the calibration above. On the robot, the firmware's **Use my calibration** (on whenever
the policy's robot matches `APP_POLICY_ROBOT`) runs them on the robot's own offsets. The joint bounds near the track
ends move with those offsets, and the firmware refuses a calibration that puts a target within 15° of a track end.

The get-up needs the most room. From the robot's tests, it needs about:

| Joint | Room needed (rel. to the stand, deg) |
|---|---|
| FL thigh | −30 … +169 |
| FL calf | up to +140 |
| FR thigh | −100 … +128 |
| FR calf | −92 … +116 |
| RR thigh | up to +110 |
| RR calf | from −118 |
| RL thigh | from −142 |

If the get-up works in simulation but not on the robot, check the calibration first. The app's RL page saves the last
get-up's per-servo log (`getup_last.csv`).

A new calibration does not need new training when the room above is there. If you calibrate very differently, change
the offsets in both contracts (`recovery/tools/make_contract.py` writes the recovery one) and train again, so the
simulated robot matches.

## Another weight

Make a mass file with `walk/tools/robot_masses.py --robot sku1v2 --total-g <g> --front-g <g> --rear-g <g>` and copy it
to `recovery/src/mp4_recovery/data/`. Then point `$Masses` in `scripts/common.ps1` at it. The recovery repository reads
`sku1v2_masses_405g.json` by name (`recovery/src/mp4_recovery/paths.py`), so change that name there too.
