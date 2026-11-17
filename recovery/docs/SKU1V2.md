# SKU1 v2

`MP4R_ROBOT=sku1v2` (read once, when Python starts; `paths.py`) runs everything in this repository on the Mini Pupper v4
SKU1 v2 instead of SKU2: SKU1's body with the same legs, MD01 modules, IMU and servo mapping, weighed at 405 g. Unset, or
`sku2`, nothing changes.

| | SKU2 (default) | SKU1 v2 |
|---|---|---|
| Model | `robot/assets/sku_2/pupper.xml` | `robot/assets/sku_1/pupper.xml`: minipupperesp's `sim/model/pupper.xml` (sha256 `411de14dfe18`), the model RLmp4's SKU1 v2 walks were trained on |
| Masses | `sku2_masses_478g.json` (tasks), `sku2_masses.json` | `sku1v2_masses_405g.json`: 405 g, 177 / 223 g on the front / rear feet (RLmp4's file) |
| Start height | 72.5 mm | 70.1 mm (the hips sit 2.4 mm higher on the base origin) |
| Contract | `sku2_contract.json` | `sku1v2_contract.json`: SKU2's signs and stand, the calibration of mpv2onlyRL's `default.mpxp` (offsets -9, 7, -15, 6, -34, -33, -3.61, 25, 17, -6, 8, 4) |
| Joint limits | `sku2_joint_limits.json` (measured) | `sku1v2_joint_limits.json`: **SKU2's measurement, not this robot's**, with this calibration's track ends |
| Hip envelope | `sku2_hip_envelope.json` (measured) | `sku1v2_hip_envelope.json`: **SKU2's, not measured on this robot** |
| Searched motions | `data/searched_motions` | `data/searched_motions_sku1v2` (`tools/sku1v2_motions.py`) |
| Walker | `policies/walker` | `policies/sku1v2_walker`: RLmp4's `sku1v2_405_walk_s2` ("Walk - SKU1 v2 - 405 g", from `default.mpxp`, `policies/sku1v2_walker_base`), the abductions' stand 12 deg wider and the walking bounds, as `scripts/recipes/walker.sh` does to SKU2's (`policies/sku1v2_walker_stance0`: the bounds without the wider stance, for comparison) |

**Why the limits are estimates.** The abduction stops and the hip envelope come from the chassis beside the hips. SKU1's
and SKU2's models have base boxes of the same width (40 mm half-width) reaching about as far past the hip axes (23.5 and
22.7 mm), and the hip brackets and legs are the same parts, so SKU2's measured ranges are the best guess until this
robot's are measured (ROBOT.md, "Check the joint limits": move the limp legs by hand and record the positions). The
track ends, which the bounds also use, are this robot's calibration. A measured file replaces
`sku1v2_joint_limits.json` (update `limits.LIMITS_SHA256["sku1v2"]`) and needs a retrain of the get-up.

**The searched motions.** SKU2's six open-loop get-ups are replayed on SKU1 v2 in the clone's environment (the recipe's
`wide` options: hip and knee targets 60 deg past the track ends); those that stand at least 75 % of the robots up are kept
unchanged, the others searched again from themselves (`getup_search --init`, `--widen-limits-deg 60`, their own options)
and the better kept. `runs/motions/summary.json` has the table.

**The recipe** is SKU2's, unchanged (`scripts/recipes/getup.sh`, or `getup.ps1` on Windows), reading
`data/searched_motions_sku1v2`.

**Other changes** (both robots): `getup_search --widen-limits-deg`; the evaluations and training record an empty commit
where git or a checkout is missing, instead of failing.
