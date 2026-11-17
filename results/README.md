# What runs on the robot today

The policies in the SKU1 v2 demo build of mpv2onlyRL, kept as the reference that new training runs are compared
against. `check.ps1` exports `policies/sku1v2_walker` and `policies/sku1v2_getup_s41` again and checks that the result
is byte for byte the files in `mpxp/`.

## Files

| File | Policy | Trained by | sha256 (first 16) |
|---|---|---|---|
| `mpxp/sku1v2_405_walk_s2.mpxp` | the walk (`default.mpxp` on the robot) | the walk recipe on SKU1 v2 at 405 g, seed 2 | `d6d3d25cfb016482` |
| `mpxp/sku1v2_recovery.mpxp` | the walker | the walker recipe on that walk (`policies/sku1v2_walker`) | `503c5095c5e2c858` |
| `mpxp/getup_sku1.mpxp` | the get-up | the get-up recipe on SKU1 v2, seed 41 (`policies/sku1v2_getup_s41`) | `13801b2dfdcc433f` |
| `mpxp/sku2_stairs_on_sku1.mpxp` | SKU2's stairs walk `sku2_stairs_narrow_v3`, unchanged, labelled `sku1` (`sku2_stairs.mpxp` on the robot) | SKU2's chain on SKU2 | `f2026c3dcc42060c` |
| `mpxp/sku1v2_405_stairs_4897.mpxp` | the stairs chain on SKU1 v2 at 405 g, iteration 4897 (not in the demo build) | the stairs recipe on SKU1 v2 at 405 g | `6c98eee81a86b036` |

`walk/policies/sku1v2_stairs_narrow_4897` is the first SKU1 v2 stairs run (390 g, before the robot was weighed at
405 g), kept as `walk/`'s scripts reference it. `policies/` holds the `policy.npz` files behind the walker and the get-up. `policies/sku1v2_walker_base` is the walk as
the recovery repository reads it. `recovery/searched_motions_sku1v2/` holds the get-up's six motions (`train.ps1
-ReuseMotions` starts from them), and `recovery/thresholds_s41.json` the get-up's threshold verdict.

## Simulation results

The get-up's searched motions (share of robots standing up):

| Motion | Start | SKU2's motion on SKU1 v2 | Searched again on SKU1 v2 | Kept |
|---|---|---|---|---|
| `search_left` | left | 0.98 | | SKU2's |
| `search_nose` | nose | 1.00 | | SKU2's |
| `search_right_a` | right | 1.00 | | SKU2's |
| `search_right_b` | right | 0.00 | 1.00 | SKU1 v2's |
| `search_back_a` | back | 0.80 | | SKU2's |
| `search_back_b` | back | 0.30 | 1.00 | SKU1 v2's |

The walker, shoved at 0.5-1.5 m/s every 2-4 s for 20 s (share of robots that fell):

| | standing | forward | left | right |
|---|---|---|---|---|
| the 405 g walk with its own stance | 0.77 | 0.88 | 0.91 | 0.87 |
| with the stance 12° wider (the walker) | 0.38 | 0.59 | 0.59 | 0.58 |

The get-up, seed 41 (256 robots per start, 8 s): success and median time to a held stand.

| left | right | nose | back | mid-fall | crouch | random |
|---|---|---|---|---|---|---|
| 1.00, 1.5 s | 1.00, 3.1 s | 1.00, 1.1 s | 1.00, 2.7 s | 0.99, 2.6 s | 1.00, 0.9 s | 1.00, 1.7 s |

The recovery repository's thresholds: **pass**. The walk / shove / fall / get-up / walk sequence (256 robots, 60 s):
topple rate 0.12; walking again within 8 s: 0.52; robots down for over 15 s: 160 (`recovery/tools/sku1v2_report.py`'s
summary of `eval.sequence`).

These are simulation results. On the real robot, the get-up also depends on the calibration
([docs/ROBOT.md](../docs/ROBOT.md#calibration)).
