# MD01 id_2 — outer coaxial output (hip)

Fitted with `md01vp` on `datasets/md01/id_2/unit_B`: 115 logs of id_2 on unit B,
recorded 2026-09-24/26 in the configuration given in `../README.md`.

Load blocks on a pendulum arm:
- **0.15 m arm (19.62 g):** no payload and 160.5 g at kp 40, 60, 80, 120 and 160; 79.8 g and 238.5 g at kp 80.
- **0.10 m arm (16.18 g):** 120.5 g and 303.9 g at all five gains; 238.3 g at kp 80.

Each with `lift_and_drop`, `sin_sin`, `sin_time_square`, `steps` and `up_and_down`.
kp 120 is held out.

**Working model: `m6.json`** (alternative `m5.json`).

| | held-out (kp 120) | current | `repeat` | `kd0` | `heavy_repeat` |
|---|---|---|---|---|---|
| m3 | 25.2 mrad | 131 mA | 14.8 | 19.5 | 60.0 |
| m5 | 25.1 | 129 | 15.1 | 19.7 | 58.7 |
| **m6** | **24.1** | **133** | **13.7** | **18.4** | **58.9** |

The extras are not fitted:

- `repeat`: the 120.5 g block at kp 80, recorded again. It differs from the
  same logs in the fit set by 6.9–15.2 mrad per trajectory (all samples).
- `kd0`: the same block at `kd_position` 0.
- `heavy_repeat`: the 303.9 g block at all five gains, recorded again after
  remounting the arm.

Per block (m6, all gains), the six lighter blocks read 10–28 mrad, and the
303.9 g block 56 mrad. Held at full duty the winding heats within seconds, and
the firmware does not derate. The duty probes in
`datasets/md01/id_2/unit_B/raw/probe_duty` show the
stall current falling 14 % over four lifts at duty 0.99. The model has a fixed
R.

## Figures

- `figures/validation_kp120.png`: every held-out log (4 blocks × 5
  trajectories): goal, measured, m1, m5 and m6, with each panel's MAE.
- The per-log figures show angle, speed and current from the bare arm to the
  heaviest block.
- `figures/unit_C_kp120.png`: m6 on unit C's kp-120 logs (`datasets/md01/id_2/unit_C`, a unit not
  used in the fit): `plot_validation.py --unseen --trim 3 --q-offset 0.0245 --labels m6`.
- `figures/gait_hip_238g_10cm_kp80.png`: the `gait_hip` stream on unit B
  (`extra/gait`): `plot_detail.py --trim 3 --q-offset 0.005 --labels m6`.
- Regenerate with `bam/mangdang/tools/plot_validation.py` and `plot_detail.py`.
