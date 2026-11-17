# MD01 id_3 — inner coaxial output (knee)

Fitted with `md01vp` on `datasets/md01/id_3/unit_A`: 175 logs of id_3 on unit A,
recorded 2026-09-23 in the configuration given in `../README.md`. Seven load
blocks on a pendulum arm: the 0.15 m arm (19.62 g) with no payload, 79, 160 and
238 g, and the 0.10 m arm (15.95 g) with 120, 238 and 303 g; each at kp 40, 60,
80, 120 and 160 with `lift_and_drop`, `sin_sin`, `sin_time_square`, `steps` and
`up_and_down`. kp 120 is held out.

**Working model: `m6.json`** (alternative `m5.json`).

| | held-out (kp 120) | current | `repeat` | `kd0` | `heavy_repeat` |
|---|---|---|---|---|---|
| m3 | 27.5 mrad | 114 mA | 13.0 | 15.3 | 40.3 |
| m5 | 25.8 | 124 | 14.4 | 16.2 | 38.0 |
| **m6** | **25.9** | **109** | **13.1** | **15.4** | **38.5** |

The extras are not fitted:

- `repeat`: the 120 g block at kp 80, recorded again. It differs from the same
  logs in the fit set by 4.0–10.9 mrad per trajectory (all samples).
- `kd0`: the same block at `kd_position` 0.
- `heavy_repeat`: the 238 g block on the 0.15 m arm at all five gains,
  recorded again after remounting the arm.

Per block (m6, all gains), the five light and medium blocks read 16–25 mrad,
303 g on the 0.10 m arm 30 mrad, and 238 g on the 0.15 m arm 42 mrad.

## Figures

- `figures/validation_kp120.png`: every held-out log (7 blocks × 5
  trajectories): goal, measured, m1, m5 and m6, with each panel's MAE.
- The per-log figures show angle, speed and current for the most informative
  held-out logs. The current is never seen by the fit.
- `figures/unit_C_kp120.png`: m6 on unit C's kp-120 logs (`datasets/md01/id_3/unit_C`, a unit not
  used in the fit): `plot_validation.py --unseen --trim 3 --q-offset -0.0223 --labels m6`.
- Regenerate with `bam/mangdang/tools/plot_validation.py` and `plot_detail.py`.
