# MD01 id_1 — side output (abduction)

Fitted with `md01vp` on `datasets/md01/id_1/unit_A`: 77 logs of id_1 on unit A,
recorded 2026-09-28 in the configuration given in `../README.md`.

The abduction mostly holds the leg, so besides the five standard trajectories
the dataset has a small-signal set: `abd_steps`, `abd_sts`, `abd_sweep` and
`abd_drop`, within ±0.35 rad of the vertical (`bam/mangdang/trajectories.py`).

Load blocks:
- **0.10 m arm (16.18 g):** 120.5 g and 238.3 g at kp 40, 60, 80, 120 and 160 with the standard trajectories, plus the small-signal set at kp 80 and 120.
- **0.15 m arm (21.36 g):** 160.0 g at kp 80 and 120, plus `lift_and_drop` at kp 160.

kp 120 is held out.

**Working model: `m6.json`** (alternative `m5.json`; `m3.json` is the simplest).

| | held-out (kp 120) | current | `repeat` | `kd0` | `small_repeat` |
|---|---|---|---|---|---|
| m3 | 12.2 mrad | 69 mA | 10.7 | 12.5 | 6.5 |
| m5 | 11.9 | 83 | 10.1 | 13.1 | 6.4 |
| **m6** | **12.2** | **76** | **10.7** | **11.3** | **9.0** |

The extras are not fitted:

- `repeat`: the 120.5 g block at kp 80, recorded again. It differs from the
  same logs in the fit set by 4.0–7.8 mrad per trajectory and 27.8 mrad on
  `lift_and_drop` (all samples).
- `small_repeat`: the small-signal set, recorded again; 1.0–4.7 mrad.
- `kd0`: the standard trajectories at `kd_position` 0.

A second unit's id_1, `datasets/md01/id_1/unit_B` (73 logs), reads 19.6 mrad on
all its logs with this m6 and one `q_offset` refit.

## Figures

- `figures/validation_kp120.png` (standard trajectories) and
  `figures/validation_small_kp120.png` (small-signal set): the held-out logs,
  goal, measured, m1, m5 and m6, with each panel's MAE.
- The per-log figures show angle, speed and current for held-out logs from both
  arms and both trajectory sets.
- `figures/unit_C_kp120.png`: m6 on unit C's kp-120 logs (`datasets/md01/id_1/unit_C`, a unit not
  used in the fit): `plot_validation.py --unseen --trim 3 --q-offset -0.0427 --labels m6`.
- Regenerate with `bam/mangdang/tools/plot_validation.py` and `plot_detail.py`.
