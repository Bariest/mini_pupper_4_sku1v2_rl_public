# MD01 — MangDang 3-in-1 servo module

One MD01 module drives one leg of the Mini Pupper 3. It holds three brushed DC
motors with gearboxes and potentiometers (two of them coaxial) and one AT32
controller. Its three outputs are named by their channel on the module:

| Output | What it is | Robot joint |
|---|---|---|
| `id_1` | side output | abduction |
| `id_2` | outer coaxial output | hip |
| `id_3` | inner coaxial output | knee |

Each subfolder holds M1–M6 for one output, fitted with the `md01vp` actuator
(`bam/mangdang/actuator.py`). **m6 is the working model for all three.**

| Set | Identified on | Logs | Held-out MAE (kp 120) | Current MAE |
|---|---|---|---|---|
| `md01/id_1` | id_1 of unit A | 77 | 12.2 mrad | 76 mA |
| `md01/id_2` | id_2 of unit B | 115 | 24.1 mrad | 133 mA |
| `md01/id_3` | id_3 of unit A | 175 | 25.9 mrad | 109 mA |

Each MAE is on its own dataset and does not compare across sets: the id_1
dataset has lighter loads and a small-signal trajectory set. Current MAE is the
model's current against the reported one, torque on, loaded logs. The data are
in `datasets/md01/` at the repository root.

## Configuration

AT32 firmware 0.0.8 in `control_mode 1`: the position PID drives the H-bridge
duty directly,

    duty = clip(kp_position * kff_current * error_deg - kd_position * 2.87e-5 * dq, +/-max_pwm_duty_cycle)

with `dq` the measured speed in rad/s (the D term is scaled by the fitted
`kd_ratio`). The data were recorded at `kp_position` 40–160 (120 held out),
`kd_position` 400, `ki_position` 0, `kp_current` 0, `kff_current` 0.001,
`max_pwm_duty_cycle` 0.99, frame current cap 1500 mA and a 12.0 V supply. The
frame cap is a duty cap in this mode (`kff_current * cap` = 1.5), so nothing
limits the current.

The reference simulator (`bam.fit`, `bam.mae`, `eval_fits.py`) gives `md01vp`
each log's kp, kd, kff_current, max_pwm and supply; `bam.mae --mjlab` passes
only kp and the supply. Without a log `md01vp` runs kp 80, kd 400, kff_current
0.001, max_pwm 0.99 and 12 V; set `actuator.kp` / `actuator.kd` to simulate
other gains.

## The actuator model

`md01vp` is BAM's voltage-controlled law with the firmware's P and D terms,
and three plant terms the stock law does not have, each measured on the bench:

- a back-EMF constant `kt_emf` separate from the torque constant `kt`;
- a soft conduction knee instead of a hard `V0` dead band;
- braking torque from the idle H-bridge when torque is off.

Position data only fix combinations of the plant constants (stiffness
`kt*V*G/R`, damping `kt*kt_emf/R + friction_viscous`), so `md01vp` holds kt,
`kt_emf`, R and V0 to the ranges measured at the bench. `md01v` is the same
class with wide ranges.

## How the sets compare

m6 of each set on each dataset, all samples, one `q_offset` refit per dataset
(`bam/mangdang/tools/eval_fits.py --refit-q-offset`). The datasets a set was
fitted on (id_1 and id_3 of unit A, id_2 of unit B) are scored on their
held-out logs (kp 120), the others on all their logs.

| Set | id_1, unit A | id_1, unit B | id_2, unit B | id_3, unit A | id_3, unit B |
|---|---|---|---|---|---|
| `md01/id_1` | **12.0** | **19.6** | 35.1 | 31.8 | 18.2 |
| `md01/id_2` | 16.9 | 21.6 | **24.1** | 27.8 | 16.8 |
| `md01/id_3` | 17.5 | 22.0 | 26.7 | **25.9** | **15.9** |

On the powered trajectories (everything but `lift_and_drop` and `abd_drop`,
all gains) every set is within 4.5 mrad of the best one on every dataset. The
larger differences are in the torque-off drops, where the static friction band
and the idle-bridge braking dominate. The id_1 set is stiffer and stronger
than the other two (`bam/mangdang/tools/compare_params.py`, m6):

| | `md01/id_1` | `md01/id_2` | `md01/id_3` |
|---|---|---|---|
| stiffness at 5° [Nm/rad] | 3.37 | 2.69 | 2.58 |
| motor damping at a hold [Nm·s/rad] | 0.040 | 0.030 | 0.028 |
| torque ceiling at 0 / 5 rad/s [Nm] | 0.73 / 0.57 | 0.61 / 0.47 | 0.56 / 0.43 |

For training, treat the spread between the sets as the uncertainty on
stiffness, damping and the torque ceiling.

## Unit C

Each set was fitted on one unit. A third unit, C, not used in any fit, shows how
well they carry over to another module. All three of its outputs were recorded
with 120 g and 238 g on the 0.10 m arm at kp 80 and 120; id_2 also on the
0.15 m arm, bare and with 160.92 g; id_2 and id_3 also with the gait target
streams (`datasets/md01/id_n/unit_C`).

m6 on all of unit C's logs, one `q_offset` refit, the first 15 ms of each log
dropped (`eval_fits.py --refit-q-offset --trim 3`; `bam.process` extrapolates
each log back to t = 0):

| Set | id_1, unit C | id_2, unit C | id_3, unit C |
|---|---|---|---|
| `md01/id_1` | **19.9** | 29.0 | 15.9 |
| `md01/id_2` | 23.9 | **23.4** | 10.6 |
| `md01/id_3` | 24.3 | 25.2 | **10.0** |

On the powered trajectories both units recorded, the set's own unit against
unit C (`unit_transfer.py transfer`, the mean over those cells):

| Set | Own unit, kp 80 / 120 | Unit C, kp 80 / 120 |
|---|---|---|
| `md01/id_1` | A: 10.0 / 9.6 | 16.3 / 16.1 |
| `md01/id_2` | B: 18.1 / 17.6 | 19.9 / 24.5 |
| `md01/id_3` | A: 17.7 / 19.2 | 9.7 / 8.8 |

The best scale on `kt` per unit is x1.025 for id_3 on all three units,
x0.925–1.125 for id_2 over units B and C and both arms, and x0.900–1.075 for
id_1 over units A, B and C (`unit_transfer.py transfer`). Randomise the torque
by about ±10 % in training. The grids are in `id_n/figures/unit_C_kp120.png`.

## Known limits

- In every set some plant constants sit on their bounds (`kt_emf` and
  `knee_extra` in all three). The individual constants compensate for each
  other; stiffness, damping and the torque ceiling are the identified
  quantities.
- Sustained heavy load when warm is not modelled. At full duty the winding
  heats within seconds, the firmware does not derate, and the model has a
  fixed R. The heaviest blocks (0.31–0.37 Nm of gravity torque) read 28–56
  mrad against 10–26 for the others.
- Large step targets (`steps`, a four-level staircase) are the weakest powered
  trajectory for every set: 35–38 mrad on id_2 and id_3, all gains.
- `bam.mujoco` and `bam.mjlab` apply no torque while torque is off, so the
  idle-bridge brake acts only in the reference simulator, and
  `bam.mae --mujoco` / `--mjlab` are not valid on logs with torque-off
  windows. A policy keeps torque on.
- Identified at `kd_position` 400.
- Fast, small steps on a gravity load, as in a walking gait (the `gait_hip` and
  `gait_knee` streams at kp 80, 238 g on the 0.10 m arm): on id_2 of units B and
  C every set reads 37–64 mrad, against 5.5–8.8 mrad between repeats of the
  same stream. The output moves faster than the model, swings further past the
  setpoints and briefly sticks while driven; changing the D term, the friction
  or the delay does not close it. With 120 g, `gait_knee` on unit C's id_2
  reads 67–77 mrad. On id_3 of unit C the id_2 and id_3 sets read 17–24 mrad
  (`unit_transfer.py stream`; `id_2/figures/gait_hip_238g_10cm_kp80.png`).
- Holding a level at kp 120 with a payload on the 0.10 m arm, id_2 of units B
  and C and id_3 of unit A oscillate at 3.6–5.2 Hz by 15–43 mrad rms; the sets
  reach at most 3 mrad on the same logs (`unit_transfer.py holds`). At kp 80
  the same outputs read 5–15 mrad.
- Unit A's reported current is clipped at 1306 mA; unit B's is not. The fit
  uses positions only; the current MAEs above are a check.

## Reproduce

```bash
mkdir -p fits/id_3
python -m bam.fit --actuator md01vp --model m6 --logdir datasets/md01/id_3/unit_A/processed \
    --validation_kp 120 --trials 60000 --output fits/id_3/m6.json
python bam/mangdang/tools/eval_fits.py --fits bam/params/md01/id_3 \
    --logdir datasets/md01/id_3/unit_A/processed --validation_kp 120 --current
```

`bam/mangdang/tools/run_fits.sh` runs M1–M6 in parallel. In each subfolder the
held-out grids come from `bam/mangdang/tools/plot_validation.py` and the per-log
figures from `bam/mangdang/tools/plot_detail.py`.
