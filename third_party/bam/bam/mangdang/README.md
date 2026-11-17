# bam.mangdang — MangDang MD01

Support for the MD01, the 3-in-1 servo module of the Mini Pupper 3:

- `actuator.py`: the `md01v` / `md01vp` actuator models;
- the bench tools that record BAM logs from the module;
- `tools/`: fitting and evaluation helpers.

The parameter sets are in `bam/params/md01/` (see its README for which set to
use), the data in `datasets/md01/`.

## Channels

A module has three outputs, named by their channel:

- `id_1`: the side output (abduction);
- `id_2`: the outer coaxial output (hip);
- `id_3`: the inner coaxial output (knee).

The tools address servos by bus id (`--id`). On the robot, channel n of the
module on driver-board slot k (0–3) is bus id 3k + n, so the bench module on
slot 2 (connector CN3) answers as 7, 8 and 9.

## Hardware chain

    host (spi.py) --USB serial--> ESP32-S3 (bam-firmware bridge) --SPI--> AT32 per module --> motors + potentiometers

- **Frames.** Each SPI transaction is one 36-byte frame per module. The AT32
  answers with the state at the previous frame, so the recorder stamps each
  sample with the previous transaction (`sample_scheme = set_pos_reply_lag1`,
  about 4.5 ms per sample).
- **Configuration commands.** After a configuration command the board's next
  frame carries the reply in channel 1's slot; `spi.py` discards that frame.
- **Port reset.** Opening the serial port resets the ESP32.

## Configuration

**Units do not ship with the same flash configuration.** The recorder writes
the configuration to RAM before each run and logs what the AT32 reads back
(`at32_*`); nothing is saved to flash. The robot's configuration, in which the
parameter sets were recorded (with `kp_position` swept 40–160):

- `control_mode` 1;
- `kp_position` 80, `kd_position` 400, `ki_position` 0;
- `kp_current` 0, `kff_current` 0.001;
- `max_pwm_duty_cycle` 0.99;
- frame current cap 1500 mA.

Pass them explicitly: `--max-pwm-duty-cycle` sets the firmware's ceiling,
while `--max-pwm` only writes BAM's metadata. `check` flags any control_mode 1
log off this configuration as `NOTROBOT`.

In control_mode 1 the frame's current cap limits the duty, not the current: at
1500 mA it is 1.5, so nothing limits the current (about 1 A at stall at 12 V).
Keep blocked holds short. The housing temperature (live id 12, logged) lags
the winding.

## Tools

Bench (`python -m bam.mangdang.<tool>`):

| Tool | Does |
|---|---|
| `record` | one trajectory, one JSON log |
| `all_record` | kp × trajectory sweep, `--shuffle` for a random kp order per block |
| `check` | per-log table of sampling, current, saturation, stalls, the torque-off rest, hold error and configuration flags; `FROZEN` (the output still for over 0.3 s while driven hard) and `SHORT` (a log that ends early) |
| `zero` | two-sided zero settle with a payload; gives `zero_dd` |
| `stall` | blocked-output ladder against a stop: duty against error, current against voltage |
| `ramp_probe` | constant-velocity ramps at kd 0 and kd 400 for the D term |
| `watch` | read-only live monitor (pings only) |

`trajectories.py` adds the small-signal set (`abd_steps`, `abd_sts`,
`abd_sweep`, `abd_drop`) and two recorded target streams, `gait_hip` and
`gait_knee` (`streams/`: the hip and knee setpoints of a walking gait on the
robot, 50 Hz, played about +0.5 rad after a 1 s lead-in), to BAM's trajectories.

`FROZEN` marks an output that is driven hard and does not move. A gravity hold
near the top of `up_and_down` on a high-friction output can trip it too: look at
the log before acting on it.

Offline (`bam/mangdang/tools/`):

| Tool | Does |
|---|---|
| `run_fits.sh` | M1–M6 in parallel |
| `eval_fits.py` | MAE per model and block, `--current`, `--extra`, `--refit-q-offset`, `--trim`; per-log rows in `--json` |
| `unit_transfer.py` | a set on another unit's logs: paired differences against reference units, floors, torque and friction scale scans (`transfer`); tracking of the target streams (`stream`); oscillation while holding a level (`holds`) |
| `plot_validation.py` | held-out grid |
| `plot_detail.py` | angle, speed and current for one log |
| `plant_from_logs.py` | R, V0, kt and `kt_emf` from ladders and logs |
| `compare_params.py` | stiffness, damping, ceiling and static band of parameter sets |
| `duty_probe.py` | regression of the logged firmware duty |

## Recording a dataset

With the module held fixed and a pendulum arm on the output under test (here
id_2 of the module on CN3):

```bash
python -m bam.mangdang.zero --id 8 --lift-deg 12 --logdir raw/zero_arm15
python -m bam.mangdang.all_record --id 8 --mass 0.1605 --length 0.15 --arm-mass 0.01962 \
    --kps 40,60,80,120,160 --kd 400 --shuffle --logdir raw/logs \
    --trajectories lift_and_drop,sin_sin,sin_time_square,steps,up_and_down \
    --extra "--module B --zero-dd 1556 --control-mode 1 --kp-current 0 --kff-current 0.001 \
             --max-pwm-duty-cycle 0.99 --ki-position 0 --supply-v 12"
python -m bam.mangdang.check --logdir raw/logs
mkdir -p processed && python -m bam.process --raw raw/logs --logdir processed --dt 0.005
```

Then fit with `tools/run_fits.sh` or `bam.fit --actuator md01vp`.

`bam.process` extrapolates each log back to t = 0, so the first processed
samples can carry a large speed, from which a rollout starts. `eval_fits.py
--trim 3`, `unit_transfer.py` (by default) and the plot tools' `--trim` drop
them.

The `bam-firmware` repository (`tools/bam_bridge.py dump`) reads a unit's
parameters before anything is written.
