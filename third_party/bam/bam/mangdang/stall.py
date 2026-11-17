# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Blocked-output test of the AT32 current loop.

With the arm resting against a rigid stop, the position loop is given a goal
a few degrees beyond the stop. The error is then constant, the motor stalls
against the stop, and the live values show what the inner loop delivers
against its setpoint: ``setpoint_cur_mA = f(kp_position * error)``,
``present_cur_mA`` and ``pwm_duty`` at zero speed. From those, per level:
``R ~ duty * V / I`` and the loop law
``duty ~ kp_current * (setpoint - present) + kff_current * setpoint``.

    uv run python -m bam.mangdang.stall --id 6 --direction +1 \
        --config normal --out data_raw/at32_stall_test.csv

A bench supply's ammeter cannot tell the motor current from the supply-side
current here (it reads about the same number either way: the average under one
reading, the pulse peak under the other). Record the same ladder at two supply
voltages instead and compare the reported current against ``duty * V``
(2026-09-22).

``--config`` selects the AT32 configuration written before the run (RAM only;
the values read at start are restored afterwards unless ``--no-restore``):
``normal`` = the flash configuration of the legacy current loop
(control_mode 0, kp_c 6e-4, kff 3e-4, max_pwm 0.99), ``kff0`` = the same with
kff_current 0, ``strong`` = the robot's Strong preset (6e-5 / 1.1e-3 / 0.99);
``mode1`` = firmware >= 0.0.8 ``control_mode 1`` (PID -> PWM) as the new
module ships it except max_pwm 0.99 (kp_c 0, kff 0.001, ki 0). ``--kff``,
``--max-pwm``, ``--kd``, ``--ki`` override single values for the scale and
D/I probes. Holds are short and separated by idle periods; in mode 1 nothing
may cap the current, so the default hold is 1.5 s and ``--max-ma`` 1300.
The housing temperature is read from live id 12 (fw >= 0.0.8) when it
answers, else from the feedback ``res`` word.
"""

import argparse
import csv
import os
import time

import numpy as np

from bam.mangdang.spi import LIVE, PARAM, SpiMd01IO, pick_port

CONFIGS = {
    "normal": {"control_mode": 0, "kp_current": 6e-4, "kff_current": 3e-4, "max_pwm_duty_cycle": 0.99},
    "kff0": {"control_mode": 0, "kp_current": 6e-4, "kff_current": 0.0, "max_pwm_duty_cycle": 0.99},
    "strong": {"control_mode": 0, "kp_current": 6e-5, "kff_current": 1.1e-3, "max_pwm_duty_cycle": 0.99},
    "mode1": {"control_mode": 1, "ki_position": 0.0, "kp_current": 0.0, "kff_current": 1e-3, "max_pwm_duty_cycle": 0.99},
}

parser = argparse.ArgumentParser(description="Blocked-output current-loop test")
parser.add_argument("--id", type=int, required=True, help="bus id 1..12")
parser.add_argument("--port", type=str, default=None)
parser.add_argument("--kp", type=float, default=80.0)
parser.add_argument("--cur", type=int, default=1500, help="frame current cap [mA]")
parser.add_argument("--errors", type=str, default=None, help="deg beyond the stop (default per config)")
parser.add_argument("--direction", type=int, choices=(-1, 1), required=True, help="+1 pushes toward increasing dd")
parser.add_argument("--hold", type=float, default=None, help="seconds per level (default 3 legacy, 1.5 mode 1)")
parser.add_argument("--rest", type=float, default=None, help="idle seconds between levels (default 3 / 6)")
parser.add_argument("--config", type=str, choices=sorted(CONFIGS), default="normal")
parser.add_argument("--kd", type=float, default=0.0, help="kd_position during the test")
parser.add_argument("--ki", type=float, default=None, help="override ki_position (fw >= 0.0.7)")
parser.add_argument("--kff", type=float, default=None, help="override kff_current")
parser.add_argument("--max-pwm", type=float, default=None, help="override max_pwm_duty_cycle")
parser.add_argument("--skip-mode", action="store_true", help="do not write control_mode (fw < 0.0.8)")
parser.add_argument("--label", type=str, default="", help="free text stored with every row (supply used)")
parser.add_argument("--out", type=str, required=True, help="CSV, appended to")
parser.add_argument("--no-restore", action="store_true")
parser.add_argument("--max-ma", type=int, default=None, help="abort if the frame current exceeds this (1600 / 1300)")
parser.add_argument("--max-temp", type=float, default=62.0, help="abort if the housing exceeds this [C]")
args = parser.parse_args()

MODE1 = CONFIGS[args.config].get("control_mode", 0) == 1
if args.errors is None:
    args.errors = "1,2,3,4,6,8,10,12,15" if MODE1 else "2.5,5,7.5,10,15,20"
if args.hold is None:
    args.hold = 1.5 if MODE1 else 3.0
if args.rest is None:
    args.rest = 6.0 if MODE1 else 3.0
if args.max_ma is None:
    args.max_ma = 1300 if MODE1 else 1600

FIELDS = [
    "config", "label", "t", "level_deg", "setpoint_pos_deg", "present_pos_deg", "error_pos_deg",
    "max_current_mA", "setpoint_cur_mA", "present_cur_mA", "error_cur_mA", "pwm_duty",
    "frame_cur_mA", "temp_c", "kp_position", "kd_position", "ki_position", "control_mode",
    "kp_current", "kff_current", "max_pwm_duty_cycle",
]


def read_live(io, servo, names):
    return {name: float(io.get_live(servo, LIVE[name])["val"]) for name in names}


def read_temperature(io, servo) -> float:
    """Housing temperature [C]: live id 12 when the firmware answers it, else the res word."""
    try:
        t = float(io.get_live(servo, LIVE["temperature_c"])["val"])
        if 0.0 < t < 150.0:
            return t
    except Exception:
        pass
    res = io.last_res & 0xFFFF
    return res / 10.0 if 0 < res < 3000 else 0.0


def main() -> None:
    io = SpiMd01IO(pick_port(args.port), cur_ma=args.cur)
    servo = args.id
    io.sync_scale_from_board(servo)
    flash = io.dump_params(servo)
    gains = dict(CONFIGS[args.config])
    if args.kff is not None:
        gains["kff_current"] = args.kff
    if args.max_pwm is not None:
        gains["max_pwm_duty_cycle"] = args.max_pwm
    if args.ki is not None:
        gains["ki_position"] = args.ki
    if args.skip_mode:
        gains.pop("control_mode", None)
        gains.pop("ki_position", None)
    # The channel is idle here (nothing commanded yet): mode and gains change at rest.
    io.set_idle(servo)
    io.set_P_coefficient({servo: args.kp})
    io.set_D_coefficient({servo: args.kd})
    for name, value in gains.items():
        io.set_param(servo, PARAM[name], value)
    active = io.dump_params(servo)
    for name, value in gains.items():
        assert abs(active[name] - value) < 1e-6 * max(1.0, abs(value)), (name, active[name], value)
    assert abs(active["kd_position"] - args.kd) < 1e-6, ("kd_position", active["kd_position"])
    print(f"config {args.config}: mode {active.get('control_mode')} kp {active['kp_position']} "
          f"kd {active['kd_position']} ki {active.get('ki_position')} kp_c {active['kp_current']:.2e} "
          f"kff {active['kff_current']:.2e} max_pwm {active['max_pwm_duty_cycle']}  "
          f"T {read_temperature(io, servo):.1f} C  live cap {read_live(io, servo, ['max_current_mA'])['max_current_mA']:.0f} mA")

    exists = os.path.exists(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    stop_dd = float(np.mean([io.ping(servo)["pos_dd"] for _ in range(10)]))
    print(f"arm on the stop at {stop_dd:.0f} dd ({io.dd_to_deg(stop_dd):+.1f} deg from centre)")
    summary = []
    t0 = time.monotonic()
    try:
        for level in [float(x) for x in args.errors.split(",")]:
            goal_dd = int(round(stop_dd + args.direction * level * 10.0))
            rows = []
            t_start = time.monotonic()
            while time.monotonic() - t_start < args.hold:
                r = io.set_pos_dd(servo, goal_dd)
                if abs(r["cur_mA"]) > args.max_ma:
                    raise RuntimeError(f"frame current {r['cur_mA']} mA above --max-ma; aborting")
                live = read_live(io, servo, [
                    "setpoint_pos_deg", "present_pos_deg", "error_pos_deg", "max_current_mA",
                    "setpoint_cur_mA", "present_cur_mA", "error_cur_mA", "pwm_duty",
                ])
                temp = read_temperature(io, servo)
                row = {
                    "config": args.config, "label": args.label, "t": round(time.monotonic() - t0, 3),
                    "level_deg": level, **{k: round(v, 3) for k, v in live.items()},
                    "frame_cur_mA": r["cur_mA"], "temp_c": temp,
                    "kp_position": active["kp_position"], "kd_position": active["kd_position"],
                    "ki_position": active.get("ki_position"), "control_mode": active.get("control_mode"),
                    "kp_current": active["kp_current"], "kff_current": active["kff_current"],
                    "max_pwm_duty_cycle": active["max_pwm_duty_cycle"],
                }
                writer.writerow(row)
                rows.append(row)
                if temp > args.max_temp:
                    raise RuntimeError(f"housing {temp:.1f} C above --max-temp; aborting")
                time.sleep(0.05)
            io.set_idle(servo)
            tail = [r for r in rows if r["t"] - rows[0]["t"] > args.hold * 0.5]
            m = {k: float(np.mean([r[k] for r in tail])) for k in
                 ("error_pos_deg", "setpoint_cur_mA", "present_cur_mA", "pwm_duty", "frame_cur_mA", "temp_c")}
            moved = io.dd_to_deg(np.mean([r["present_pos_deg"] * 10 for r in tail])) - io.dd_to_deg(stop_dd)
            summary.append((level, m, moved))
            print(f"  level {level:5.1f} deg: err {m['error_pos_deg']:6.2f} deg  setpoint {m['setpoint_cur_mA']:6.0f} mA  "
                  f"present {m['present_cur_mA']:6.0f} mA (frame {m['frame_cur_mA']:6.0f})  duty {m['pwm_duty']:+.3f}  "
                  f"T {m['temp_c']:.1f} C  arm moved {moved:+.2f} deg")
            # Against a stop the output barely moves (a gap to the stop and its
            # compliance take up to about a degree). If it followed the goal,
            # --direction points away from the stop and nothing is measured.
            if args.direction * moved > max(1.5, 0.5 * level):
                raise RuntimeError(
                    f"the arm followed the goal by {moved:+.2f} deg at level {level:g} deg: it is not "
                    "pressing on the stop; the stop is on the other side, so flip --direction")
            time.sleep(args.rest)
    finally:
        io.set_idle(servo)
        out.close()
        if not args.no_restore:
            for name in gains:
                io.set_param(servo, PARAM[name], flash[name])
            io.set_D_coefficient({servo: flash["kd_position"]})
            io.set_P_coefficient({servo: flash["kp_position"]})
            print("start-of-run parameters restored in RAM")
        io.close()
    print(f"{len(summary)} levels -> {args.out}")


if __name__ == "__main__":
    main()
