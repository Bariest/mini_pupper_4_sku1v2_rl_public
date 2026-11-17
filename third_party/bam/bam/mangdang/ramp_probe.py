# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Constant-velocity ramps with the duty read per sample: what does ``kd`` multiply?

Drives the goal at a constant rate across the rig zero and reads ``pwm_duty``
after every position command. In steady tracking ``d(error)/dt`` is ~0 while the
velocity is not, so a D term on the measured velocity leaves a speed-proportional
offset in ``duty - kp*kff_current*error_deg`` and a D term on the error rate
leaves none. Run it once at ``--kd 0`` and once at the robot's kd; the kd-0 run
is not a control but a *necessary* subtraction, because the position lag (bridge
round trip + the AT32's one-frame reply, ~10.5 ms) puts a residual of the same
shape and five times the size into both runs.

    uv run python -m bam.mangdang.ramp_probe --id 9 --kd 0   --zero-dd 1554 --out probe/ramp_kd0.json
    uv run python -m bam.mangdang.ramp_probe --id 9 --kd 400 --zero-dd 1554 --out probe/ramp_kd400.json

Analysis (id_3 of unit A): pair each (speed, direction) segment with its
twin in the other file, take means over a settle window, and correct for the
small speed difference between the runs::

    KD*kd*v400 = (resid400 - resid0) - L*(v400 - v0),  L = resid0 / v0

Without the correction the gain comes out ~10 % low. Result: 0.0115 duty per
rad/s at kd 400, i.e. 2.87e-5 per unit kd.

Coupling between the two coaxial shafts: ``--other`` names the other channel
of the same module, ``--other-mode idle`` leaves it idle (its motor braked by
the idle bridge) and ``--other-mode hold`` holds it where it is in the robot's
configuration (mode 1, kp 80, kd 400). Either way it is pinged after every
sample, so a drag shows as motion of the idle partner and a coupling torque as
current in the held one::

    uv run python -m bam.mangdang.ramp_probe --id 8 --kd 0 --zero-dd 1550 \
        --other 9 --other-mode hold --out probe/ramp_s8_kd0_hold9.json

Moves between segments slew at ``--slew`` rad/s instead of stepping, so the
repositioning does not throw the shaft.

Output JSON: a flat list of samples, ``{v, dir, t, goal, pos, cur, duty, kd}``
with ``goal``/``pos`` in rad about the rig zero and ``t`` restarted per segment
(the timestamp is taken after the duty read, so every quantity in a row belongs
to the same round trip), plus ``other_dd`` and ``other_cur`` with ``--other``.
This is NOT a BAM recorder log.
"""

import argparse
import json
import math
import time

import numpy as np

from bam.mangdang.spi import LIVE, PARAM, SpiMd01IO, module_servos, pick_port

parser = argparse.ArgumentParser(description="Constant-velocity ramp probe for the AT32 D term")
parser.add_argument("--id", type=int, required=True, help="bus id 1..12")
parser.add_argument("--port", type=str, default=None)
parser.add_argument("--kd", type=float, required=True, help="kd_position for this run")
parser.add_argument("--kp", type=float, default=80.0)
parser.add_argument("--kff", type=float, default=0.001, help="kff_current (the mA -> duty scale)")
parser.add_argument("--max-pwm", type=float, default=0.99)
parser.add_argument("--cur", type=int, default=1500, help="frame current cap [mA]")
parser.add_argument("--zero-dd", type=float, required=True, help="rig zero from bam.mangdang.zero")
parser.add_argument("--amp-deg", type=float, default=55.0, help="ramp half-range about the zero")
parser.add_argument("--speeds", type=str, default="0.5,1,2,4", help="rad/s")
parser.add_argument("--max-ma", type=int, default=1300, help="abort above this current")
parser.add_argument("--other", type=int, default=None, help="the other shaft of the module, logged per sample")
parser.add_argument("--other-mode", choices=("idle", "hold"), default="idle",
                    help="idle: left idle; hold: held where it is at kp 80 / kd 400 in mode 1")
parser.add_argument("--slew", type=float, default=2.0, help="rad/s for the moves between segments")
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

#: The robot's configuration, written to every channel this tool drives.
ROBOT = (("kp_current", 0.0), ("kff_current", 0.001), ("max_pwm_duty_cycle", 0.99),
         ("ki_position", 0.0), ("control_mode", 1.0))
RESTORE = ("kp_current", "kff_current", "max_pwm_duty_cycle", "ki_position", "control_mode")


def slew(io, servo: int, target: float, speed: float) -> None:
    """Move the goal from the present position to ``target`` [rad] at ``speed`` rad/s."""
    start = io.dd_to_rad(io.ping(servo)["pos_dd"])
    steps = max(1, int(abs(target - start) / speed / 0.01))
    for k in range(1, steps + 1):
        io.set_pos_dd(servo, io.rad_to_dd(start + (target - start) * k / steps))
        time.sleep(0.01)


def main() -> None:
    servo, amp = args.id, math.radians(args.amp_deg)
    module = module_servos(servo)
    other = args.other
    if other is not None and (other == servo or other not in module):
        raise SystemExit(f"--other must be another channel of servo {servo}'s module {module}")
    io = SpiMd01IO(pick_port(args.port), cur_ma=args.cur, zero_dd=args.zero_dd)
    for s in module:
        io.set_idle(s)
    time.sleep(0.3)
    flash = io.dump_params(servo)
    io.set_P_coefficient({servo: args.kp})
    io.set_D_coefficient({servo: args.kd})
    for name, value in ROBOT:
        io.set_param(servo, PARAM[name], args.kff if name == "kff_current" else
                     args.max_pwm if name == "max_pwm_duty_cycle" else value)
    active = io.dump_params(servo)
    assert abs(active["kd_position"] - args.kd) < 1e-6, active
    assert abs(active["control_mode"] - 1.0) < 1e-6, active
    other_flash, hold_dd = None, None
    if other is not None:
        other_flash = io.dump_params(other)
        if args.other_mode == "hold":
            io.set_P_coefficient({other: 80.0})
            io.set_D_coefficient({other: 400.0})
            for name, value in ROBOT:
                io.set_param(other, PARAM[name], value)
            held = io.dump_params(other)
            assert abs(held["control_mode"] - 1.0) < 1e-6 and abs(held["kp_position"] - 80.0) < 1e-6, held
            hold_dd = io.ping(other)["pos_dd"]
            io.set_pos_dd(other, hold_dd)
    print(f"kd {active['kd_position']:.0f}, kp {active['kp_position']:.0f}, "
          f"kff {active['kff_current']:.4f}, max_pwm {active['max_pwm_duty_cycle']:.2f}, "
          f"T {io.get_live(servo, LIVE['temperature_c'])['val']:.1f} C"
          + (f"; servo {other} {args.other_mode}" + (f" at {hold_dd} dd" if hold_dd is not None else "")
             if other is not None else ""))

    rows: list[dict] = []
    try:
        slew(io, servo, -amp, args.slew)
        time.sleep(1.0)
        for speed in [float(x) for x in args.speeds.split(",")]:
            for direction in (+1, -1):
                start = -amp * direction
                slew(io, servo, start, args.slew)
                time.sleep(0.8)
                t0 = time.perf_counter()
                while True:
                    t = time.perf_counter() - t0
                    goal = start + direction * speed * t
                    if abs(goal) > amp:
                        break
                    reply = io.set_pos_dd(servo, io.rad_to_dd(goal))
                    duty = float(io.get_live(servo, LIVE["pwm_duty"])["val"])
                    row = {"v": speed, "dir": direction, "t": t, "goal": goal,
                           "pos": io.dd_to_rad(reply["pos_dd"]),
                           "cur": reply["cur_mA"], "duty": duty, "kd": args.kd}
                    if other is not None:
                        o = io.ping(other)
                        row.update(other_dd=o["pos_dd"], other_cur=o["cur_mA"], other_mode=args.other_mode)
                        if abs(o["cur_mA"]) > args.max_ma:
                            raise RuntimeError(f"servo {other}: {o['cur_mA']} mA above --max-ma; aborting")
                    rows.append(row)
                    if abs(reply["cur_mA"]) > args.max_ma:
                        raise RuntimeError(f"{reply['cur_mA']} mA above --max-ma; aborting")
                slew(io, servo, direction * amp, args.slew)
                time.sleep(0.5)
    finally:
        for s in module:
            io.set_idle(s)
        io.set_D_coefficient({servo: flash["kd_position"]})
        io.set_P_coefficient({servo: flash["kp_position"]})
        for name in RESTORE:
            io.set_param(servo, PARAM[name], flash[name])
        if other_flash is not None and args.other_mode == "hold":
            io.set_D_coefficient({other: other_flash["kd_position"]})
            io.set_P_coefficient({other: other_flash["kp_position"]})
            for name in RESTORE:
                io.set_param(other, PARAM[name], other_flash[name])
        json.dump(rows, open(args.out, "w"))
        io.close()
    print(f"{len(rows)} samples -> {args.out}")


if __name__ == "__main__":
    main()
