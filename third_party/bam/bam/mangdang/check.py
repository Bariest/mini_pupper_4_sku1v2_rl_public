# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Bench-side sanity table for raw MD01 logs, so a bad capture is caught now.

    uv run python -m bam.mangdang.check --logdir data_raw [--i-stall 850]

One line per log: samples, median / max period, max |current|, the fraction
of samples above 0.8 x the stall current, max |position error|, the position
range in raw deci-degrees (with the distance to the pot ends), the rest
position after a torque-off drop, the pre-roll hold error, the temperature,
the longest time the output sat still while driven hard, and whether the AT32
configuration matches the first log's.

Flags: ``FROZEN`` when the output stood still (within 5 mrad) for more than
0.3 s with torque on, more than 0.1 rad of error and more than 600 mA -- a
wedged gear stops the output at one angle under full drive, as does a load
beyond the torque ceiling; ``SHORT`` when the log ends early (an interrupted
run still writes its log).
"""

import argparse
import glob
import json
import os

import numpy as np

from bam.mangdang.trajectories import trajectories

parser = argparse.ArgumentParser(description="Sanity-check raw MD01 logs")
parser.add_argument("--logdir", type=str, required=True)
parser.add_argument("--i-stall", type=float, default=None,
                    help="stall current [mA] for the saturation column (default 850 legacy loop, 1000 control_mode 1)")
parser.add_argument("--end-margin", type=float, default=100.0, help="dd of margin to the pot ends to flag")
args = parser.parse_args()

CONFIG_KEYS = [
    "at32_kp_current", "at32_kff_current", "at32_max_pwm_duty_cycle", "at32_kd_position",
    "at32_ki_position", "at32_control_mode",
    "cur_cap_ma", "zero_dd", "supply_v", "supply_ilim_a",
]

#: The robot's control_mode 1 configuration (kp and kd vary by design). A mode-1
#: log that read back anything else was recorded with part of the unit's flash
#: configuration still in force, and is flagged whatever the first log says.
ROBOT_MODE1 = {"at32_kp_current": 0.0, "at32_kff_current": 0.001,
               "at32_max_pwm_duty_cycle": 0.99, "at32_ki_position": 0.0}


#: FROZEN: driven hard (torque on, |error| and |current| above these) yet still.
FROZEN_ERR_RAD = 0.1
FROZEN_CUR_MA = 600.0
FROZEN_STILL_RAD = 0.005
FROZEN_MIN_S = 0.3


def frozen_span(t, q, goal, enable, cur) -> tuple[float, float]:
    """Longest time [s] the output stayed within ``FROZEN_STILL_RAD`` while driven hard, and where [rad]."""
    driven = enable & (np.abs(goal - q) > FROZEN_ERR_RAD) & (cur > FROZEN_CUR_MA)
    best, where = 0.0, float("nan")
    i, n = 0, len(t)
    while i < n:
        if not driven[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and driven[j + 1]:
            j += 1
        lo = i
        for hi in range(i, j + 1):
            while np.ptp(q[lo:hi + 1]) > FROZEN_STILL_RAD:
                lo += 1
            if t[hi] - t[lo] > best:
                best, where = t[hi] - t[lo], float(np.mean(q[lo:hi + 1]))
        i = j + 1
    return best, where


def main() -> None:
    files = sorted(glob.glob(os.path.join(args.logdir, "*.json")))
    files = [f for f in files if os.path.basename(f) != "zero.json"]
    if not files:
        raise SystemExit(f"no logs in {args.logdir}")
    reference = None
    print(
        f"{'file':22s} {'traj':16s} {'kp':>4s} {'m':>6s} {'n':>5s} {'dt50':>5s} {'dtmax':>6s} "
        f"{'|I|max':>6s} {'sat%':>4s} {'|err|':>5s} {'dd min..max':>12s} {'edge':>4s} "
        f"{'rest':>6s} {'hold':>6s} {'T':>5s} {'frz':>4s} {'md':>2s} {'kd':>4s} cfg"
    )
    for f in files:
        d = json.load(open(f))
        e = d["entries"]
        t = np.array([x["timestamp"] for x in e])
        q = np.array([x["position"] for x in e])
        goal = np.array([x["goal_position"] for x in e])
        en = np.array([x["torque_enable"] for x in e], dtype=bool)
        cur = np.abs(np.array([x["load"] for x in e]))
        temp = np.array([x.get("temp", 0.0) for x in e])
        mode = int(d.get("at32_control_mode", 0) or 0)
        i_stall = args.i_stall if args.i_stall is not None else (1000.0 if mode == 1 else 850.0)
        duty = np.abs(np.array([x["pwm_duty"] for x in e])) if "pwm_duty" in e[0] else None
        t_live = d.get("temp_live_end_c") or d.get("temp_live_start_c")
        t_max = max(temp.max(), t_live or 0.0)
        dt = np.diff(t) * 1e3
        zero_dd = d.get("zero_dd", d.get("scale_dd", 3100.0) / 2.0)
        scale = d.get("scale_dd", 3100.0)
        dd = q / (0.1 * np.pi / 180.0) + zero_dd
        edge = min(dd.min(), scale - dd.max())
        err = np.abs(goal[en] - q[en]).max() if en.any() else 0.0
        if duty is not None and en.any():
            sat = 100.0 * np.mean(duty[en] > 0.9 * float(d.get("at32_max_pwm_duty_cycle", 1.0)))
        else:
            sat = 100.0 * np.mean(cur[en] > 0.8 * i_stall) if en.any() else 0.0
        rest = np.mean(q[-len(q) // 6:]) if (~en).any() else float("nan")
        frozen, frozen_at = frozen_span(t, q, goal, en, cur)
        trajectory = trajectories.get(d["trajectory"])
        try:
            short = trajectory is not None and t[-1] < trajectory.duration - 0.2
        except OSError:  # a recorded stream whose file is not here
            short = False
        cfg = {k: d.get(k) for k in CONFIG_KEYS}
        if reference is None:
            reference = cfg
            cfg_flag = "ref"
        else:
            diff = [k for k in CONFIG_KEYS if cfg[k] != reference[k]]
            cfg_flag = "ok" if not diff else "DIFF:" + ",".join(k.replace("at32_", "") for k in diff)
        flags = []
        if np.median(dt) > 5.5:
            flags.append("slow")
        if dt.max() > 20:
            flags.append("gap")
        if edge < args.end_margin:
            flags.append("EDGE")
        if sat > 30:
            flags.append("STALL")
        if abs(d.get("q_offset", 0.0)) > 0.02:
            flags.append("hold")
        if t_max > 62.0:
            flags.append("HOT")
        if frozen > FROZEN_MIN_S:
            flags.append(f"FROZEN@{np.degrees(frozen_at):+.1f}deg")
        if short:
            flags.append("SHORT")
        if mode == 1:
            off = [k for k, v in ROBOT_MODE1.items() if d.get(k) is None or abs(float(d[k]) - v) > 1e-5]
            if off:
                flags.append("NOTROBOT:" + ",".join(k.replace("at32_", "") for k in off))
        print(
            f"{os.path.basename(f)[:22]:22s} {d['trajectory'][:16]:16s} {d['kp']:4.0f} {d['mass']:6.3f} "
            f"{len(e):5d} {np.median(dt):5.2f} {dt.max():6.1f} {cur.max():6.0f} {sat:4.0f} "
            f"{err:5.2f} {dd.min():5.0f}..{dd.max():5.0f} {edge:4.0f} "
            f"{rest:+6.3f} {d.get('q_offset', 0.0):+6.3f} {t_max:5.1f} {frozen:4.1f} {mode:2d} {d.get('kd', 0):4.0f} {cfg_flag} {' '.join(flags)}"
        )


if __name__ == "__main__":
    main()
