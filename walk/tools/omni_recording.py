#!/usr/bin/env python
"""An mpx-recording take (``mpxrec-*.csv``) checked against the host-built firmware gait, and summarised per keyframe.

The recording firmware (mpx-recording ``main/gait.c``) runs the same stanford_gait.c / stanford_kinematics.c as
minipupperesp and records one row per 15 ms gait tick (``flush``). This replays the take's keyframe schedule
(``gait_task``: halt_in, then each keyframe ramps from the previous one over ramp_ms, eased, and holds for hold_ms;
full stick 200 mm/s, 2 rad/s) through ``firmware_gait.Gait`` with the take's own trims, and compares the ticks with
the recorded commands. It then reports, per keyframe hold, the commanded and measured joint motion in the policy's
coordinates (the contract's signs; the take's trims).

    python -B tools/omni_recording.py ../mpx-sim/mpxrec-omni.csv [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mp3_walk_policy.contract import Contract                                          # noqa: E402
from mp3_walk_policy.firmware_gait import Gait                                         # noqa: E402
from mp3_walk_policy.takes import (VX_MAX_MM_S as VX_MAX, VY_MAX_MM_S as VY_MAX,      # noqa: E402
                                   WZ_MAX_RAD_S as WZ_MAX, FirmwareLegs, read_take, schedule)

HALT_MS = 200


def replay(head, take, t_walk0):
    """The host gait's ticks for every row from the first walking tick (at `t_walk0` ms) until the take's end,
    through gait_task's leg phases (`takes.FirmwareLegs`: a stand keyframe ramps the feet home and holds the stance)."""
    g = Gait(head["trim_deg"])
    legs = FirmwareLegs(g, head["height_mm"])
    out = np.full_like(take["cmd"], -1)
    motion = []
    for i, t in enumerate(take["t"]):
        if t < t_walk0:
            continue
        m = schedule(head["keyframes"], t - t_walk0, head["height_mm"])
        if m is None:
            break
        out[i] = g.ticks(legs.tick(t, m))
        motion.append((i, m))
    return out, motion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--json")
    a = ap.parse_args()
    head, take = read_take(a.csv)
    c = Contract.load()
    t = take["t"]
    dt = np.diff(t)
    print(f"{len(t)} rows over {t[-1] - t[0]} ms; tick median {np.median(dt):.0f} ms, max {dt.max()} ms")

    # the first walking tick: the first row at least HALT_MS after the recording started (gait_task's halt_in)
    best = None
    # (the gait's clock is read a millisecond or so before the row's timestamp, so try every ms, not only row times)
    for t0 in range(HALT_MS - 10, HALT_MS + 20):
        ticks, motion = replay(head, take, t0)
        sel = ticks[:, 0] >= 0
        err = np.abs(ticks[sel] - take["cmd"][sel]).max(axis=1)
        score = (float((err <= 1).mean()), float((err == 0).mean()))
        if best is None or score > best[0]:
            best = (score, t0, ticks, motion, sel)
    (within1, match), t0, ticks, motion, sel = best
    err = np.abs(ticks[sel] - take["cmd"][sel])
    print(f"replay from t={t0} ms: {sel.sum()} ticks, {100 * match:.1f} % match every servo exactly, "
          f"{100 * within1:.1f} % within 1 tick; max |tick error| {err.max()}, mean {err.mean():.4f}")

    # per keyframe hold: commanded and measured motion in policy coordinates
    def rel_of_ticks(x):
        k = x - 511.0
        return c.rel_from_servo_deg(k * 3100.0 / 10240.0 - head["trim_deg"])

    def rel_of_dd(dd):
        dd0 = 3100.0 - 511.0 * 3100.0 / 1024.0
        return c.rel_from_servo_deg((dd0 - dd) / 10.0 - head["trim_deg"])

    report = {"rows": int(len(t)), "replay_t0_ms": t0, "replay_ticks": int(sel.sum()), "exact_match_frac": match,
              "keyframes": []}
    seg0 = t0
    for n, k in enumerate(head["keyframes"]):
        settle = min(500, k["hold_ms"] // 2)                                       # skip up to 0.5 s of settling
        hs, he = seg0 + k["ramp_ms"] + settle, seg0 + k["ramp_ms"] + k["hold_ms"]
        seg0 += k["ramp_ms"] + k["hold_ms"]
        r = (t >= hs) & (t < he)
        cmd, act = rel_of_ticks(take["cmd"][r]), rel_of_dd(take["deg"][r])
        e = act - cmd
        row = {"kf": n, "vx_mm_s": k["vx"] * VX_MAX / 100, "vy_mm_s": k["vy"] * VY_MAX / 100,
               "wz_rad_s": k["wz"] * WZ_MAX / 100, "stand": k["stand"],
               "cmd_range_deg": np.degrees(cmd.max(0) - cmd.min(0)).round(1).tolist(),
               "act_range_deg": np.degrees(act.max(0) - act.min(0)).round(1).tolist(),
               "track_rms_deg": np.degrees(np.sqrt((e ** 2).mean(0))).round(2).tolist(),
               "cur_rms_ma": np.sqrt((take["cur"][r] ** 2).mean(0)).round(0).tolist()}
        report["keyframes"].append(row)
        abd = [0, 3, 6, 9]
        print(f"kf{n:<2} {'stand' if k['stand'] else 'walk '} vx {row['vx_mm_s']:+5.0f} vy {row['vy_mm_s']:+5.0f} "
              f"mm/s wz {row['wz_rad_s']:+4.1f} | cmd range abd "
              f"{np.mean([row['cmd_range_deg'][i] for i in abd]):4.1f} hip "
              f"{np.mean([row['cmd_range_deg'][i + 1] for i in abd]):4.1f} knee "
              f"{np.mean([row['cmd_range_deg'][i + 2] for i in abd]):4.1f} deg | tracking rms abd "
              f"{np.mean([row['track_rms_deg'][i] for i in abd]):4.2f} hip "
              f"{np.mean([row['track_rms_deg'][i + 1] for i in abd]):4.2f} knee "
              f"{np.mean([row['track_rms_deg'][i + 2] for i in abd]):4.2f} deg")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
