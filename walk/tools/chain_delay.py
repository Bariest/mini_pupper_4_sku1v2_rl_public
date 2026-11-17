#!/usr/bin/env python
"""Policy-to-servo delay of ``runner run`` logs, measured on the ESP32's clock.

    python -B tools/chain_delay.py LOG [...] [--expect 10] [--json out.json]     # LOG = path without .npz / .policy.npz

For each policy step: from the time of the STATE frame whose joint positions the step observed to the time of the
exchange that delivered the step's targets (the first STATE whose ``seq`` echoes the step's SET_ALL; that exchange's
time is the next frame's ``t_us``). This is the delay the simulator draws per episode (``mdp.reset_command_delay``),
on top of each actuator model's own latency. Also reported: the IMU sample's age at delivery, and the fraction of
steps more than 0.5 ms from ``--expect`` or never echoed within 10 frames (the first and last steps are left out).
Numpy only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ECHO_FRAMES = 10


def u32_unwrap(x):
    x = x.astype(np.int64)
    return x + np.concatenate([[0], np.cumsum(np.diff(x) < -(1 << 31))]) * (1 << 32)


def take_chain(stem):
    d = np.load(f"{stem}.npz", allow_pickle=True)
    p = np.load(f"{stem}.policy.npz", allow_pickle=True)
    tick = d["tick"].astype(np.int64)
    t_us, imu_us, seq = u32_unwrap(d["t_us"]), u32_unwrap(d["imu_t_us"]), d["seq"].astype(np.int64) & 0xFFFF
    idx = {int(t): i for i, t in enumerate(tick)}
    joint, imu = [], []
    for t, s in zip(p["tick"], p["seq"]):
        k = idx.get(int(t))
        m = None
        if k is not None:
            hits = np.nonzero(seq[k:k + ECHO_FRAMES] == (int(s) & 0xFFFF))[0]
            if len(hits) and k + hits[0] + 1 < len(t_us):
                m = k + int(hits[0])
        if m is None:
            joint.append(np.nan)
            imu.append(np.nan)
            continue
        joint.append((t_us[m + 1] - t_us[k]) / 1000.0)
        imu.append((t_us[m + 1] - imu_us[k]) / 1000.0)
    return np.asarray(joint), np.asarray(imu)


def summarise(stem, expect):
    joint, imu = take_chain(stem)
    joint, imu = joint[1:-1], imu[1:-1]
    ok = np.isfinite(joint)
    off = ~ok | (np.abs(np.where(ok, joint, 0.0) - expect) > 0.5)
    return {"steps": int(len(joint)), "joint_ms_median": float(np.nanmedian(joint)),
            "joint_ms_p1_p99": [float(np.nanpercentile(joint, 1)), float(np.nanpercentile(joint, 99))],
            "imu_age_ms_median": float(np.nanmedian(imu)), "not_echoed": int((~ok).sum()),
            "off_frac": float(off.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--expect", type=float, default=10.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for stem in a.stems:
        stem = str(stem).removesuffix(".policy.npz").removesuffix(".npz")
        if not (Path(f"{stem}.npz").exists() and Path(f"{stem}.policy.npz").exists()):
            print(f"{Path(stem).name}: no link or policy log")
            continue
        r = res[Path(stem).name] = summarise(stem, a.expect)
        print(f"{Path(stem).name}: chain {r['joint_ms_median']:.2f} ms (p1 {r['joint_ms_p1_p99'][0]:.2f}, p99 "
              f"{r['joint_ms_p1_p99'][1]:.2f}), IMU age {r['imu_age_ms_median']:.2f} ms, off {100 * r['off_frac']:.2f} %"
              f" of {r['steps']}{'  more than 1 % off' if r['off_frac'] > 0.01 else ''}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"protocol": __doc__, "expect_ms": a.expect, "takes": res}, indent=1))


if __name__ == "__main__":
    main()
