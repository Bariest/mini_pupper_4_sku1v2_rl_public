#!/usr/bin/env python
"""The IMU's mounting from a tilt test: which of the four candidates (sensor -> body, z flipped) fits the robot?

    python3 tools/tilt_mount.py LOG.npz [--write]        # a runner `imu --log` or `mp3link ping --log` capture
    python3 tools/tilt_mount.py --selftest

The protocol (motors idle, the robot held by hand): about 3 s still; nose down about 30 deg, hold 2 s, back; right
side down about 30 deg, hold 2 s, back; a turn to the left (counterclockwise seen from above) of about 90 deg.
Body axes: x forward, y left, z up (the model's).

The tool low-passes the accelerometer (0.1 s), finds the first two excursions beyond 20 deg (the nose and the side
tilts), and checks each candidate against all five signs: gravity in the body frame (nose down: g_x > 0.3 dominant;
right side down: g_y < -0.3 dominant), and the gyro, its bias from the still start, integrated over each tilt's way
down (pitch rate y > 0 nose down, roll rate x > 0 right side down) and after the side tilt (yaw z > 0.5 rad, left).
``--write`` stores the single fitting candidate in ``mp3_walk_policy/data/imu_mount.json`` with the log's hash.
Numpy only (runs on the Pi or the workstation).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mp3_walk_policy.runner import IMU_MOUNTS                              # noqa: E402

DT = 0.005
IMU_OK = 1 << 5


def lowpass(x, tau=0.1):
    y = np.empty_like(x)
    y[0] = x[0]
    k = DT / tau
    for i in range(1, len(x)):
        y[i] = y[i - 1] + k * (x[i] - y[i - 1])
    return y


def segments(mask, min_len):
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def analyse(acc_mg, gyr_ddps):
    acc = np.asarray(acc_mg, dtype=float)
    gyr = np.asarray(gyr_ddps, dtype=float) * 0.1 * math.pi / 180.0
    al = lowpass(acc)
    tilt = np.degrees(np.arccos(np.clip(-al[:, 2] / np.linalg.norm(al, axis=1), -1, 1)))
    segs = segments(tilt > 20.0, int(0.3 / DT))
    if len(segs) < 2:
        return {"error": f"found {len(segs)} tilts beyond 20 deg, need 2 (nose, then side)"}
    still = tilt < 5.0
    n0 = np.argmax(~still[: int(5 / DT)]) if not still[: int(5 / DT)].all() else int(5 / DT)
    bias = gyr[: max(n0 - int(0.2 / DT), int(0.5 / DT))].mean(axis=0)
    (a0, a1), (b0, b1) = segs[0], segs[1]

    def way_down(s0, s1):              # from the last still frame before the tilt to its peak
        pre = s0
        while pre > 0 and tilt[pre] > 5.0:
            pre -= 1
        return pre, s0 + int(np.argmax(tilt[s0:s1]))

    res = {"tilts": [{"frames": [int(a0), int(a1)], "peak_deg": float(tilt[a0:a1].max())},
                     {"frames": [int(b0), int(b1)], "peak_deg": float(tilt[b0:b1].max())}],
           "gyro_bias_rad_s": bias.tolist(), "candidates": {}}
    nd, sd = way_down(a0, a1), way_down(b0, b1)
    for name, R in IMU_MOUNTS.items():
        g_nose = -(R @ al[a0:a1].mean(0)); g_nose /= np.linalg.norm(g_nose)
        g_side = -(R @ al[b0:b1].mean(0)); g_side /= np.linalg.norm(g_side)
        w = (gyr - bias) @ R.T
        pitch = float(w[nd[0]:nd[1] + 1, 1].sum() * DT)
        roll = float(w[sd[0]:sd[1] + 1, 0].sum() * DT)
        yaw = float(w[b1:, 2].sum() * DT)
        checks = {"nose_gx": bool(g_nose[0] > 0.3 and abs(g_nose[0]) > abs(g_nose[1])),
                  "side_gy": bool(g_side[1] < -0.3 and abs(g_side[1]) > abs(g_side[0])),
                  "nose_pitch_rate": pitch > 0.2, "side_roll_rate": roll > 0.2, "left_yaw": yaw > 0.5}
        res["candidates"][name] = {"gravity_nose": g_nose.round(3).tolist(), "gravity_side": g_side.round(3).tolist(),
                                   "pitch_rad": round(pitch, 3), "roll_rad": round(roll, 3), "yaw_rad": round(yaw, 3),
                                   "checks": checks, "fits": all(checks.values())}
    fits = [k for k, v in res["candidates"].items() if v["fits"]]
    res["fits"] = fits
    return res


def synth(R, seed=0):
    """A tilt session as a candidate mounting R would record it (sensor frame), with vibration-free noise."""
    rng = np.random.default_rng(seed)

    def rot(axis, ang):
        c, s = math.cos(ang), math.sin(ang)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    plan = [("x", 0.0, 3.0), ("y", 0.52, 1.0), ("y", 0.52, 2.0), ("y", 0.0, 1.0), ("y", 0.0, 1.0),
            ("x", 0.52, 1.0), ("x", 0.52, 2.0), ("x", 0.0, 1.0), ("x", 0.0, 1.0), ("z", 1.57, 2.0), ("z", 1.57, 1.0)]
    acc, gyr, ang = [], [], {"x": 0.0, "y": 0.0, "z": 0.0}
    for axis, target, secs in plan:
        n = int(secs / DT)
        start = ang[axis]
        for k in range(n):
            new = start + (target - start) * (k + 1) / n
            rate = (new - ang[axis]) / DT
            ang[axis] = new
            Rwb = rot("z", ang["z"]) @ rot("y", ang["y"]) @ rot("x", ang["x"])
            f_body = Rwb.T @ np.array([0.0, 0.0, 1000.0])                 # specific force, mg
            w_body = np.zeros(3)
            w_body["xyz".index(axis)] = rate
            acc.append(R.T @ f_body + rng.normal(0, 3, 3))
            gyr.append(R.T @ np.degrees(w_body) * 10.0 + rng.normal(0, 2, 3) + np.array([5.0, -3.0, 2.0]))
    return np.array(acc), np.array(gyr)


def selftest():
    for name, R in IMU_MOUNTS.items():
        acc, gyr = synth(R)
        res = analyse(acc, gyr)
        assert res.get("fits") == [name], (name, res.get("fits"), res.get("error"))
    print("selftest: each candidate recovered from its own synthetic tilt session")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    d = np.load(a.log)
    m = min(len(d["acc_mg"]), len(d["gyr_ddps"]), len(d["flags"]))
    ok = ((d["flags"][:m] & IMU_OK) > 0) & np.any(d["acc_mg"][:m] != 0, axis=1)
    res = analyse(d["acc_mg"][:m][ok], d["gyr_ddps"][:m][ok])
    print(json.dumps(res, indent=1))
    if a.write:
        if len(res.get("fits", [])) != 1:
            raise SystemExit(f"not written: {len(res.get('fits', []))} candidates fit")
        out = REPO / "mp3_walk_policy" / "data" / "imu_mount.json"
        out.write_text(json.dumps({"mount": res["fits"][0], "R_sensor_to_body": IMU_MOUNTS[res["fits"][0]].tolist(),
                                   "log": str(Path(a.log).name),
                                   "log_sha256": hashlib.sha256(Path(a.log).read_bytes()).hexdigest(),
                                   "analysis": res["candidates"][res["fits"][0]]}, indent=1))
        print("written", out)


if __name__ == "__main__":
    main()
