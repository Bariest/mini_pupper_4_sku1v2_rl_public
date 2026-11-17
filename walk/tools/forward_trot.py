#!/usr/bin/env python
"""The firmware's trot, as the robot would execute it, predicted in the SKU2 environment: a check of the
simulated plant against the same trot on the robot (distance against tape).

    python tools/forward_trot.py --vx 100 [--seconds 10] [--json out]

The stream is minipupperesp's own gait (``firmware_gait``, the robot's IK and calibration offsets): one target
per 15 ms gait tick, turned into the dd the robot receives and back into policy coordinates. It starts after a 2 s
hold at the stand and is applied on the 1 ms physics grid with a constant chain delay (default 13 ms, the middle of
8-19 ms) on top of the arm's own latency. The body is free on the floor. Reported: the distance travelled (world x;
the episode starts heading +x), the lateral drift, the yaw, and the speed over the trot.

``--write-stream FILE.npz`` writes the same trot as a command file for ``mp3link stream`` (rows of 15 ms ``delta_dd``
about the stand, ``period_s``), for the same trot on the robot: ``mp3link stream`` ramps to the first row and holds
it 1 s, so the file starts with 9 s at the stand (10 s in all), then the trot, then a 1 s ramp back to the stand and
4 s there (mp3link adds its own ramp and 1 s hold: 5 s or more), as the runner's floor takes. Every trot row is a
firmware tick's dd.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3_walk_policy.contract import Contract                         # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions                       # noqa: E402
from mp3_walk_policy.gates import make_env                            # noqa: E402
from mp3_walk_policy.mdp import _yaw, servo_action                    # noqa: E402
from mp3_walk_policy.teacher import _gait_module                      # noqa: E402

TICK_S = 0.015


def firmware_stream(vx_mm_s, seconds, c):
    fg = _gait_module()
    g = fg.Gait(c.offsets_deg)
    g.reset()
    n = int(round(seconds / TICK_S))
    ticks = np.array([g.ticks(g.step(vx_mm_s)) for _ in range(n)])
    return c.rel_from_dd(c.dd_from_ticks(ticks))                    # (n, 12) policy coordinates


def write_stream(path, vx_mm_s, seconds, hold_before=9.0, ramp_after=1.0, hold_after=4.0):
    c = Contract.load()
    fg = _gait_module()
    g = fg.Gait(c.offsets_deg)
    g.reset()
    n = int(round(seconds / TICK_S))
    dd = np.array([c.dd_from_ticks(g.ticks(g.step(vx_mm_s))) for _ in range(n)], dtype=float)
    stand = np.asarray(c.stand_dd, dtype=float)
    trot = dd - stand
    nr = int(round(ramp_after / TICK_S))
    s = np.array([(k + 1) / nr for k in range(nr)])
    ramp = trot[-1] * (1.0 - s * s * (3.0 - 2.0 * s))[:, None]
    rows = np.concatenate([np.zeros((int(round(hold_before / TICK_S)), 12)), trot, np.rint(ramp),
                           np.zeros((int(round(hold_after / TICK_S)), 12))])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    note = (f"firmware forward trot {vx_mm_s:.0f} mm/s for {seconds:.0f} s (minipupperesp's gait via "
            f"firmware_gait, the contract's offsets); {hold_before:.0f} s stand before, {ramp_after:.0f} s ramp and "
            f"{hold_after:.0f} s stand after; rows {len(rows)} at {TICK_S * 1000:.0f} ms")
    np.savez(path, delta_dd=rows.astype(np.int64), period_s=TICK_S, stand_dd=stand.astype(np.int64), note=note,
             trot_rows=np.array([int(round(hold_before / TICK_S)), int(round(hold_before / TICK_S)) + n]))
    print(note, "->", path)


def run(vx_mm_s, seconds, chain_ms, arm, hold_s=2.0, resample_ms=0):
    c = Contract.load()
    rel = firmware_stream(vx_mm_s, seconds, c)
    env = make_env(TaskOptions(arm=arm, num_envs=1, noisy=False, randomise=False, noise_ramp=False,
                               episode_s=1.0e6))
    env.reset()
    robot = env.scene["robot"]
    term = servo_action(env)
    lat = env.cfg.events["command_delay"].params["latency_steps"]
    for act, la in zip(robot.actuators, lat):
        act.set_lags(torch.full((1,), int(la + chain_ms), dtype=torch.long, device=env.device))
    rel_t = torch.tensor(rel, dtype=torch.float32, device=env.device)
    n_hold, n_trot = int(round(hold_s * 1000)), int(round(seconds * 1000))
    xs, ys, yaws = [], [], []
    for i in range(n_hold + n_trot):
        j = i - n_hold
        if resample_ms:                      # the stream as a policy at 1/resample_ms would hold it
            j = (j // resample_ms) * resample_ms
        r = torch.zeros(1, 12, device=env.device) if i < n_hold else \
            rel_t[min(j // int(TICK_S * 1000), len(rel_t) - 1)][None]
        model = term.tc.stand + r
        robot.set_joint_position_target(model[:, term._jidx], joint_ids=term._tj)
        robot.set_tendon_len_target(model[:, term._kidx], tendon_ids=term._tt)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        if i % 20 == 0:
            env.sim.forward()
            p = robot.data.root_link_pos_w[0] - env.scene.env_origins[0]
            xs.append(float(p[0])), ys.append(float(p[1])), yaws.append(float(_yaw(robot.data.root_link_quat_w)[0]))
    env.close()
    k0 = int(hold_s * 50)
    return {"vx_mm_s": vx_mm_s, "seconds": seconds, "chain_ms": chain_ms, "arm": arm,
            "distance_m": xs[-1] - xs[k0], "speed_m_s": (xs[-1] - xs[k0]) / seconds,
            "lateral_m": ys[-1] - ys[k0], "yaw_deg": float(np.degrees(yaws[-1] - yaws[k0]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", default="100", help="mm/s, comma-separated")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--chain-ms", type=int, default=13)
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--resample-ms", type=int, default=0, help="hold the stream at this period (20: the policy rate)")
    ap.add_argument("--json", default="")
    ap.add_argument("--write-stream", default="", help="write the trot as an mp3link stream file instead")
    a = ap.parse_args()
    if a.write_stream:
        write_stream(a.write_stream, float(a.vx), a.seconds)
        return
    res = [run(float(v), a.seconds, a.chain_ms, a.arm, resample_ms=a.resample_ms) for v in a.vx.split(",")]
    for r in res:
        print(f"vx {r['vx_mm_s']:+.0f} mm/s: {r['distance_m'] * 1000:+.0f} mm in {r['seconds']:.0f} s "
              f"({r['speed_m_s']:+.3f} m/s), lateral {r['lateral_m'] * 1000:+.0f} mm, yaw {r['yaw_deg']:+.1f} deg")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
