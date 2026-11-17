#!/usr/bin/env python
"""The runner and the simulator, on the same inputs: do they build the same observation and send the same targets?

    python tools/runner_equivalence.py --policy policies/stock_v3n2_s3 \
        [--arm stock] [--vx 0.10] [--seconds 5] [--mount swap_xy] [--json out.json]

The policy walks in mjlab (nominal plant, clean observations, the chain delay fixed). After every policy step, the
step's four 200 Hz joint samples (the action term's history: 5, 10, 15 ms and the observation's instant) and the IMU
are written as STATE frames in the robot's units: positions as the AT32's integer deci-degrees, the accelerometer in
mg and the gyro in 0.1 deg/s, rotated into the sensor's frame by a mounting candidate. Two versions: the IMU exact
(floats) and quantised (int16, as the firmware sends). The runner's ``ObservationBuilder``, ``NumpyPolicy`` and
``ActionDecoder`` process them, and are compared with the simulator's observation, the action it applied and the dd
its decoder applied (the quantised target; comparing re-quantised dd would itself move ticks below 511). The gate
(exact IMU): every observation channel to 1e-5, actions to 1e-3, the applied targets equal. The quantised version is
reported, not gated: it sizes the effect of the firmware's integer IMU units.

A policy with the heading term (``contract.has_heading_obs``): the forward task has no heading loop, so both sides
observe the same error, the true yaw's from the start (``contract.heading_obs``), and the gate is unchanged; the
runner's ``HeadingLoop`` (open loop: its wz is not applied) integrates the frames' gyro alongside, and its observed
error's distance from the true one is reported (``heading_error``, not gated: the frames repeat one IMU sample per
step).
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

from mp3_walk_policy.contract import DD0, ActionDecoder, Contract         # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions                            # noqa: E402
from mp3_walk_policy.export import NumpyPolicy                             # noqa: E402
from mp3_walk_policy.gates import make_env                                 # noqa: E402
from mp3_walk_policy.mdp import servo_action                               # noqa: E402
from mp3_walk_policy.runner import IMU_MOUNTS, HeadingLoop, ObservationBuilder   # noqa: E402

G = 9.80665
SLICES = {"ang_vel": slice(0, 3), "gravity": slice(3, 6), "command": slice(6, 9), "pos": slice(9, 21),
          "vel": slice(21, 33), "prev_action": slice(33, 45)}


def dd_of_rel(c: Contract, rel: np.ndarray) -> np.ndarray:
    """The integer deci-degree the AT32 reports for a quantised sample (the inverse of mdp.measure_rel)."""
    sdeg = np.degrees(rel) * c.signs
    return np.rint(DD0 - 10.0 * (sdeg + c.offsets_deg)).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arm", default=None, help="the simulator's arm (default: the one the policy was trained in)")
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--mount", default="flip_x", choices=sorted(IMU_MOUNTS))
    ap.add_argument("--json")
    a = ap.parse_args()
    from mp3_walk_policy.arms import trained_arm
    a.arm = a.arm or trained_arm(a.policy) or "stock"
    c = Contract.load()
    pol = NumpyPolicy(Path(a.policy) / "policy.npz")
    R = IMU_MOUNTS[a.mount]
    env = make_env(TaskOptions(arm=a.arm, num_envs=1, noisy=False, randomise=False, noise_ramp=False,
                               vx_range=(a.vx, a.vx), episode_s=60.0,
                               action_scale=tuple(float(x) for x in pol.scale)))   # the policy's own units
    term = servo_action(env)
    obs_t, _ = env.reset()
    runners = {q: (ObservationBuilder(c, R), ActionDecoder(c, pol.scale)) for q in ("exact", "quantised")}
    # a policy with observation history: both sides stack their 45-float observations as the runner does
    # (contract.HistoryStacker, checked against mjlab's own history in tests/test_terrain.py)
    hist_n = int(pol.meta.get("history", 0) or 0)
    from mp3_walk_policy.contract import HistoryStacker, has_heading_obs, heading_obs
    he = has_heading_obs(pol.meta)
    loops = {q: HeadingLoop() for q in runners} if he else {}
    yaw_start = float(env.scene["robot"].data.heading_w[0])
    stacks = {q: HistoryStacker(hist_n, he) for q in ["sim", *runners]} if hist_n else None
    stacked = (lambda q, o: stacks[q](o)) if hist_n else (lambda q, o: o)
    tick = 0

    def frames(samples_rel, gyro_b, acc_b, quantise):
        nonlocal tick
        out = []
        for j, rel in enumerate(samples_rel):
            gyr = R.T @ (np.degrees(gyro_b) * 10.0)              # body -> sensor, 0.1 deg/s
            acc = R.T @ (acc_b / G * 1000.0)                     # mg
            if quantise:
                gyr, acc = np.rint(gyr), np.rint(acc)
            out.append(SimpleNamespace(tick=tick + j, pos_dd=dd_of_rel(c, rel), gyr_ddps=gyr, acc_mg=acc))
        return out

    def imu_body():
        from mjlab.envs.mdp import builtin_sensor
        g = builtin_sensor(env, sensor_name="robot/imu_ang_vel")[0].cpu().numpy().astype(float)
        acc = builtin_sensor(env, sensor_name="robot/imu_lin_acc")[0].cpu().numpy().astype(float)
        return g, acc

    hist0 = term.hist[0].cpu().numpy().astype(float)
    g0, a0 = imu_body()
    for q, (ob, dec) in runners.items():
        for f in frames([hist0[-1]], g0, a0, q == "quantised"):
            ob.push(f)
            if he:
                loops[q].push(ob.gyro_body(f), ob.gravity_body(f))
    tick += 1
    err = {q: {k: 0.0 for k in list(SLICES) + ["action", "target_diff_deg", "target_mismatch_steps"]
               + (["heading_error"] if he else [])} for q in runners}
    n = int(round(a.seconds / 0.02))
    for k in range(n):
        o_sim = obs_t["actor"][0].cpu().numpy().astype(float)
        e_sim = None
        if he:          # the true yaw's error back to the start, as training's loop would see it (no bias, no noise)
            yaw = float(env.scene["robot"].data.heading_w[0])
            e_sim = heading_obs((yaw_start - yaw + np.pi) % (2 * np.pi) - np.pi)
            o_sim = np.append(o_sim, e_sim)
        act_sim = pol(stacked("sim", o_sim.astype(np.float32)))
        for q, (ob, dec) in runners.items():
            o_run = ob.observe(a.vx, dec.prev_action, e_sim)
            if he:
                loops[q].wz()
                err[q]["heading_error"] = max(err[q]["heading_error"], abs(heading_obs(loops[q].err) - e_sim))
            for name, sl in SLICES.items():
                err[q][name] = max(err[q][name], float(np.abs(o_run[sl] - o_sim[sl]).max()))
            act_run = pol(stacked(q, o_run))
            err[q]["action"] = max(err[q]["action"], float(np.abs(act_run - act_sim).max()))
            err[q].setdefault("_rel", []).append(dec(act_run)["rel"])
        obs_t, *_ = env.step(torch.from_numpy(act_sim).to(env.device)[None, :])
        rel_sim = term.prev_rel[0].cpu().numpy().astype(float)        # the simulator's applied (quantised) target
        hist = term.hist[0].cpu().numpy().astype(float)
        gb, ab = imu_body()
        for q, (ob, dec) in runners.items():
            diff = np.degrees(np.abs(err[q]["_rel"][-1] - rel_sim))
            err[q]["target_diff_deg"] = max(err[q]["target_diff_deg"], float(diff.max()))
            err[q]["target_mismatch_steps"] += int(diff.max() > 1e-4)
            for f in frames(hist[-4:], gb, ab, q == "quantised"):
                ob.push(f)
                if he:
                    loops[q].push(ob.gyro_body(f), ob.gravity_body(f))
        tick += 4
    res = {"policy": str(a.policy), "arm": a.arm, "vx": a.vx, "steps": n, "mount": a.mount, "history": hist_n,
           "heading_obs": he}
    for q in runners:
        err[q].pop("_rel")
        res[q] = err[q]
    # the quantised IMU: 0.1 deg/s on the gyro, 1 mg on the accelerometer; a tick's difference in an action (from the
    # IMU's quantisation) shows in the previous action as one tick (0.30 deg) over the abduction's 0.10 rad scale
    bars = {"exact": {k: 1e-5 for k in SLICES}}
    # the gate is the exact version; the quantised one sizes what the firmware's int16 IMU units change (information)
    res["pass"] = bool(all(res["exact"][k] <= bars["exact"][k] for k in SLICES) and res["exact"]["action"] <= 1e-3
                       and res["exact"]["target_diff_deg"] <= 1e-4)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
