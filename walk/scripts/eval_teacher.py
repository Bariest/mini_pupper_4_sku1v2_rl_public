#!/usr/bin/env python
"""The teacher (the firmware trot at the policy rate) played open-loop in the SKU2 environment.

    python scripts/eval_teacher.py [--arm stock] [--json out.json]

Nominal plant (no randomisation; the chain delay at its mid-range) at each speed, then the randomised plant at
0.10 m/s. The teacher's numbers are the clone gate's reference.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3_walk_policy.env_cfg import TaskOptions                     # noqa: E402
from mp3_walk_policy.gates import make_env, metrics, rollout, teacher_fn  # noqa: E402
from mp3_walk_policy.teacher import TorchTeacher                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--speeds", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"arm": a.arm, "nominal": {}, "randomised": {}}
    for vx in [float(v) for v in a.speeds.split(",")]:
        env = make_env(TaskOptions(arm=a.arm, num_envs=4, noisy=False, randomise=False, noise_ramp=False,
                                   vx_range=(vx, vx), episode_s=60.0))
        m = metrics(rollout(env, teacher_fn(TorchTeacher(env.device), env)))
        env.close()
        out["nominal"][f"{vx:.2f}"] = m
        print(f"nominal vx {vx:.2f}: {m}")
    env = make_env(TaskOptions(arm=a.arm, num_envs=256, noisy=False, randomise=True, noise_ramp=False,
                               vx_range=(0.10, 0.10), episode_s=60.0))
    m = metrics(rollout(env, teacher_fn(TorchTeacher(env.device), env)))
    env.close()
    out["randomised"]["0.10"] = m
    print(f"randomised vx 0.10: {m}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
