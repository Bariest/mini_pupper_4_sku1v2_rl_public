#!/usr/bin/env python
"""A policy's observations in simulation, step by step, as the robot's runner logs them (``<take>.policy.npz``'s
``obs``): clean observations, the nominal plant, one environment from the stand, at each commanded speed.

    python tools/sim_obs.py --policy policies/stock_v3_s4 \
        --vx 0.10,0.15,0.20 --steps 350 --out runs/sim_obs_stock_v3_s4.npz

Writes per speed ``<vx>_obs`` (steps, 45), ``<vx>_x`` (m) and ``<vx>_yaw`` (deg): the body after each step.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mp3_walk_policy.env_cfg import TaskOptions                        # noqa: E402
from mp3_walk_policy.export import NumpyPolicy                         # noqa: E402
from mp3_walk_policy.gates import make_env                             # noqa: E402
from mp3_walk_policy.mdp import _yaw                                   # noqa: E402


@torch.no_grad()
def run(policy, arm, vx, steps, chain_ms=None):
    opt = TaskOptions(arm=arm, num_envs=1, noisy=False, randomise=False, noise_ramp=False, vx_range=(vx, vx),
                      episode_s=60.0)
    if chain_ms is not None:
        opt.chain_ms = (chain_ms, chain_ms)
    env = make_env(opt)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    rec = {"obs": [], "x": [], "yaw": []}
    for _ in range(steps):
        o = obs["actor"][0].cpu().numpy()
        rec["obs"].append(o)
        a = torch.tensor(policy(o[None]), dtype=torch.float32, device=env.device)
        obs, _, _, _, _ = env.step(a)
        p = robot.data.root_link_pos_w - env.scene.env_origins
        rec["x"].append(float(p[0, 0]))
        rec["yaw"].append(float(np.degrees(float(_yaw(robot.data.root_link_quat_w)[0]))))
    env.close()
    return {k: np.array(v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="an exported policy directory (policy.npz)")
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--vx", default="0.10,0.15,0.20")
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pol = NumpyPolicy(Path(a.policy) / "policy.npz")
    out = {}
    for v in a.vx.split(","):
        r = run(pol, a.arm, float(v), a.steps)
        for k, x in r.items():
            out[f"{float(v):.2f}_{k}"] = x
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
