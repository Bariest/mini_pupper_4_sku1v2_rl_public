#!/usr/bin/env python
"""Render a policy walking to an mp4 (offscreen, MUJOCO_GL=egl).

    MUJOCO_GL=egl python scripts/render.py \
        --ckpt policies/stock_v3n2_s3 --vx 0.10 --seconds 10 --out runs/video/stock_v3n2_s3.mp4
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio                                              # noqa: E402
import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                  # noqa: E402

from mp3_walk_policy.env_cfg import TaskOptions, make_env_cfg, omni_options  # noqa: E402
from mp3_walk_policy.export import NumpyPolicy                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a policies/<name> directory (policy.npz)")
    ap.add_argument("--arm", default=None, help="the simulator's arm (default: the one the policy was trained in)")
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--vy", type=float, default=0.0, help="m/s, + = left (omni policies)")
    ap.add_argument("--wz", type=float, default=0.0, help="rad/s, + = turn left (omni policies)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mp3_walk_policy.arms import trained_arm
    a.arm = a.arm or trained_arm(a.ckpt) or "stock"
    pol = NumpyPolicy(Path(a.ckpt) / "policy.npz")
    kw = dict(arm=a.arm, num_envs=1, noisy=False, randomise=False, noise_ramp=False, episode_s=a.seconds + 5,
              visuals=True)
    if pol.meta.get("omni"):
        opt = omni_options(fixed_command=(a.vx, a.vy, a.wz), **kw)
    elif a.vy or a.wz:
        raise SystemExit("--vy / --wz need an omni policy")
    else:
        opt = TaskOptions(vx_range=(a.vx, a.vx), **kw)
    cfg = make_env_cfg(opt)
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    for k in range(int(a.seconds / 0.02)):
        act = torch.from_numpy(pol(obs["actor"].cpu().numpy())).to(env.device)
        obs, *_ = env.step(act)
        frames.append(env.render())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, [np.asarray(f) for f in frames], fps=50)
    print("VIDEO", a.out, len(frames), "frames", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
