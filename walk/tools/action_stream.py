#!/usr/bin/env python
"""Record a policy's raw action stream in simulation, for the runner's ``replay`` (the open-loop air test on SKU2).

    python tools/action_stream.py --policy policies/<name> \
        [--arm stock] [--vx 0.10] [--seconds 10] --out data/streams/<name>_0p10.npz

The policy walks on the nominal plant with clean observations from the stand (as the runner starts); its raw actions
are saved with what the simulator applied and measured (the decoded targets and the joint positions, policy
coordinates), so the robot's replay can be compared with the simulation step by step.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

from mp3_walk_policy.env_cfg import TaskOptions                            # noqa: E402
from mp3_walk_policy.export import NumpyPolicy                             # noqa: E402
from mp3_walk_policy.gates import make_env                                 # noqa: E402
from mp3_walk_policy.mdp import servo_action                               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arm", default=None, help="the simulator's arm (default: the one the policy was trained in)")
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mp3_walk_policy.arms import trained_arm
    a.arm = a.arm or trained_arm(a.policy) or "stock"
    path = Path(a.policy) / "policy.npz"
    pol = NumpyPolicy(path)
    env = make_env(TaskOptions(arm=a.arm, num_envs=1, noisy=False, randomise=False, noise_ramp=False,
                               vx_range=(a.vx, a.vx), episode_s=a.seconds + 5.0))
    term = servo_action(env)
    obs, _ = env.reset()
    raw, target, pos = [], [], []
    for _ in range(int(round(a.seconds / 0.02))):
        act = pol(obs["actor"][0].cpu().numpy())
        raw.append(act)
        obs, *_ = env.step(torch.from_numpy(act).to(env.device)[None, :])
        target.append(term.prev_rel[0].cpu().numpy())
        pos.append(term.true_rel()[0].cpu().numpy())
    meta = {"policy": str(path.resolve()), "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arm": a.arm, "vx": a.vx, "seconds": a.seconds, "plant": "nominal, clean observations, floor"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, raw_actions=np.array(raw, dtype=np.float32), target_rel=np.array(target),
             pos_rel=np.array(pos), meta=json.dumps(meta))
    print("STREAM", a.out, len(raw), "steps", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
