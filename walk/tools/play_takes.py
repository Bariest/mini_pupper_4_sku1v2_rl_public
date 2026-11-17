#!/usr/bin/env python
"""Play the recorded takes' keyframe schedules (``data/takes_stops.json``: walks in eight directions, turns in place
at 1.2 rad/s, stops between them) in simulation: a policy (``--ckpt``, a run dir, a .pt or an exported policy dir) or
the firmware itself open loop (``--teacher``). Per keyframe hold: the travel along / across the command or the
drift, the yaw rate, the feet's lift, and the stop gates (``gates.stop_gate``).

    python -B tools/play_takes.py --ckpt policies/bam_t5ub06_omni_v0 --arm bam_t5ub06 [--n 2] [--randomise]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mp3_walk_policy import distill                                     # noqa: E402
from mp3_walk_policy.gates import print_takes, takes_eval               # noqa: E402


def npz_actor(path: Path, device: str = "cuda:0"):
    """An exported ``policy.npz`` as the training actor (the same layers, ELU between them)."""
    import torch
    d = np.load(path, allow_pickle=False)
    n = int(d["n_layers"])
    net = distill.mlp(d["W0"].shape[1], d[f"W{n - 1}"].shape[0])
    sd = {}
    for i in range(n):
        sd[f"{2 * i}.weight"], sd[f"{2 * i}.bias"] = torch.from_numpy(d[f"W{i}"]), torch.from_numpy(d[f"b{i}"])
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def load(ckpt: str):
    """A run dir (its last model_<k>.pt), a .pt, or an exported policy dir (its policy.npz)."""
    p = Path(ckpt)
    if p.is_dir():
        pts = sorted(p.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
        if pts:
            return distill.load_actor(pts[-1])
        return npz_actor(p / "policy.npz")
    return distill.load_actor(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--arm", default="bam_t5ub06")
    ap.add_argument("--n", type=int, default=1, help="environments per take")
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    actor = None if a.teacher else load(a.ckpt)
    res = takes_eval(a.arm, actor, a.n, a.randomise)
    print_takes(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
