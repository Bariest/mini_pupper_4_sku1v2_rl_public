#!/usr/bin/env python
"""The gait fundamental and speed of a training run, checkpoint by checkpoint: when did a policy leave the firmware
trot's rhythm?

    python tools/f0_trajectory.py runs/ppo/<run> [<run> ...] \
        [--every 50] [--vx 0.10 0.20] [--json out.json]

Each checkpoint is rolled out 10 s on the nominal plant with clean observations (one environment per speed: the
evaluation is deterministic), in the arm the run trained in (its config.json), and scored with ``gates.metrics``.
Reported per checkpoint and speed: f0, its distance from the firmware trot's 2.78 Hz, the speed, falls.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions                            # noqa: E402
from mp3_walk_policy.gates import FIRMWARE_F0_HZ, make_env, metrics, policy_fn, rollout  # noqa: E402


def checkpoints(run: Path, every: int) -> list[tuple[int, Path]]:
    found = sorted((int(re.search(r"model_(\d+)\.pt", p.name).group(1)), p) for p in run.glob("model_*.pt"))
    last = found[-1][0]
    return [(i, p) for i, p in found if i % every == 0 or i == last]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--vx", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--json")
    a = ap.parse_args()
    envs, out = {}, {}
    for run in map(Path, a.runs):
        arm = json.loads((run / "config.json").read_text())["task"]["arm"]
        rows = []
        for it, ck in checkpoints(run, a.every):
            actor = distill.load_actor(ck)
            row = {"iter": it}
            for vx in a.vx:
                key = (arm, vx)
                if key not in envs:
                    envs[key] = make_env(TaskOptions(arm=arm, num_envs=1, noisy=False, randomise=False,
                                                     noise_ramp=False, vx_range=(vx, vx), episode_s=60.0))
                m = metrics(rollout(envs[key], policy_fn(actor)))
                f0 = m.get("f0_hz", float("nan"))
                row[f"{vx:.2f}"] = {"f0_hz": f0, "df0_hz": f0 - FIRMWARE_F0_HZ, "speed": m.get("speed", float("nan")),
                                    "fall": m["fall_frac"]}
            rows.append(row)
            print(run.name, it, "  ".join(f"vx {k}: f0 {v['f0_hz']:.2f} v {v['speed']:.3f} fall {v['fall']:.0f}"
                                          for k, v in row.items() if k != "iter"), flush=True)
        out[run.name] = {"arm": arm, "rows": rows}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
