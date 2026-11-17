#!/usr/bin/env python
"""Per-joint excursions and simulated motor currents of a policy on a course (abduction gear load and the servo
current guard limit walking on terrain).

    python -B tools/joint_stats.py --policy policies/X --heading-loop [--terrain square_walls] --json out.json

Each env starts as in ``terrain_eval.py`` under noise on randomised plants. Per joint (contract order): the joint angle
relative to the stand, p1 / p99 and the extremes (deg); the commanded target relative to the stand, p1 / p99 (deg);
for BAM arms the motor current (|torque| / kt, A): p99, maximum, the share of samples above 0.5 and 1.0 A and the
longest run above 1.0 A (a current guard of > 1 A held 0.5 s). BAM's current scale against the robot is
unverified: read currents as relative between policies.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mp3_walk_policy.env_cfg import noise_profile  # noqa: E402
from mp3_walk_policy.heading import ANTIWINDUP, WZ_MAX, pi_step  # noqa: E402
from mp3_walk_policy.terrain_task import TerrainOptions  # noqa: E402

_spec = importlib.util.spec_from_file_location("terrain_eval", Path(__file__).resolve().parent / "terrain_eval.py")
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)
DT = 0.02


@torch.no_grad()
def run(args) -> dict:
    pol = te.TorchPolicy(Path(args.policy) / "policy.npz", "cuda:0")
    omni = bool(pol.meta.get("omni"))
    from mp3_walk_policy.arms import trained_arm
    arm = args.arm or trained_arm(args.policy) or "stock"
    opt = TerrainOptions(arm=arm, num_envs=args.envs, terrain=args.terrain, rows=1, noisy=True, noise_ramp=False,
                         noise_end=noise_profile("n2"), randomise=True, push_mps=0.0,
                         fixed_command=(args.vx, 0.0, 0.0), level=0, action_scale=pol.scale,
                         episode_s=args.seconds + 5.0, history=int(pol.meta.get("history", 0) or 0))
    env = te.build(opt, seed=args.seed)
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    n, steps = env.num_envs, int(round(args.seconds / DT))
    robot = env.scene["robot"]
    names = te.JOINT_ORDER
    jidx = [robot.joint_names.index(j) for j in names]
    stand = robot.data.default_joint_pos[:, jidx].clone()
    scale = torch.tensor(pol.scale, device=env.device)
    yaw0 = te.yaw_of(robot.data.root_link_quat_w)
    integ = torch.zeros(n, device=env.device)
    cmd = env.command_manager.get_term("twist")
    live = torch.ones(n, dtype=torch.bool, device=env.device)
    pos, tgt, cur = [], [], []
    run_over = torch.zeros(n, 12, device=env.device)
    longest = torch.zeros(n, 12, device=env.device)
    for _ in range(steps):
        if omni and args.heading_loop:
            err = te.wrap(yaw0 - te.yaw_of(robot.data.root_link_quat_w))
            wz, integ = pi_step(err, integ, DT, wz_max=args.heading_wz_max, antiwindup=args.heading_aw)
            cmd.vel_command_b[:, 2] = wz
        a = pol(obs["actor"])
        obs, _, term, trunc, _ = env.step(a)
        live &= ~(term | trunc)
        m = live[:, None]
        q = robot.data.joint_pos[:, jidx] - stand
        pos.append(torch.where(m, q, torch.full_like(q, float("nan"))))
        t = a.clamp(-1.0, 1.0) * scale
        tgt.append(torch.where(m, t, torch.full_like(t, float("nan"))))
        c = te.bam_currents(env)
        if c is not None:
            cur.append(torch.where(m, c, torch.full_like(c, float("nan"))))
            run_over = torch.where((c > 1.0) & m, run_over + DT, torch.zeros_like(run_over))
            longest = torch.maximum(longest, run_over)
    env.close()
    P = np.degrees(torch.stack(pos).cpu().numpy()).reshape(-1, 12)
    T = np.degrees(torch.stack(tgt).cpu().numpy()).reshape(-1, 12)
    C = torch.stack(cur).cpu().numpy().reshape(-1, 12) if cur else None
    out = {"policy": args.policy, "terrain": args.terrain, "envs": n, "seconds": args.seconds, "seed": args.seed,
           "vx": args.vx, "heading_loop": bool(omni and args.heading_loop), "heading_aw": args.heading_aw, "heading_wz_max": args.heading_wz_max, "joints": {}}
    for j, name in enumerate(names):
        r = {"pos_p1_deg": float(np.nanpercentile(P[:, j], 1)), "pos_p99_deg": float(np.nanpercentile(P[:, j], 99)),
             "pos_min_deg": float(np.nanmin(P[:, j])), "pos_max_deg": float(np.nanmax(P[:, j])),
             "target_p1_deg": float(np.nanpercentile(T[:, j], 1)), "target_p99_deg": float(np.nanpercentile(T[:, j], 99))}
        if C is not None:
            r.update(cur_p99_a=float(np.nanpercentile(C[:, j], 99)), cur_max_a=float(np.nanmax(C[:, j])),
                     over_0p5=float(np.nanmean(C[:, j] > 0.5)), over_1p0=float(np.nanmean(C[:, j] > 1.0)),
                     longest_over_1a_s=float(longest[:, j].max()))
        out["joints"][name] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--terrain", default="square_walls", choices=("square", "square_walls", "strip", "flat"))
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--heading-loop", action="store_true")
    ap.add_argument("--heading-aw", choices=ANTIWINDUP, default="cond", help="the loop's anti-windup (heading.pi_step; none = no anti-windup)")
    ap.add_argument("--heading-wz-max", type=float, default=WZ_MAX, help="rad/s: the loop's output limit")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = run(a)
    print(f"{'joint':12s} {'pos p1..p99 (min..max) deg':>30s} {'target p1..p99':>16s} {'I p99':>6s} {'I max':>6s} "
          f"{'>0.5A':>6s} {'>1A':>6s} {'run>1A':>7s}")
    for name, r in res["joints"].items():
        line = (f"{name:12s} {r['pos_p1_deg']:6.1f}..{r['pos_p99_deg']:5.1f} ({r['pos_min_deg']:6.1f}..{r['pos_max_deg']:5.1f})"
                f"   {r['target_p1_deg']:6.1f}..{r['target_p99_deg']:5.1f}")
        if "cur_p99_a" in r:
            line += (f" {r['cur_p99_a']:6.2f} {r['cur_max_a']:6.2f} {100 * r['over_0p5']:5.1f}% {100 * r['over_1p0']:5.2f}% "
                     f"{r['longest_over_1a_s']:6.2f}s")
        print(line)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
