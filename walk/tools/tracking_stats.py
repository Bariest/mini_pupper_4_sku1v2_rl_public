#!/usr/bin/env python
"""Joint tracking error and simulated currents by phase on a course, for comparison with the robot's logs (the same
definitions applied to a take's log):

- error = the target sent at control step k (the servo action's target, servo coordinates, relative to the stand;
  the knee is its tendon drive) minus the joint position 20 ms later (one control step), |error| in degrees;
- phases from the body pitch (gravity averaged over 1 s; nose up >= 4 deg for >= 0.1 s): "approach" before the first
  nose-up, "edge" from its onset to the end of the last nose-up, "square" after it, until the trial ends (a fall, the
  goal with --stop-at-goal, or the end); "all" = every live step;
- p50 and p95 of |error| per joint group (abduction, hip, knee) and phase; for BAM actuator models the current (|torque| / kt, A)
  p50 and p99.

    python -B tools/tracking_stats.py --policy policies/X --heading-loop --arm stock --terrain square_walls \
        --goal 0.65 --stop-at-goal --seed 11 --foot-solref 0.01 1 --json out.json      # or --teacher for the firmware trot
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mp3_walk_policy.env_cfg import noise_profile  # noqa: E402
from mp3_walk_policy.heading import ANTIWINDUP, WZ_MAX, pi_step  # noqa: E402
from mp3_walk_policy.mdp import servo_action  # noqa: E402
from mp3_walk_policy.terrain_task import TerrainOptions  # noqa: E402

_spec = importlib.util.spec_from_file_location("terrain_eval", Path(__file__).resolve().parent / "terrain_eval.py")
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)
DT = 0.02
GROUPS = {"abduction": [0, 3, 6, 9], "hip": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
PHASES = ("approach", "edge", "square")


def phases(pitch_deg: np.ndarray, live: np.ndarray) -> np.ndarray:
    """(steps, envs) -> phase index per sample (-1 where not live); the same split is applied to robot logs."""
    T, n = pitch_deg.shape
    k = np.ones(50) / 50
    out = np.full((T, n), -1)
    for e in range(n):
        lv = live[:, e]
        if not lv.any():
            continue
        pu = np.convolve(pitch_deg[:, e], k, "same")
        up = (pu >= 4.0) & lv
        d = np.diff(np.r_[0, up.astype(int), 0])
        on, off = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        iv = [(a, b) for a, b in zip(on, off) if b - a + 1 >= 5]
        ph = np.zeros(T, dtype=int)
        if iv:
            ph[iv[0][0]:iv[-1][1] + 1] = 1
            ph[iv[-1][1] + 1:] = 2
        out[:, e] = np.where(lv, ph, -1)
    return out


@torch.no_grad()
def run(args) -> dict:
    pol = te.TorchPolicy(Path(args.policy) / "policy.npz", "cuda:0") if args.policy else None
    omni = bool(pol and pol.meta.get("omni")) or args.teacher
    arm = args.arm
    if arm is None and args.policy:
        from mp3_walk_policy.arms import trained_arm
        arm = trained_arm(args.policy)
    scale = pol.scale if pol else TerrainOptions().action_scale
    teacher = None
    if args.teacher:
        from mp3_walk_policy.teacher import TorchOmniTeacher
        teacher = TorchOmniTeacher("cuda:0")
        scale = (te.TEACHER_SCALE,) * 12
    opt = TerrainOptions(arm=arm or "bam_t5ub06", num_envs=args.envs, terrain=args.terrain, rows=1, noisy=True,
                         noise_ramp=False, noise_end=noise_profile("n2"), randomise=True, push_mps=0.0,
                         fixed_command=(args.vx, 0.0, 0.0), level=0, action_scale=scale, episode_s=args.seconds + 5.0,
                         history=int(pol.meta.get("history", 0) or 0) if pol else 0,
                         **({"foot_solref": tuple(args.foot_solref)} if args.foot_solref else {}))
    env = te.build(opt, seed=args.seed)
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    n, steps = env.num_envs, int(round(args.seconds / DT))
    robot = env.scene["robot"]
    term = servo_action(env)
    origin = env.scene.env_origins.clone()
    yaw0 = te.yaw_of(robot.data.root_link_quat_w)
    integ = torch.zeros(n, device=env.device)
    cmd = env.command_manager.get_term("twist")
    live = torch.ones(n, dtype=torch.bool, device=env.device)
    errs, pitches, lives, curs = [], [], [], []
    for _ in range(steps):
        if omni and args.heading_loop:
            err = te.wrap(yaw0 - te.yaw_of(robot.data.root_link_quat_w))
            wz, integ = pi_step(err, integ, DT, wz_max=args.heading_wz_max, antiwindup=args.heading_aw)
            cmd.vel_command_b[:, 2] = wz
        if teacher is not None:
            a = (teacher.target_rel(cmd.vel_command_b, env.episode_length_buf) / te.TEACHER_SCALE).clamp(-1.0, 1.0)
        else:
            a = pol(obs["actor"])
        obs, _, terminated, truncated, _ = env.step(a)
        # the step's target (set at its start) against the position at its end, 20 ms later
        e = (term.prev_rel - term.true_rel()).abs()
        live &= ~(terminated | truncated)
        if args.stop_at_goal:
            live &= (robot.data.root_link_pos_w - origin)[:, 0] < args.goal
        g = robot.data.projected_gravity_b
        errs.append(torch.rad2deg(e).cpu())
        pitches.append(torch.rad2deg(torch.asin((-g[:, 0]).clamp(-1, 1))).cpu())
        lives.append(live.cpu())
        c = te.bam_currents(env)
        if c is not None:
            curs.append(c.cpu())
    env.close()
    E = torch.stack(errs).numpy()                       # (T, n, 12)
    L = torch.stack(lives).numpy()
    PH = phases(torch.stack(pitches).numpy(), L)
    C = torch.stack(curs).numpy() if curs else None
    out = {"policy": "firmware_gait (omni teacher)" if args.teacher else args.policy, "arm": opt.arm,
           "terrain": args.terrain, "envs": n, "seconds": args.seconds, "seed": args.seed, "vx": args.vx,
           "foot_solref": opt.foot_solref, "heading_loop": bool(omni and args.heading_loop),
           "heading_aw": args.heading_aw, "goal_m": args.goal, "stop_at_goal": bool(args.stop_at_goal), "phases": {}}
    for name, mask in [("all", PH >= 0)] + [(p, PH == i) for i, p in enumerate(PHASES)]:
        row = {"n_samples": int(mask.sum())}
        for g, idx in GROUPS.items():
            if mask.any():
                x = E[mask][:, idx]
                row[f"{g}_err_p50_deg"] = float(np.percentile(x, 50))
                row[f"{g}_err_p95_deg"] = float(np.percentile(x, 95))
                if C is not None:
                    cu = C[mask][:, idx]
                    row[f"{g}_cur_p50_a"] = float(np.percentile(cu, 50))
                    row[f"{g}_cur_p99_a"] = float(np.percentile(cu, 99))
        out["phases"][name] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy")
    ap.add_argument("--teacher", action="store_true", help="the firmware trot (the omni teacher) instead of a policy")
    ap.add_argument("--arm")
    ap.add_argument("--terrain", default="square_walls", choices=("square", "square_walls", "strip", "flat"))
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--goal", type=float, default=0.65)
    ap.add_argument("--stop-at-goal", action="store_true")
    ap.add_argument("--heading-loop", action="store_true")
    ap.add_argument("--heading-aw", choices=ANTIWINDUP, default="cond")
    ap.add_argument("--heading-wz-max", type=float, default=WZ_MAX)
    ap.add_argument("--foot-solref", type=float, nargs=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json")
    args = ap.parse_args()
    if not (args.policy or args.teacher):
        ap.error("--policy or --teacher")
    res = run(args)
    for ph, r in res["phases"].items():
        print(f"{res['arm']:11s} {ph:9s} n {r['n_samples']:7d} " + " ".join(
            f"{g[:3]} {r.get(g + '_err_p50_deg', float('nan')):4.1f}/{r.get(g + '_err_p95_deg', float('nan')):4.1f}°"
            for g in GROUPS), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
