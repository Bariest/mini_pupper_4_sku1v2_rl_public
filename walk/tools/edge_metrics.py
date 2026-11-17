#!/usr/bin/env python
"""How a policy's feet meet the square's (or strip's) near edge, and how high each foot swings, in simulation.

    python -B tools/edge_metrics.py --policy policies/X --heading-loop --json runs/eval/X.edge.json
    python -B tools/edge_metrics.py --policy policies/X --heading-loop --terrain flat     # swing height only

Each env starts as in ``terrain_eval.py`` (the robot's centre 0.15 m before the near edge, facing +x, commanded
``--vx``; the heading loop for policies that observe the yaw-rate command), under noise on randomised plants.

Per foot (FR, FL, RR, RL):
- **swing apex:** the foot sphere's lowest point above the ground under it, at its highest in each swing (from lift-off
  to touch-down, contact from the feet's contact sensor); median and p10 over all swings;
- **strikes:** steps with a contact whose net force is mostly horizontal (|F_xy| > ``--ratio`` · |F_z|: the stumble
  test of Rudin et al. 2022), split by where the foot is: within ``--zone`` m of the near edge, or elsewhere;
- **edge time:** per trial, the time from the first foot within the edge zone to the last foot past it (all four
  feet beyond edge + zone), for trials that cross;
- **shin and thigh contacts** with the terrain within the edge zone, per leg.
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

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402

from mp3_walk_policy.env_cfg import noise_profile  # noqa: E402
from mp3_walk_policy.heading import ANTIWINDUP, WZ_MAX, pi_step  # noqa: E402
from mp3_walk_policy.sku2 import FOOT_GEOMS, FOOT_SITES, LEGS  # noqa: E402
from mp3_walk_policy.terrain_task import FOOT_R, TerrainOptions, foot_heights_centre, make_terrain_env_cfg  # noqa: E402

_spec = importlib.util.spec_from_file_location("terrain_eval", Path(__file__).resolve().parent / "terrain_eval.py")
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)

DT = 0.02
EDGE_X = 0.15          # the near edge, from the spawn point (TileStripCfg.spawn_gap)


@torch.no_grad()
def run(args) -> dict:
    pol = te.TorchPolicy(Path(args.policy) / "policy.npz", "cuda:0")
    omni = bool(pol.meta.get("omni"))
    from mp3_walk_policy.arms import trained_arm
    arm = args.arm or trained_arm(args.policy) or "stock"
    opt = TerrainOptions(arm=arm, num_envs=args.envs, terrain=args.terrain, rows=1, noisy=not args.clean,
                         noise_ramp=False, noise_end=noise_profile("n2"), randomise=not args.clean, push_mps=0.0,
                         fixed_command=(args.vx, 0.0, 0.0), level=0, action_scale=pol.scale,
                         episode_s=args.seconds + 5.0, history=int(pol.meta.get("history", 0) or 0),
                         **({"foot_solref": tuple(args.foot_solref)} if args.foot_solref else {}))
    cfg = make_terrain_env_cfg(opt)
    legs = {leg: (f"{leg}_thigh_collision", f"{leg}_calf_collision") for leg in LEGS}
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(name="foot_force", primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
                         reduce="netforce", num_slots=1),
        ContactSensorCfg(name="leg_terrain", primary=ContactMatch(mode="geom", pattern=sum(legs.values(), ()),
                                                                  entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="none",
                         num_slots=1),
    )
    cfg.seed = args.seed                    # before the constructor: its startup events draw the plants
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    n, steps = env.num_envs, int(round(args.seconds / DT))
    robot = env.scene["robot"]
    origin = env.scene.env_origins.clone()
    yaw0 = te.yaw_of(robot.data.root_link_quat_w)
    integ = torch.zeros(n, device=env.device)
    cmd = env.command_manager.get_term("twist")
    sites = [robot.site_names.index(s) for s in FOOT_SITES]
    ended_any = torch.zeros(n, dtype=torch.bool, device=env.device)
    apex = torch.zeros(n, 4, device=env.device)
    in_swing = torch.zeros(n, 4, dtype=torch.bool, device=env.device)
    apexes = [[] for _ in range(4)]
    strike_edge = torch.zeros(n, 4, device=env.device)
    strike_else = torch.zeros(n, 4, device=env.device)
    shin_edge = torch.zeros(n, 4, device=env.device)
    t_first = torch.full((n,), float("nan"), device=env.device)
    t_last = torch.full((n,), float("nan"), device=env.device)
    for k in range(steps):
        if omni and args.heading_loop:
            err = te.wrap(yaw0 - te.yaw_of(robot.data.root_link_quat_w))
            wz, integ = pi_step(err, integ, DT, wz_max=args.heading_wz_max, antiwindup=args.heading_aw)
            cmd.vel_command_b[:, 2] = wz
        obs, _, terminated, truncated, _ = env.step(pol(obs["actor"]))
        ended_any |= terminated | truncated
        live = ~ended_any
        fx = robot.data.site_pos_w[:, sites, 0] - origin[:, :1]                      # (n, 4) foot x from the spawn
        near = (fx - EDGE_X).abs() <= args.zone
        ff = env.scene["foot_force"].data
        found = ff.found.view(n, 4) > 0
        f = ff.force.view(n, 4, 3)
        strike = found & (f[..., :2].norm(dim=-1) > args.ratio * f[..., 2].abs())
        strike_edge += (strike & near & live[:, None]).float()
        strike_else += (strike & ~near & live[:, None]).float()
        lc = env.scene["leg_terrain"].data.found.view(n, 4, 2) > 0
        shin_edge += (lc.any(-1) & near & live[:, None]).float()
        # swing apex: the foot sphere's bottom above the ground under it, maximum over each airborne interval
        h = foot_heights_centre(env) - FOOT_R
        air = ~found
        apex = torch.where(air & in_swing, torch.maximum(apex, h), torch.where(air, h, apex))
        landed = in_swing & ~air
        for j in range(4):
            sel = landed[:, j] & live
            if sel.any():
                apexes[j].append(apex[sel, j].cpu())
        in_swing = air
        # edge window: the first foot within the zone, then all four beyond it
        t = (k + 1) * DT
        first = near.any(-1) & torch.isnan(t_first) & live
        t_first = torch.where(first, torch.full_like(t_first, t), t_first)
        past = (fx > EDGE_X + args.zone).all(-1) & torch.isnan(t_last) & ~torch.isnan(t_first) & live
        t_last = torch.where(past, torch.full_like(t_last, t), t_last)
    env.close()
    se, so, sh = strike_edge.cpu().numpy() * DT, strike_else.cpu().numpy() * DT, shin_edge.cpu().numpy() * DT
    win = (t_last - t_first).cpu().numpy()
    out = {"policy": args.policy, "terrain": args.terrain, "envs": n, "seconds": args.seconds, "vx": args.vx,
           "ratio": args.ratio, "zone_m": args.zone, "foot_solref": opt.foot_solref, "seed": args.seed,
           "heading_loop": bool(args.heading_loop and omni), "heading_aw": args.heading_aw, "heading_wz_max": args.heading_wz_max, "legs": {}}
    for j, leg in enumerate(LEGS):
        a = torch.cat(apexes[j]).numpy() * 1000 if apexes[j] else np.array([np.nan])
        out["legs"][leg] = {"swings": int(a.size), "apex_med_mm": float(np.nanmedian(a)),
                            "apex_p10_mm": float(np.nanpercentile(a, 10)),
                            "edge_strike_s_mean": float(se[:, j].mean()),
                            "edge_strike_trials": float(np.mean(se[:, j] > 0)),
                            "other_strike_s_mean": float(so[:, j].mean()),
                            "edge_leg_contact_trials": float(np.mean(sh[:, j] > 0))}
    crossed = ~np.isnan(win)
    out["edge_window"] = {"crossed": float(crossed.mean()),
                          "med_s": float(np.median(win[crossed])) if crossed.any() else None,
                          "p90_s": float(np.percentile(win[crossed], 90)) if crossed.any() else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--terrain", default="square_walls", choices=("square", "square_walls", "strip", "flat"))
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--heading-loop", action="store_true")
    ap.add_argument("--heading-aw", choices=ANTIWINDUP, default="cond", help="the loop's anti-windup (heading.pi_step; none = no anti-windup)")
    ap.add_argument("--heading-wz-max", type=float, default=WZ_MAX, help="rad/s: the loop's output limit")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--foot-solref", type=float, nargs=2)
    ap.add_argument("--ratio", type=float, default=1.5, help="|F_xy| / |F_z| above which a contact is a strike")
    ap.add_argument("--zone", type=float, default=0.03, help="m either side of the near edge")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = run(a)
    print(f"{'leg':4s} {'apex med':>9s} {'p10':>6s}  {'edge strike s':>13s} {'trials':>7s}  {'other s':>8s}  "
          f"{'shin@edge':>9s}")
    for leg, r in res["legs"].items():
        print(f"{leg:4s} {r['apex_med_mm']:7.1f}mm {r['apex_p10_mm']:5.1f}  {r['edge_strike_s_mean']:13.3f} "
              f"{100 * r['edge_strike_trials']:6.0f}%  {r['other_strike_s_mean']:8.3f}  "
              f"{100 * r['edge_leg_contact_trials']:8.0f}%")
    w = res["edge_window"]
    if w["med_s"] is not None:
        print(f"edge window (first foot in the zone -> all four past): crossed {100 * w['crossed']:.0f}%, "
              f"median {w['med_s']:.2f} s, p90 {w['p90_s']:.2f} s")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
