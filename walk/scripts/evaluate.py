#!/usr/bin/env python
"""The simulation gates for a trained policy (MP2's gate table, for this robot).

    python scripts/evaluate.py --ckpt runs/ppo/<run>/model_1299.pt

1. Clean observations, nominal plant, 10 s at 0.05 / 0.10 / 0.15 / 0.20 m/s: speed, fundamental, clearance, pitch,
   drift; the gates (speed within 20 % of the command, clearance >= 8 mm, RMS pitch < 3 deg, no falls).
2. Robustness at 0.10 m/s: 256 randomised environments, the full observation noise (the ramp's end), the chain delay
   drawn: no falls, and the rhythm (the fundamental within 0.3 Hz of the firmware trot's). The noise is the run's
   training profile (``--noise``, else read from the run's config). For an N2 run also a safety check under twice
   N2 (falls <= 1 %), and the falls under N1 (reported).
3. The open-loop bar (MP2): the policy's clean action stream at 0.10 m/s on the nominal plant, replayed blind on 256
   randomised plants; the ratio of its distance to the closed loop's on those plants >= 0.75, and the blind speed
   >= 0.06 m/s. The plants share friction and COM between the two rollouts; the chain delay is
   redrawn at each reset.
Writes ``<run>/eval_<checkpoint>.json``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions, noise_profile                             # noqa: E402
from mp3_walk_policy.gates import gate_table, make_env, metrics, policy_fn, replay_fn, rollout  # noqa: E402

OPEN_LOOP_BAR = 0.75
OPEN_LOOP_MIN_SPEED = 0.06      # m/s at the 0.10 command


SAFETY_MAX_FALLS = 0.01        # falls allowed under twice the N2 noise


def run(arm, fn, vx, n, noisy, randomise, record=False, noise="n1"):
    env = make_env(TaskOptions(arm=arm, num_envs=n, noisy=noisy, randomise=randomise, noise_ramp=False,
                               vx_range=(vx, vx), episode_s=60.0, noise_end=noise_profile(noise)))
    r = rollout(env, fn, record_actions=record)
    env.close()
    return r


def evaluate(ckpt, arm, noise="n1"):
    actor = distill.load_actor(ckpt)
    res = {"checkpoint": str(ckpt), "arm": arm, "noise": noise, "clean": {}}
    for vx in (0.05, 0.10, 0.15, 0.20):
        m = metrics(run(arm, policy_fn(actor), vx, 4, False, False))
        res["clean"][f"{vx:.2f}"] = {"metrics": m, "gates": gate_table(m, vx)}
    rob = metrics(run(arm, policy_fn(actor), 0.10, 256, True, True, noise=noise))
    g = gate_table(rob, 0.10)
    res["robust_0.10"] = {"metrics": rob, "gates": {"no_falls": g["no_falls"], "rhythm_f0": g["f0"],
                                                    "pass": g["no_falls"] and g["f0"]}}
    if noise == "n2":
        m2 = metrics(run(arm, policy_fn(actor), 0.10, 256, True, True, noise="n2x2"))
        res["safety_2xN2"] = {"metrics": m2, "pass": m2["fall_frac"] <= SAFETY_MAX_FALLS}
        res["robust_N1"] = {"metrics": metrics(run(arm, policy_fn(actor), 0.10, 256, True, True, noise="n1"))}
    # The open-loop bar (MP2): the policy's action stream recorded on the nominal plant, replayed blind on 256
    # randomised plants, against the closed loop on those same plants. (Recording and replaying on one deterministic
    # plant reproduces the closed loop bit for bit: a ratio of exactly 1, by construction.)
    nominal = run(arm, policy_fn(actor), 0.10, 1, False, False, record=True)
    env = make_env(TaskOptions(arm=arm, num_envs=256, noisy=False, randomise=True, noise_ramp=False,
                               vx_range=(0.10, 0.10), episode_s=60.0))
    rec = rollout(env, policy_fn(actor))
    actions = torch.tensor(nominal["actions"], device=env.device).expand(-1, 256, -1).contiguous()
    blind_r = rollout(env, replay_fn(actions))
    env.close()
    closed, blind = metrics(rec), metrics(blind_r)
    ratio = blind.get("speed", 0.0) / max(closed.get("speed", 1e-9), 1e-9)
    # a ratio alone passes a robot that does not walk (near-zero over near-zero): the blind replay must also carry the
    # body at least OPEN_LOOP_MIN_SPEED (MP2's bar was absolute: 660 mm in 400 steps at 0.10 m/s)
    res["open_loop"] = {"closed": closed, "blind": blind, "ratio": ratio,
                        "pass": ratio >= OPEN_LOOP_BAR and blind.get("speed", 0.0) >= OPEN_LOOP_MIN_SPEED}
    res["pass"] = (all(v["gates"]["pass"] for v in res["clean"].values()) and res["robust_0.10"]["gates"]["pass"]
                   and res["open_loop"]["pass"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default=None, help="the simulator's arm (default: the one the policy was trained in)")
    ap.add_argument("--noise", default="", help="n1 or n2; default: the run's own training noise")
    a = ap.parse_args()
    ck = Path(a.ckpt)
    from mp3_walk_policy.arms import trained_arm
    a.arm = a.arm or trained_arm(ck) or "stock"
    noise = a.noise or (json.loads((ck.parent / "config.json").read_text()).get("noise", "n1")
                        if (ck.parent / "config.json").exists() else "n1")
    res = evaluate(ck, a.arm, noise)
    trained_arm = json.loads((ck.parent / "config.json").read_text())["task"]["arm"] if (ck.parent / "config.json").exists() else a.arm
    suffix = "" if a.arm == trained_arm else f"_in_{a.arm}"          # a simulator swap keeps its own file
    (ck.parent / f"eval_{ck.stem}{suffix}.json").write_text(json.dumps(res, indent=1, default=float))
    for vx, v in res["clean"].items():
        m = v["metrics"]
        print(f"clean {vx}: speed {m.get('speed', float('nan')):.3f} f0 {m.get('f0_hz', float('nan')):.2f} "
              f"clear {m.get('clearance_mm', 0):.1f} pitch {m.get('pitch_deg', 0):.2f} lat {m.get('lateral_mm', 0):.0f} "
              f"yaw {m.get('yaw_deg', 0):.1f} falls {m['fall_frac']:.2f} -> {v['gates']}")
    r = res["robust_0.10"]["metrics"]
    print(f"robust 0.10: falls {r['fall_frac']:.3f} speed {r.get('speed', float('nan')):.3f} f0 {r.get('f0_hz', 0):.2f}")
    if "safety_2xN2" in res:
        print(f"safety 2xN2: falls {res['safety_2xN2']['metrics']['fall_frac']:.3f} pass {res['safety_2xN2']['pass']}"
              f"  |  N1 falls {res['robust_N1']['metrics']['fall_frac']:.3f}")
    o = res["open_loop"]
    print(f"open loop: closed {o['closed'].get('speed', 0):.3f} blind {o['blind'].get('speed', 0):.3f} "
          f"ratio {o['ratio']:.2f} pass {o['pass']}  |  ALL PASS {res['pass']}")


if __name__ == "__main__":
    main()
