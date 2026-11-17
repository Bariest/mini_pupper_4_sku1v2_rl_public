#!/usr/bin/env python
"""The simulation gates for an omni policy.

    python -B scripts/evaluate_omni.py --ckpt runs/ppo/<run>/model_1299.pt [--arm <arm>]

1. Clean observations, nominal plant, 10 s on every ``gates.OMNI_EVAL_COMMANDS`` command (eight directions at
   0.10 m/s, the speed range's corners, turning in place, two curves): ``gates.omni_gate`` (speed along the command
   within 20 %, across it under 20 %, the commanded yaw rate within 20 % or the heading within 10 deg, clearance
   >= 8 mm, RMS pitch and roll < 3 deg, the firmware rhythm, no falls).
2. Robustness: 16 randomised plants per command under the run's training noise (the ramp's end): no falls and the
   rhythm on every command. For an N2 run also twice N2 (falls <= 1 % overall).
3. The joystick: 256 randomised plants under the training noise, a fresh random command every 2 s (the training
   distribution; for a ``--stops`` run, random take-like schedules) for 10 s: falls <= 1 %.
4. The recorded takes (``gates.takes_eval``: eight directions, turns in place at 1.2 rad/s, stops between them): clean,
   ``gates.stop_gate`` on every keyframe (a stop: the feet down and the body still); randomised under the training
   noise, no falls and the stops still.
Writes ``<run>/eval_omni_<checkpoint>.json``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                         # noqa: E402

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import noise_profile, omni_options, stops_options  # noqa: E402
from mp3_walk_policy.gates import (OMNI_EVAL_COMMANDS, make_env, omni_gate, omni_metrics, policy_fn,  # noqa: E402
                                   print_takes, rollout, takes_eval)

SAFETY_MAX_FALLS = 0.01


def run(arm, actor, n_per, noisy, randomise, noise="n1", commands=OMNI_EVAL_COMMANDS):
    env = make_env(omni_options(arm=arm, num_envs=len(commands) * n_per, noisy=noisy, randomise=randomise,
                                noise_ramp=False, fixed_command=commands, episode_s=60.0,
                                noise_end=noise_profile(noise)))
    m = omni_metrics(rollout(env, policy_fn(actor)), commands)
    env.close()
    return m


def joystick(arm, actor, noise, n=256, seconds=10.0, stops=False):
    """``stops``: random take-like schedules (walks, turns in place, stops) instead of a fresh command every 2 s."""
    kw = dict(arm=arm, num_envs=n, noisy=True, randomise=True, noise_ramp=False, episode_s=seconds + 5.0,
              noise_end=noise_profile(noise))
    opt = stops_options(schedule_seed=12345, **kw) if stops else omni_options(resample_s=(2.0, 2.0), **kw)
    env = make_env(opt)
    r = rollout(env, policy_fn(actor), seconds=seconds)
    env.close()
    return {"n": n, "fall_frac": float(r["fell"].mean()), "stops": stops}


def evaluate(ckpt, arm, noise, stops=False):
    actor = distill.load_actor(ckpt)
    res = {"checkpoint": str(ckpt), "arm": arm, "noise": noise}
    # the recorded takes: clean (every gate), and randomised under the training noise (no falls, the stops still)
    res["takes"] = takes_eval(arm, actor, n_per=2)
    tr = takes_eval(arm, actor, n_per=8, randomise=True, noisy=True, noise_end=noise_profile(noise))
    for rows in tr["takes"].values():
        for r in rows:
            g = r["gates"]
            r["gates"] = {"no_falls": g["no_falls"], **({"still": g["still"]} if "still" in g else {})}
            r["gates"]["pass"] = all(r["gates"].values())
    tr["pass"] = all(r["gates"]["pass"] for rows in tr["takes"].values() for r in rows)
    res["takes_robust"] = tr
    clean = run(arm, actor, 2, False, False)
    res["clean"] = {k: {"metrics": v, "gates": omni_gate(v)} for k, v in clean.items()}
    rob = run(arm, actor, 16, True, True, noise)
    rg = {k: {"no_falls": v["fall_frac"] == 0.0, "rhythm_f0": omni_gate(v).get("f0", False)} for k, v in rob.items()}
    for g in rg.values():
        g["pass"] = g["no_falls"] and g["rhythm_f0"]
    res["robust"] = {"metrics": rob, "gates": rg, "pass": all(g["pass"] for g in rg.values())}
    if noise == "n2":
        m2 = run(arm, actor, 16, True, True, "n2x2")
        falls = float(np.mean([v["fall_frac"] for v in m2.values()]))
        res["safety_2xN2"] = {"metrics": m2, "fall_frac": falls, "pass": falls <= SAFETY_MAX_FALLS}
    js = joystick(arm, actor, noise, stops=stops)
    res["joystick"] = dict(js, **{"pass": js["fall_frac"] <= SAFETY_MAX_FALLS})
    res["pass"] = (all(v["gates"]["pass"] for v in res["clean"].values()) and res["robust"]["pass"]
                   and res["joystick"]["pass"] and res.get("safety_2xN2", {"pass": True})["pass"]
                   and res["takes"]["pass"] and res["takes_robust"]["pass"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default=None, help="the simulator's arm (default: the one the policy was trained in)")
    ap.add_argument("--noise", default="", help="n1 or n2; default: the run's own training noise")
    a = ap.parse_args()
    ck = Path(a.ckpt)
    cfg = json.loads((ck.parent / "config.json").read_text()) if (ck.parent / "config.json").exists() else {}
    trained = cfg.get("task", {}).get("arm", "stock")
    arm = a.arm or trained
    noise = a.noise or cfg.get("noise", "n2")
    stops = bool(cfg.get("task", {}).get("options", {}).get("stops", False))
    res = evaluate(ck, arm, noise, stops)
    suffix = "" if arm == trained else f"_in_{arm}"
    (ck.parent / f"eval_omni_{ck.stem}{suffix}.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["clean"].items():
        m = v["metrics"]
        s = " ".join(f"{x} {m[x]:+.3f}" for x in ("along", "cross", "drift", "yaw_rate") if x in m)
        fails = [x for x, ok in v["gates"].items() if not ok and x != "pass"]
        print(f"clean {k}  {s:38s} heading {m.get('heading_deg', 0):5.1f} f0 {m.get('f0_hz', 0):.2f} "
              f"clear {m.get('clearance_mm', 0):4.1f} pitch {m.get('pitch_deg', 0):.1f} roll {m.get('roll_deg', 0):.1f} "
              f"falls {m['fall_frac']:.2f}  {'ok' if not fails else 'fails ' + ','.join(fails)}")
    print_takes(res["takes"])
    tf = [f"{n} kf{r['kf']}" for n, rows in res["takes_robust"]["takes"].items() for r in rows if not r["gates"]["pass"]]
    print(f"takes robust ({noise}, randomised): {'pass' if not tf else 'fails on ' + ', '.join(tf)}")
    rf = [k for k, g in res["robust"]["gates"].items() if not g["pass"]]
    print(f"robust ({noise}): {'pass' if not rf else 'fails on ' + ', '.join(rf)}")
    if "safety_2xN2" in res:
        print(f"safety 2xN2: falls {res['safety_2xN2']['fall_frac']:.3f} pass {res['safety_2xN2']['pass']}")
    print(f"joystick: falls {res['joystick']['fall_frac']:.3f} pass {res['joystick']['pass']}  |  ALL PASS {res['pass']}")


if __name__ == "__main__":
    main()
