#!/usr/bin/env python
"""Collect the teacher's data, train the clone, and run the clone gate.

    python scripts/distill.py --arm stock --out runs/clone/stock

Writes ``<out>/clone.pt`` (+ ``.meta.json``), ``<out>/gate.json``. The clone gate: at 0.10 m/s on
the nominal plant, the clone's closed-loop speed within 20 % of the open-loop teacher's, its gait fundamental within
0.2 Hz, and no falls. It decides whether the clone can anchor the fine-tuning.

``--omni``: the omni task (one joystick command per episode) and the omni teacher; the gate is ``gates.omni_gate``
against the teacher on every ``gates.OMNI_EVAL_COMMANDS`` command, nominal and randomised.

``--stops`` (implies ``--omni``): the data are take-like schedules (``takes.schedule_bank``: walks, turns in place to
1.2 rad/s, stops and restarts), labelled by the firmware playing each env's schedule through its own leg phases; the
gate adds the recorded takes (``gates.takes_eval``), the clone against the firmware open loop, nominal and randomised.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                              # noqa: E402

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions, omni_options              # noqa: E402
from mp3_walk_policy.gates import (OMNI_EVAL_COMMANDS, gate_table, make_env, metrics, omni_gate,  # noqa: E402
                                   omni_metrics, policy_fn, print_takes, rollout, takes_eval, teacher_fn)
from mp3_walk_policy.teacher import TorchOmniTeacher, TorchTeacher         # noqa: E402


def evaluate(arm, fn_builder, vx=0.10, randomise=False, n=4, noisy=False):
    env = make_env(TaskOptions(arm=arm, num_envs=n, noisy=noisy, randomise=randomise, noise_ramp=False,
                               vx_range=(vx, vx), episode_s=60.0))
    m = metrics(rollout(env, fn_builder(env)))
    env.close()
    return m


def evaluate_omni(arm, fn_builder, randomise=False, n=2):
    C = OMNI_EVAL_COMMANDS
    env = make_env(omni_options(arm=arm, num_envs=len(C) * n, noisy=False, randomise=randomise, noise_ramp=False,
                                fixed_command=C, episode_s=60.0))
    m = omni_metrics(rollout(env, fn_builder(env)), C)
    env.close()
    return m


def omni_gates(arm, actor):
    """The omni clone gate: on every evaluation command, the clone against the open-loop omni teacher."""
    res = {}
    for name, rnd, n in (("nominal", False, 2), ("randomised", True, 16)):
        t = evaluate_omni(arm, lambda e: teacher_fn(TorchOmniTeacher(e.device), e, omni=True), rnd, n)
        c = evaluate_omni(arm, lambda e: policy_fn(actor), rnd, n)
        gates = {k: omni_gate(c[k], teacher=t[k]) for k in c}
        res[name] = {"teacher": t, "clone": c, "gates": gates, "pass": all(g["pass"] for g in gates.values())}
        for k in c:
            fails = [x for x, ok in gates[k].items() if not ok and x != "pass"]
            tc = " ".join(f"{x} {t[k].get(x, 0):+.3f}/{c[k].get(x, 0):+.3f}" for x in ("along", "drift", "yaw_rate")
                          if x in c[k])
            print(f"{name} {k}  teacher/clone {tc}  falls {c[k]['fall_frac']:.2f}  "
                  f"{'ok' if not fails else 'fails ' + ','.join(fails)}")
        print(name, "PASS" if res[name]["pass"] else "FAIL")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omni", action="store_true", help="the omni task and teacher; the omni clone gate")
    ap.add_argument("--stops", action="store_true",
                    help="omni on take-like schedules with stops, labelled by the firmware; adds the takes gate")
    a = ap.parse_args()
    a.omni |= a.stops
    out = Path(a.out)
    data = distill.collect(a.arm, a.envs, action_noise=a.noise, seed=a.seed, omni=a.omni, stops=a.stops)
    print("collected", data["meta"])
    ckpt, meta = distill.train(data, epochs=a.epochs, seed=a.seed)
    meta.update({"collect": data["meta"]})
    digest = distill.save_clone(ckpt, meta, out / "clone.pt")
    print("clone", {k: meta[k] for k in ("final_train_mse", "final_val_mse", "rows")}, "sha256", digest)
    actor = distill.load_actor(out / "clone.pt")
    res = {"clone_sha256": digest}
    if a.omni:
        res.update(omni_gates(a.arm, actor))
        if a.stops:
            for name, rnd, n in (("takes_nominal", False, 2), ("takes_randomised", True, 8)):
                t = takes_eval(a.arm, None, n, rnd)
                res[name] = takes_eval(a.arm, actor, n, rnd, teacher_rows=t["takes"])
                res[name]["teacher"] = t["takes"]
                print_takes(res[name])
                print(name, "PASS" if res[name]["pass"] else "FAIL")
        (out / "gate.json").write_text(json.dumps(res, indent=1, default=float))
        return
    for name, rnd in (("nominal", False), ("randomised", True)):
        t = evaluate(a.arm, lambda e: teacher_fn(TorchTeacher(e.device), e), randomise=rnd, n=256 if rnd else 4)
        c = evaluate(a.arm, lambda e: policy_fn(actor), randomise=rnd, n=256 if rnd else 4)
        res[name] = {"teacher": t, "clone": c, "gate": gate_table(c, 0.10, teacher=t)}
        print(name, "teacher", t, "\n", name, "clone", c, "\n", name, "gate", res[name]["gate"])
    (out / "gate.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
