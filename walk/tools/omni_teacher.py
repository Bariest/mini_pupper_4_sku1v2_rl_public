#!/usr/bin/env python
"""The omni teacher (the firmware trot at (vx, vy, wz), ``teacher.TorchOmniTeacher``) played open loop on the omni
evaluation commands (``gates.OMNI_EVAL_COMMANDS``), clean observations: what the clone has to copy, per direction.

    python -B tools/omni_teacher.py [--arm stock] [--randomise] [--n 2] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3_walk_policy.env_cfg import omni_options                                          # noqa: E402
from mp3_walk_policy.gates import (OMNI_EVAL_COMMANDS, make_env, omni_gate, omni_metrics,  # noqa: E402
                                   rollout, teacher_fn)
from mp3_walk_policy.teacher import TorchOmniTeacher                                      # noqa: E402


def print_table(m: dict, gates: dict) -> None:
    for k, v in m.items():
        s = " ".join(f"{x} {v[x]:+.3f}" for x in ("along", "cross", "drift", "yaw_rate") if x in v)
        fails = [a for a, b in gates[k].items() if not b and a != "pass"]
        print(f"{k}  {s:38s} heading {v.get('heading_deg', 0):5.1f} f0 {v.get('f0_hz', 0):.2f} "
              f"clear {v.get('clearance_mm', 0):4.1f} pitch {v.get('pitch_deg', 0):.1f} roll {v.get('roll_deg', 0):.1f} "
              f"falls {v['fall_frac']:.2f}  {'ok' if not fails else 'fails ' + ','.join(fails)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--n", type=int, default=2, help="environments per command")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    C = OMNI_EVAL_COMMANDS
    env = make_env(omni_options(arm=a.arm, num_envs=len(C) * a.n, noisy=False, randomise=a.randomise,
                                noise_ramp=False, fixed_command=C, episode_s=a.seconds + 5.0))
    r = rollout(env, teacher_fn(TorchOmniTeacher(env.device), env, omni=True), seconds=a.seconds)
    env.close()
    m = omni_metrics(r, C)
    gates = {k: omni_gate(v) for k, v in m.items()}
    print_table(m, gates)
    if a.json:
        Path(a.json).write_text(json.dumps({"arm": a.arm, "randomise": a.randomise, "metrics": m, "gates": gates},
                                           indent=1, default=float))


if __name__ == "__main__":
    main()
