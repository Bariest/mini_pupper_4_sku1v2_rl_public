#!/usr/bin/env python
"""Watch a policy (a training checkpoint, or a clone) walk, live.

    python -B scripts/watch.py --ckpt runs/ppo/<run>                 # the run's newest checkpoint
    python -B scripts/watch.py --ckpt runs/ppo/<run>/model_600.pt --viewer viser
    python -B scripts/watch.py --ckpt runs/clone/stock_omni_e150_n06/clone.pt --arm stock --omni

``--viewer native`` (default): a MuJoCo window. An omni policy tours ``gates.OMNI_EVAL_COMMANDS``, ``--hold`` s each
(the command is printed as it changes); ``--cmd vx,vy,wz`` holds one command instead. ``--viewer viser``: a browser
viewer (the URL is printed) with joystick sliders for (vx, vy, wz) under "Twist": tick "Enable" to drive.
The plant is nominal and the observations clean unless ``--noisy`` / ``--randomise``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper                                    # noqa: E402

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import TaskOptions, make_env_cfg, noise_profile, omni_options  # noqa: E402
from mp3_walk_policy.gates import OMNI_EVAL_COMMANDS                       # noqa: E402


def resolve(ckpt: Path) -> Path:
    if ckpt.is_dir():
        models = sorted(ckpt.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not models:
            raise SystemExit(f"no model_*.pt in {ckpt}")
        return models[-1]
    return ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a checkpoint (.pt) or a run directory (its newest checkpoint)")
    ap.add_argument("--arm", default=None, help="default: the run's arm")
    ap.add_argument("--omni", action="store_true", help="an omni policy (default: from the run's config)")
    ap.add_argument("--viewer", default="native", choices=("native", "viser"))
    ap.add_argument("--cmd", default=None, help="vx,vy,wz held for the whole session")
    ap.add_argument("--hold", type=float, default=4.0, help="s per command on the omni tour")
    ap.add_argument("--noisy", action="store_true", help="observation noise (the run's profile at the ramp's end)")
    ap.add_argument("--randomise", action="store_true", help="friction, COM and the chain delay drawn")
    a = ap.parse_args()

    ck = resolve(Path(a.ckpt))
    cfg_path = ck.parent / "config.json"
    run = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    opts = run.get("task", {}).get("options", {})
    arm = a.arm or run.get("task", {}).get("arm", "stock")
    omni = a.omni or bool(opts.get("omni", False))
    print(f"checkpoint {ck}  arm {arm}  {'omni' if omni else 'forward'}", flush=True)

    kw = dict(arm=arm, num_envs=1, noisy=a.noisy, randomise=a.randomise, noise_ramp=False, episode_s=3600.0,
              visuals=True, noise_end=noise_profile(run.get("noise", "n2")))
    cmd = tuple(float(v) for v in a.cmd.split(",")) if a.cmd else None
    if omni:
        opt = omni_options(fixed_command=cmd or OMNI_EVAL_COMMANDS[0], **kw)
    else:
        vx = cmd[0] if cmd else 0.10
        opt = TaskOptions(vx_range=(vx, vx), **kw)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=make_env_cfg(opt), device="cuda:0"), clip_actions=1.0)
    actor = distill.load_actor(ck)
    term = env.unwrapped.command_manager.get_term("twist")
    tour = omni and cmd is None and a.viewer == "native"
    steps_per = max(1, int(round(a.hold / env.unwrapped.step_dt)))
    state = {"k": 0, "i": -1}

    @torch.no_grad()
    def policy(obs):
        if tour:
            i = (state["k"] // steps_per) % len(OMNI_EVAL_COMMANDS)
            if i != state["i"]:
                state["i"] = i
                c = OMNI_EVAL_COMMANDS[i]
                term.vel_command_b[:] = torch.tensor(c, device=term.vel_command_b.device)
                print(f"command vx {c[0]:+.3f} vy {c[1]:+.3f} m/s  wz {c[2]:+.2f} rad/s", flush=True)
            state["k"] += 1
        return actor(obs["actor"])

    if a.viewer == "native":
        from mjlab.viewer import NativeMujocoViewer
        NativeMujocoViewer(env, policy).run()
    else:
        from mjlab.viewer import ViserPlayViewer
        ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
    main()
