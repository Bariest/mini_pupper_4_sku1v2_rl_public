#!/usr/bin/env python
"""Distil a privileged terrain teacher into a blind student (DAgger: the student acts, the teacher labels every
step; rsl_rl's ``Distillation``, MSE on the actions).

    python scripts/distill_terrain.py \\
        --teacher runs/terrain/<teacher run>/model_k.pt --student-init runs/terrain/<blind run>/model_k.pt --seed 1

- The teacher is a ``train_terrain.py --privileged`` checkpoint: its actor reads the "actor" group (the robot's
  observations with their history) and the "privileged" group (the body scan and the foot heights).
- The student is the robot's network (the "actor" group only), initialised from a blind terrain checkpoint
  (``--student-init``), with its action noise std set to ``--student-std`` while it collects data.
- The environment is the teacher's (its run's ``config.json`` task options), with ``--num-envs`` envs.

Checkpoints ``model_<k>.pt`` hold ``student_state_dict`` (exported by ``scripts/export_policy.py`` as a robot policy)
and ``teacher_state_dict``; ``config.json`` as for training runs, with ``"distillation": true``.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                   # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper                                    # noqa: E402
from rsl_rl.runners import DistillationRunner                              # noqa: E402

from mp3_walk_policy.env_cfg import describe                               # noqa: E402
from mp3_walk_policy.paths import sha256                                   # noqa: E402
from mp3_walk_policy.terrain_task import TerrainOptions, make_terrain_env_cfg  # noqa: E402


def options_from(run_dir: Path, **over) -> TerrainOptions:
    opts = json.loads((run_dir / "config.json").read_text())["task"]["options"]
    opts = {k: (tuple(v) if isinstance(v, list) else v) for k, v in opts.items()}
    opts.update(over)
    return TerrainOptions(**opts)


def model_cfg(std: float) -> dict:
    return {"class_name": "MLPModel", "hidden_dims": (128, 128, 64), "activation": "elu", "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": std, "std_type": "scalar"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="a --privileged terrain run's model_<k>.pt")
    ap.add_argument("--student-init", required=True, help="a blind terrain run's model_<k>.pt (the student's start)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2, help="learning epochs over each rollout")
    ap.add_argument("--gradient-length", type=int, default=15, help="minibatches per gradient step")
    ap.add_argument("--student-std", type=float, default=0.05, help="the student's action noise while collecting")
    ap.add_argument("--out", default=str(REPO / "runs/distill"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    ap.add_argument("--wandb-project", default="mp3_walk_policy")
    ap.add_argument("--wandb-entity")
    a = ap.parse_args()
    if a.logger == "wandb":
        if os.environ.get("WANDB_MODE") in ("disabled", "offline"):
            raise SystemExit("--logger wandb with WANDB_MODE=%s: unset it to log online" % os.environ["WANDB_MODE"])
        if a.wandb_entity:
            os.environ["WANDB_USERNAME"] = a.wandb_entity
        from mp3_walk_policy import wandb_compat
        wandb_compat.patch()

    teacher_run, student_run = Path(a.teacher).parent, Path(a.student_init).parent
    opt = options_from(teacher_run, num_envs=a.num_envs, privileged=True)
    s_opts = json.loads((student_run / "config.json").read_text())["task"]["options"]
    assert s_opts["history"] == opt.history and list(s_opts["action_scale"]) == list(opt.action_scale), \
        "the student's history and action scale must be the teacher's"
    env_cfg = make_terrain_env_cfg(opt)
    env_cfg.seed = a.seed
    name = f"{datetime.now():%Y%m%d_%H%M%S}_distill_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    log_dir = Path(a.out) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"seed": a.seed, "num_steps_per_env": 24, "max_iterations": a.iters, "save_interval": 50,
           "experiment_name": "mp3_distill", "run_name": "", "logger": a.logger, "wandb_project": a.wandb_project,
           "obs_groups": {"student": ["actor"], "teacher": ["actor", "privileged"]},
           "student": model_cfg(a.student_std), "teacher": model_cfg(a.student_std),
           "algorithm": {"class_name": "Distillation", "num_learning_epochs": a.epochs,
                         "gradient_length": a.gradient_length, "learning_rate": a.lr, "max_grad_norm": 1.0,
                         "loss_type": "mse"},
           "clip_actions": 1.0, "multi_gpu": None}
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = DistillationRunner(venv, cfg, str(log_dir), "cuda:0")
    runner.load(a.teacher, load_cfg={"teacher": True}, map_location="cuda:0")      # its actor_state_dict
    sd = torch.load(a.student_init, map_location="cuda:0", weights_only=False)["actor_state_dict"]
    runner.alg.student.load_state_dict(sd, strict=True)
    with torch.no_grad():
        runner.alg.student.distribution.std_param.fill_(a.student_std)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "terrain_task": True, "distillation": True, "runner": cfg,
        "teacher": a.teacher, "teacher_sha256": sha256(Path(a.teacher)),
        "student_init": a.student_init, "student_init_sha256": sha256(Path(a.student_init)),
        "repo_commit": commit, "repo_dirty": dirty, "argv": sys.argv}, indent=1, default=str))
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
