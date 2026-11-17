#!/usr/bin/env python
"""Anchored PPO from the clone (MP2's phase 2): warm-start the actor from the clone and pull its mean towards the
clone's for the whole run.

    python scripts/train.py \
        --arm stock --clone runs/clone/stock/clone.pt --seed 1 --out runs/ppo

The defaults are MP2's hyperparameters (mp2_x_walk_policy agents.py) except the initial std (0.10, below; MP2: 0.8):
actor and critic 128-128-64 ELU, PPO lr 1e-3 adaptive (KL 0.01), gamma 0.99, lambda 0.95, clip 0.2, entropy 0.005, 5
epochs x 4 minibatches, 24 steps x 512 environments per iteration, 1300 iterations, a constant anchor beta 1.2011598
(MP2's rule). The critic sees the clean observation plus the true base velocity, height and foot contacts; the actor
sees only the robot's observation.

``--beta 0`` is plain PPO from the clone. Recipe v3 (the shipped policies) sets ``--beta 20 --fixed-std
--critic-warmup 50 --lr0 1e-4`` with the default noise N2. Options: ``--fixed-std`` holds the exploration std at
``--init-std`` (not learned), ``--critic-warmup N`` trains only the value function for the first N updates, ``--lr0``
sets the learning rate the adaptive schedule starts from (MP2: 1e-3). ``--omni`` trains the omni task (from an omni
clone, ``scripts/distill.py --omni``). Every parameter, the sources' hashes and this
repository's commit go into ``<run>/config.json``.
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                   # noqa: E402
from mjlab.rl import (MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg,  # noqa: E402
                      RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)

from mp3_walk_policy.env_cfg import (OMNI_RESAMPLE_S, SPRINT_CHAIN_MS, TaskOptions, describe,  # noqa: E402
                                     make_env_cfg, noise_profile, omni_options, sprint_options, stops_options)
from mp3_walk_policy.paths import sha256                                 # noqa: E402

MP2_BETA = 1.2011598444702776
# The initial exploration std. MP2 used 0.8 on a 0.25 rad action scale behind a soft, 50-90 ms plant. Here the scale is
# 0.40-0.42 rad and the stock servo stiff and fast: with the clone's actions plus white noise of std 0.8 / 0.5 / 0.3 /
# 0.2 / 0.15 / 0.1, 100 / 98 / 68 % (over 4 s) and 38 / 7 / 0 % (over 10 s) of 256 randomised robots fall.
# 0.1 is the largest that does not tip the clone over; the same in every arm.
INIT_STD = 0.10


def runner_cfg(seed, iters, envs_steps=24, logger="tensorboard", wandb_project="mp3_walk_policy"):
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=envs_steps, max_iterations=iters, save_interval=25,
        experiment_name="mp3_walk", logger=logger, wandb_project=wandb_project, clip_actions=1.0,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0, class_name="mp3_walk_policy.algorithms:AnchorPPO"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--clone", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1300)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--beta", type=float, default=MP2_BETA)
    ap.add_argument("--init-std", type=float, default=INIT_STD)
    ap.add_argument("--fixed-std", action="store_true", help="hold the exploration std at --init-std")
    ap.add_argument("--critic-warmup", type=int, default=0, help="value-only updates before the actor moves")
    ap.add_argument("--lr0", type=float, default=1.0e-3, help="the adaptive learning rate's starting value")
    ap.add_argument("--out", default=str(REPO / "runs/ppo"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--noise", default="n2", choices=("n1", "n2"), help="the observation-noise profile at the ramp's end")
    ap.add_argument("--omni", action="store_true", help="the omni task, commands resampled every 4-6 s")
    ap.add_argument("--stops", action="store_true",
                    help="the omni task on take-like schedules with stops (env_cfg.stops_options; a --stops clone)")
    ap.add_argument("--sprint", action="store_true",
                    help="the sprint: straight ahead, a speed curriculum (env_cfg.sprint_options); --clone is the walk "
                         "it starts from (tools/mpxp_to_init.py makes one from a robot policy file)")
    ap.add_argument("--sprint-max", type=float, default=0.60, help="the sprint curriculum's top speed (m/s)")
    ap.add_argument("--sprint-start", type=float, default=0.25, help="... and where it starts")
    ap.add_argument("--sprint-ramp", type=int, nargs=2, default=(100, 700), metavar=("HOLD", "RAMP"),
                    help="iterations at the start speed, then iterations to reach --sprint-max")
    ap.add_argument("--chain-ms", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the sensing-to-command chain delay range (default: 8 19; the sprint: %s %s, the arena's)"
                         % SPRINT_CHAIN_MS)
    ap.add_argument("--base-mass", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the sprint: the base's mass scaled by U(LO, HI) (default 0.85 1.15)")
    ap.add_argument("--heading-weight", type=float, default=-3.0, help="the sprint's heading term")
    ap.add_argument("--cross-weight", type=float, default=-3.0, help="the sprint's cross-track term")
    ap.add_argument("--anchor-hold", type=int, default=None,
                    help="iterations the anchor's beta is held (default: the whole run)")
    ap.add_argument("--anchor-decay", type=int, default=0, help="then iterations to decay it to --beta-floor")
    ap.add_argument("--beta-floor", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0", help="cuda:0; cpu for a slow check without a GPU")
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard",
                    help="wandb also writes the TensorBoard events (needs `wandb login` and WANDB_MODE not disabled)")
    ap.add_argument("--wandb-project", default="mp3_walk_policy")
    ap.add_argument("--wandb-entity", help="the W&B user or team (default: the account's default entity)")
    a = ap.parse_args()
    if a.logger == "wandb" and os.environ.get("WANDB_MODE") in ("disabled", "offline"):
        raise SystemExit("--logger wandb with WANDB_MODE=%s: unset it to log online" % os.environ["WANDB_MODE"])
    if a.wandb_entity:
        os.environ["WANDB_USERNAME"] = a.wandb_entity           # rsl_rl reads the entity from it
    if a.logger == "wandb":
        from mp3_walk_policy import wandb_compat
        wandb_compat.patch()

    kw = dict(arm=a.arm, num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True, noise_end=noise_profile(a.noise))
    if a.chain_ms is not None:
        kw["chain_ms"] = tuple(a.chain_ms)
    if a.sprint:
        skw = dict(sprint_max=a.sprint_max, sprint_start_hi=a.sprint_start, sprint_hold_iters=a.sprint_ramp[0],
                   sprint_ramp_iters=a.sprint_ramp[1], heading_weight=a.heading_weight, cross_weight=a.cross_weight)
        if a.base_mass is not None:
            skw["base_mass_scale"] = tuple(a.base_mass)
        opt = sprint_options(**kw, **skw)
    elif a.stops:
        opt = stops_options(schedule_seed=a.seed, **kw)
    elif a.omni:
        opt = omni_options(resample_s=OMNI_RESAMPLE_S, **kw)
    else:
        opt = TaskOptions(**kw)
    env_cfg = make_env_cfg(opt)
    env_cfg.seed = a.seed
    kind = "_sprint" if a.sprint else "_stops" if a.stops else "_omni" if a.omni else ""
    name = (f"{datetime.now():%Y%m%d_%H%M%S}_{a.arm}{kind}_s{a.seed}"
            + (f"_{a.tag}" if a.tag else ""))
    log_dir = Path(a.out) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
    venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
    clone = Path(a.clone).resolve()
    cfg = asdict(runner_cfg(a.seed, a.iters, logger=a.logger, wandb_project=a.wandb_project))
    cfg["algorithm"].update(anchor_path=str(clone), anchor_sha256=sha256(clone), anchor_beta0=a.beta,
                            anchor_hold_iters=a.iters if a.anchor_hold is None else a.anchor_hold,
                            anchor_decay_iters=a.anchor_decay, anchor_beta_floor=a.beta_floor,
                            anchor_obs_group="anchor", actor_warmup_iters=a.critic_warmup)
    cfg["algorithm"]["learning_rate"] = a.lr0
    runner = MjlabOnPolicyRunner(venv, cfg, str(log_dir), a.device)

    # warm start: the clone's mean net and its exploration std (MP2: the clone's std is set to the init std)
    sd = torch.load(clone, map_location=a.device, weights_only=False)["actor_state_dict"]
    missing, unexpected = runner.alg.actor.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("normalizer" in k for k in missing), missing
    with torch.no_grad():
        runner.alg.actor.distribution.std_param.fill_(a.init_std)
    if a.fixed_std:
        runner.alg.actor.distribution.std_param.requires_grad_(False)

    try:                     # a copy without git (a zipped bundle) or a PC without git: no commit recorded
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "runner": cfg, "clone": str(clone), "clone_sha256": sha256(clone),
        "contract": os.environ.get("MP3_CONTRACT", "") or "sku2_contract.json",
        "init_base_z": os.environ.get("MP3_INIT_Z", ""),
        "noise": a.noise, "init_std": a.init_std, "fixed_std": a.fixed_std, "critic_warmup": a.critic_warmup, "lr0": a.lr0,
        "repo_commit": commit, "repo_dirty": dirty, "argv": sys.argv}, indent=1, default=str))
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
