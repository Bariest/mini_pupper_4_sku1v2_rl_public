#!/usr/bin/env python
"""Size the anchor's beta by MP2's rule, measured in this environment.

    python tools/anchor_beta.py --arm stock \
        --clone runs/clone/stock/clone.pt --drifted runs/ppo/<run>/model_1299.pt [--frac 0.3] [--std 0.10]

MP2 (mp2_x_walk_policy agents.py:64-67) chose beta so that, with the actor 30 % of the way from the clone to an
unanchored policy, the anchor's gradient on the actor equals the PPO surrogate's (0.7946 / 0.6615 = 1.2012). Here the
"unanchored" end is a policy that drifted under a weak anchor (``--drifted``). The actor is placed at ``clone + frac *
(drifted - clone)`` with the exploration std fixed at ``--std``; the critic is the drifted run's (a trained value
function). A
PPO batch (24 steps x 512 environments, the full observation noise, randomised plants) is collected after warm-up
batches; on each of its 4 minibatches (PPO's split) the gradient norms on the actor's parameters are taken for the
surrogate at ratio 1 and for the anchor MSE at beta = 1. beta = mean |g_surrogate| / mean |g_anchor|.
"""
import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                   # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper               # noqa: E402

from mp3_walk_policy.env_cfg import TaskOptions, make_env_cfg            # noqa: E402
from mp3_walk_policy.paths import sha256                                 # noqa: E402

_spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts/train.py")
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def grad_norm(loss, params) -> float:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--clone", required=True)
    ap.add_argument("--drifted", required=True)
    ap.add_argument("--frac", type=float, nargs="+", default=[0.3])
    ap.add_argument("--std", type=float, default=0.10)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3, help="batches collected and discarded first")
    ap.add_argument("--batches", type=int, default=3, help="batches measured (the result is their mean)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()

    opt = TaskOptions(arm=a.arm, num_envs=a.envs, noisy=True, noise_ramp=False, randomise=True)
    cfg_env = make_env_cfg(opt)
    cfg_env.seed = a.seed
    env = ManagerBasedRlEnv(cfg=cfg_env, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
    clone = Path(a.clone).resolve()
    cfg = asdict(train_script.runner_cfg(a.seed, 1))
    cfg["algorithm"].update(anchor_path=str(clone), anchor_sha256=sha256(clone), anchor_beta0=1.0,
                            anchor_hold_iters=1, anchor_decay_iters=0, anchor_beta_floor=0.0, anchor_obs_group="anchor")
    runner = MjlabOnPolicyRunner(venv, cfg, str(REPO / "runs/anchor_beta_tmp"), "cuda:0")
    alg = runner.alg
    c_sd = torch.load(clone, map_location="cuda:0", weights_only=False)["actor_state_dict"]
    d_all = torch.load(a.drifted, map_location="cuda:0", weights_only=False)
    d_sd = d_all["actor_state_dict"]
    alg.critic.load_state_dict(d_all["critic_state_dict"])
    params = [p for n, p in alg.actor.named_parameters() if n.startswith("mlp.")]
    out = {"arm": a.arm, "clone": str(clone), "drifted": str(Path(a.drifted).resolve()), "std": a.std, "rows": []}
    for frac in a.frac:
        sd = {k: (c_sd[k] + frac * (d_sd[k] - c_sd[k])) if k.startswith("mlp.") else c_sd[k] for k in c_sd}
        alg.actor.load_state_dict(sd, strict=False)
        with torch.no_grad():
            alg.actor.distribution.std_param.fill_(a.std)
        obs = venv.get_observations().to("cuda:0")
        gs, ga, mse = [], [], []
        for b in range(a.warmup + a.batches):
            with torch.inference_mode():
                for _ in range(cfg["num_steps_per_env"]):
                    act = alg.act(obs)
                    obs, rew, dones, extras = venv.step(act)
                    alg.process_env_step(obs, rew, dones, extras)
                alg.compute_returns(obs)
            if b >= a.warmup:
                for batch in alg.storage.mini_batch_generator(alg.num_mini_batches, 1):
                    alg.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                              stochastic_output=True)
                    logp = alg.actor.get_output_log_prob(batch.actions)
                    ratio = torch.exp(logp - torch.squeeze(batch.old_actions_log_prob))
                    surrogate = (-torch.squeeze(batch.advantages) * ratio).mean()
                    with torch.no_grad():
                        mu_c = alg.anchor_mlp(batch.observations["anchor"])
                    anchor = (alg.actor.output_mean - mu_c).pow(2).mean()
                    gs.append(grad_norm(surrogate, params))
                    ga.append(grad_norm(anchor, params))
                    mse.append(float(anchor))
            alg.storage.clear()
        row = {"frac": frac, "g_surrogate": sum(gs) / len(gs), "g_anchor_beta1": sum(ga) / len(ga),
               "anchor_mse": sum(mse) / len(mse), "n_minibatches": len(gs)}
        row["beta"] = row["g_surrogate"] / row["g_anchor_beta1"]
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
