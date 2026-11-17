"""The clone: MP2's phase-2 distillation step, with the firmware trot as the teacher.

``collect``: the teacher (``teacher.py``) is played open-loop in the arm's own environment, randomised plant, clean
observations, with a small Gaussian perturbation on the executed action (so the data cover states off the nominal
trajectory). Per step: the 45-number observation, and the teacher's target in action units as the label.
``train``: MSE behaviour cloning of the actor's mean net (45 -> 128 -> 128 -> 64 -> 12, ELU: rsl_rl's MLP layout),
with MP2's input jitter at the clean noise floor (``env_cfg.NOISE_START``). The clone is saved in rsl_rl's actor
state-dict form (``mlp.N.*`` plus ``distribution.std_param``), which both the warm start and ``AnchorPPO`` load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env_cfg import NOISE_START, TaskOptions, omni_options, stops_options
from .gates import make_env
from .mdp import servo_action
from .teacher import TorchOmniTeacher, TorchTeacher

HID = (128, 128, 64)
INIT_STD = 0.8
# input jitter per observation slice (MP2's --augment): ang_vel, gravity, pos, vel, prev_action
JITTER = (((0, 3), NOISE_START["ang_vel"]), ((3, 6), NOISE_START["gravity"]), ((9, 21), NOISE_START["pos"]),
          ((21, 33), NOISE_START["vel"]), ((33, 45), 0.05))


def mlp(inp: int, out: int) -> nn.Sequential:
    layers, prev = [], inp
    for h in HID:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


@torch.no_grad()
def collect(arm: str, num_envs: int = 2048, seconds: float = 10.0, action_noise: float = 0.03, seed: int = 0,
            device: str = "cuda:0", omni: bool = False, stops: bool = False) -> dict:
    """``omni``: the omni task (one joystick command per episode) and the omni teacher. ``stops``: the omni task on
    take-like schedules (``takes.schedule_bank``, one per env), labelled by the firmware playing the same schedule
    (stops and restarts as the robot's own leg phases)."""
    torch.manual_seed(seed)
    kw = dict(arm=arm, num_envs=num_envs, noisy=False, randomise=True, noise_ramp=False, episode_s=seconds)
    if stops:
        env = make_env(stops_options(schedules=1, **kw), device)
        from .takes import schedule_bank
        cmd_bank, rel_bank = schedule_bank(num_envs, int(round(seconds / 0.02)), seed=seed, with_rel=True)
        env.command_manager.get_term("twist").set_bank(cmd_bank)
        rel_bank = torch.tensor(rel_bank, device=env.device)
        ids = torch.arange(num_envs, device=env.device)
    else:
        env = make_env(omni_options(**kw) if omni else TaskOptions(**kw), device)
    obs, _ = env.reset()
    term = servo_action(env)
    teacher = TorchOmniTeacher(env.device) if omni and not stops else TorchTeacher(env.device)
    steps = int(round(seconds / 0.02)) - 1
    O, A = [], []
    clipped = torch.zeros(12, device=env.device)
    falls = 0
    for _ in range(steps):
        cmd = env.command_manager.get_command("twist")
        if stops:
            target = rel_bank[ids, env.episode_length_buf] / term.scale
        else:
            target = teacher.target_rel(cmd if omni else cmd[:, 0], env.episode_length_buf) / term.scale
        clipped += (target.abs() > 1.0).float().sum(0)
        label = target.clamp(-1.0, 1.0)
        O.append(obs["actor"].clone())
        A.append(label)
        obs, _, terminated, _, _ = env.step(label + action_noise * torch.randn_like(label))
        falls += int(terminated.sum())
    env.close()
    O, A = torch.cat(O).cpu().numpy(), torch.cat(A).cpu().numpy()
    return {"obs": O, "act": A, "meta": {"arm": arm, "omni": omni, "stops": stops, "num_envs": num_envs, "seconds": seconds,
                                         "action_noise": action_noise, "seed": seed, "rows": int(len(O)),
                                         "falls": falls, "clip_frac": (clipped / len(O)).cpu().numpy().round(4).tolist()}}


def train(data: dict, epochs: int = 40, batch: int = 8192, lr: float = 1e-3, seed: int = 0,
          device: str = "cuda:0") -> tuple[dict, dict]:
    torch.manual_seed(seed)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(data["act"], dtype=torch.float32, device=device)
    n = len(obs)
    perm = torch.randperm(n, device=device)
    n_val = max(n // 10, 1)
    val, tr = perm[:n_val], perm[n_val:]
    net = mlp(obs.shape[1], act.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log = []
    for ep in range(epochs):
        net.train()
        p = tr[torch.randperm(len(tr), device=device)]
        losses = []
        for i in range(0, len(p), batch):
            idx = p[i:i + batch]
            x = obs[idx].clone()
            for (a, b), amp in JITTER:
                x[:, a:b] += (torch.rand_like(x[:, a:b]) * 2 - 1) * amp
            loss = nn.functional.mse_loss(net(x), act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(obs[val]), act[val]).item()
        log.append((ep, float(np.mean(losses)), v))
    sd = {f"mlp.{k}": v.detach().cpu() for k, v in net.state_dict().items()}
    sd["distribution.std_param"] = torch.full((act.shape[1],), INIT_STD)
    meta = {"epochs": epochs, "batch": batch, "lr": lr, "seed": seed, "rows": n, "final_val_mse": log[-1][2],
            "final_train_mse": log[-1][1], "hidden": list(HID), "jitter": [list(map(list, JITTER))]}
    return {"actor_state_dict": sd}, meta


def save_clone(ckpt: dict, meta: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=1))
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_actor(path, device: str = "cuda:0") -> nn.Sequential:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck["student_state_dict"]   # PPO or distillation
    w = {k.removeprefix("mlp."): v for k, v in sd.items() if k.startswith("mlp.")}
    net = mlp(w["0.weight"].shape[1], w[f"{2 * len(HID)}.weight"].shape[0]).to(device)
    net.load_state_dict(w, strict=True)
    net.eval()
    return net
