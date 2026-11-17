#!/usr/bin/env python
"""PPO on the SKU2 terrain task (``mp3_walk_policy.terrain_task``), from scratch or from a trot-derived policy with an
imitation term that fades out.

    python scripts/train_terrain.py --variant A --seed 1
    ... --variant C --seed 1           # warm start and anchor: policies/bam_t5ub06_stops_v1, rescaled; beta 5 -> 0

Variants (presets; every option can be set explicitly and goes into ``<run>/config.json``):
- A: the ``bam_t5ub06`` actuator model, from scratch, the 45-float observation.
- B: A with a 5-step observation history in the actor.
- C: A, warm-started from an omni policy (``--init``; default ``bam_t5ub06_stops_v1``: the firmware trot cloned on a
  (vx, vy, wz) grid, then PPO), its output layer rescaled per joint to this task's action scale, and anchored to it
  with beta ``--beta0`` held ``--beta-hold`` iterations then fading to 0 over ``--beta-decay``.
- D: A with the stock servo.

``--resume <run>/model_k.pt`` fine-tunes a terrain checkpoint instead: actor, critic, std, optimizer, the iteration
and the noise ramp's step counter come from it, the history from its run's config; plain PPO (no anchor), lr0 1e-4.
``--heading-obs`` from a run without it widens the actor and the critic with zero columns for the heading error
(``widen_for_heading``: the same actions and values at iteration 0) and drops the optimizer state.

Network 128-128-64 ELU for actor and critic (the robot runner's size); PPO lr 1e-3 adaptive (KL 0.01), gamma 0.99,
lambda 0.95, clip 0.2, entropy 0.005, 5 epochs x 4 minibatches, 24 steps per env per iteration. From scratch the
exploration std starts at ``--init-std`` 0.5 (learned); from a warm start at 0.1.

Training refuses to start while mujoco_warp's heightfield-overflow print is in place (``tools/patch_mujoco_warp.py``;
``--allow-unpatched`` overrides).
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

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                   # noqa: E402
from mjlab.rl import (MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg,  # noqa: E402
                      RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)

from mp3_walk_policy.contract import has_heading_obs                      # noqa: E402
from mp3_walk_policy.env_cfg import describe, noise_profile               # noqa: E402
from mp3_walk_policy import paths                                         # noqa: E402
from mp3_walk_policy.paths import sha256                                  # noqa: E402
from mp3_walk_policy.terrain_task import (TERRAIN_ACTION_SCALE, TERRAINS, TerrainOptions,  # noqa: E402
                                         make_terrain_env_cfg)

PREV_ACTION = slice(33, 45)        # contract.OBS_TERMS: the previous action in the 45-float observation
VARIANTS = {
    "A": dict(arm="bam_t5ub06", history=0, init=None),
    "B": dict(arm="bam_t5ub06", history=5, init=None),
    "C": dict(arm="bam_t5ub06", history=0, init="policies/bam_t5ub06_stops_v1"),
    "D": dict(arm="stock", history=0, init=None),
}


def npz_to_checkpoint(policy_dir: Path, new_scale, out: Path, history: int = 0) -> Path:
    """An exported policy as an rsl_rl actor checkpoint in a new action scale: output rows times k = old/new per joint,
    and the first layer's previous-action columns divided by k, so the same robot state gives the same target in
    radians. With ``history`` > 0 the first layer is widened to the stacked observation (``contract.HistoryStacker``:
    each term's frames oldest to newest): the original weights on each term's newest frame, zeros on the older
    frames, so the warm start acts exactly as the original until training uses the history."""
    d = np.load(policy_dir / "policy.npz", allow_pickle=False)
    n = int(d["n_layers"])
    k = np.asarray(d["scale"], dtype=np.float32) / np.asarray(new_scale, dtype=np.float32)
    sd = {}
    for i in range(n):
        w, b = d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32)
        if i == 0:   # the observed previous action is target / scale: in the new units it is k times larger
            w = w.copy()
            w[:, PREV_ACTION] = w[:, PREV_ACTION] / k[None, :]
            if history:
                from mp3_walk_policy.contract import has_heading_obs, obs_terms
                he = has_heading_obs(json.loads(str(d["meta"])))
                wide, col = np.zeros((w.shape[0], w.shape[1] * history), dtype=np.float32), 0
                for _, sl in obs_terms(he):
                    size = sl.stop - sl.start
                    wide[:, col + (history - 1) * size: col + history * size] = w[:, sl]
                    col += history * size
                w = wide
        if i == n - 1:
            w, b = w * k[:, None], b * k
        sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"] = torch.tensor(w), torch.tensor(b)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": sd, "source": str(policy_dir), "rescale": k.tolist()}, out)
    return out


def prev_action_columns(env, group: str) -> np.ndarray:
    """The columns of ``group``'s flattened observation holding the previous action, (frames, 12): each term's history
    is stored frame by frame (oldest first), so frame f's joint j is at the term's offset + 12 f + j."""
    om = env.observation_manager
    off = 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        size = int(np.prod(dims))
        if name == "prev_action":
            return off + np.arange(size).reshape(-1, 12)
        off += size
    raise KeyError(f"no prev_action in the {group!r} observation")


def rescale_checkpoint(src: Path, env, old_scale, new_scale, out: Path) -> Path:
    """A terrain checkpoint in a new action scale, acting exactly as before in radians: with k = old / new per joint,
    the actor's output rows and its std times k, the previous-action input columns (every history frame, in the actor
    and the critic) divided by k; Adam's moments transformed with their parameters (m / s, v / s^2 for p -> s p), so the
    optimizer state stays consistent. The changed columns are saved in the checkpoint under ``rescale``."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    k = torch.tensor(np.asarray(old_scale, dtype=np.float32) / np.asarray(new_scale, dtype=np.float32))
    scales = {"actor_state_dict": {}, "critic_state_dict": {}}
    cols = {g: prev_action_columns(env, g) for g in ("actor", "critic")}
    a, c = ck["actor_state_dict"], ck["critic_state_dict"]
    last = max((n for n in a if n.startswith("mlp.") and n.endswith(".weight")), key=lambda n: int(n.split(".")[1]))
    sa = {n: torch.ones_like(v) for n, v in a.items()}
    sa[last] = k[:, None].expand_as(a[last]).clone()
    sa[last.replace("weight", "bias")] = k.clone()
    sa["distribution.std_param"] = k.clone()
    for g, sd, sc in (("actor", a, sa), ("critic", c, None)):
        if sc is None:
            sc = {n: torch.ones_like(v) for n, v in sd.items()}
        for f in range(cols[g].shape[0]):
            sc["mlp.0.weight"][:, cols[g][f]] = 1.0 / k[None, :]
        scales["actor_state_dict" if g == "actor" else "critic_state_dict"] = sc
    for key, sc in scales.items():
        for n in ck[key]:
            ck[key][n] = ck[key][n] * sc[n]
    params = [scales["actor_state_dict"][n] for n in a] + [scales["critic_state_dict"][n] for n in c]
    st = ck["optimizer_state_dict"]["state"]
    for i, s in enumerate(params):
        if i in st:
            st[i]["exp_avg"] = st[i]["exp_avg"] / s.to(st[i]["exp_avg"].device)
            st[i]["exp_avg_sq"] = st[i]["exp_avg_sq"] / (s ** 2).to(st[i]["exp_avg_sq"].device)
    ck["rescale"] = {"source": str(src), "k": k.tolist(), "old_scale": list(old_scale), "new_scale": list(new_scale),
                     "prev_action_cols": {g: cols[g].tolist() for g in cols}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def widen_for_privileged(src: Path, env, out: Path) -> Path:
    """A blind terrain checkpoint as a teacher's starting point: the actor's first layer widened with zero columns for
    the "privileged" group (appended after the actor group), so it acts exactly as before until training uses the
    terrain; the optimizer state is dropped (its first-layer moments no longer fit)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    w = ck["actor_state_dict"]["mlp.0.weight"]
    extra = int(np.prod(env.observation_manager.group_obs_dim["privileged"]))
    ck["actor_state_dict"]["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    ck.pop("optimizer_state_dict", None)
    ck["widened"] = {"source": str(src), "privileged_inputs": extra}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def widen_critic(src: Path, env, out: Path, extra: int) -> Path:
    """A terrain checkpoint whose critic now also observes ``extra`` trailing inputs (the cross-track drift,
    ``track_errors``, appended last to the critic group): zero columns, so the value is unchanged until training uses
    them; the optimizer state is dropped (its first-layer moments no longer fit)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    w = ck["critic_state_dict"]["mlp.0.weight"]
    want = int(np.prod(env.observation_manager.group_obs_dim["critic"]))
    assert w.shape[1] + extra == want, (w.shape, extra, want)
    ck["critic_state_dict"]["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    ck.pop("optimizer_state_dict", None)
    ck["critic_widened"] = {"source": str(src), "extra_inputs": extra}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def insert_zero_columns(sd: dict, pos: int, n: int) -> dict:
    """``sd`` (an rsl_rl model's state dict) with ``n`` zero input columns inserted before input ``pos`` of its first
    layer (``mlp.0.weight``); an observation normaliser, if any, gets mean 0 and variance 1 there. The output is the
    same for any value of the new inputs."""
    out = dict(sd)
    w = sd["mlp.0.weight"]
    assert 0 <= pos <= w.shape[1], (pos, w.shape)
    out["mlp.0.weight"] = torch.cat([w[:, :pos], w.new_zeros(w.shape[0], n), w[:, pos:]], dim=1)
    for k, v in sd.items():
        if "normalizer" in k and k.rsplit(".", 1)[-1] in ("_mean", "_var", "_std"):
            assert v.shape[-1] == w.shape[1], (k, v.shape)
            fill = v.new_zeros(*v.shape[:-1], n) if k.endswith("_mean") else v.new_ones(*v.shape[:-1], n)
            out[k] = torch.cat([v[..., :pos], fill, v[..., pos:]], dim=-1)
    return out


def heading_columns(env) -> dict:
    """Per group, (where the heading error's inputs start, how many): the term is the actor group's last (its history
    frames one block; any "privileged" group comes after it in the actor's input) and the critic's last."""
    om = env.observation_manager
    out = {}
    for g in ("actor", "critic"):
        off = 0
        for name, dims in zip(om.active_terms[g], om.group_obs_term_dim[g]):
            size = int(np.prod(dims))
            if name == "heading_error":
                out[g] = (off, size)
            off += size
        assert g in out and out[g][0] + out[g][1] == off, (g, om.active_terms[g])
    return out


def widen_for_heading(src: Path, env, out: Path) -> Path:
    """A checkpoint from a run without ``heading_obs`` for one with it: zero columns where the heading error's inputs
    are (``heading_columns``), so actions and values are unchanged at iteration 0; the optimizer state is dropped (its
    first-layer moments no longer fit). Run before the other widenings (they append after it): in a group whose later
    terms the source also lacks (the critic's ``track_errors``) the columns go at its end (every new column is 0)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cols = heading_columns(env)
    for g in ("actor", "critic"):
        pos, n = cols[g]
        sd = ck[f"{g}_state_dict"]
        ck[f"{g}_state_dict"] = insert_zero_columns(sd, min(pos, sd["mlp.0.weight"].shape[1]), n)
    ck.pop("optimizer_state_dict", None)
    ck["heading_widened"] = {"source": str(src), "columns": {g: list(v) for g, v in cols.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def runner_cfg(seed, iters, init_std, anchored: bool, logger="tensorboard", wandb_project="mp3_walk_policy",
               privileged: bool = False):
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=24, max_iterations=iters, save_interval=50,
        experiment_name="mp3_terrain", logger=logger, wandb_project=wandb_project, clip_actions=1.0,
        obs_groups={"actor": ("actor", "privileged") if privileged else ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": init_std,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
            **({"class_name": "mp3_walk_policy.algorithms:AnchorPPO"} if anchored else {})),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True,
                    help="A: from scratch; B: from scratch with a 5-step history; C: warm start from the omni policy "
                         "(the rough-ground and stair recipe); D: from scratch with the stock servo")
    ap.add_argument("--arm")
    ap.add_argument("--history", type=int)
    ap.add_argument("--init", help="a policies/<name> directory to warm-start from (and anchor to)")
    ap.add_argument("--resume", help="a terrain run's model_<k>.pt to fine-tune (overrides --init and --history)")
    ap.add_argument("--action-scale", type=float, nargs=3, metavar=("ABD", "HIP", "KNEE"),
                    help="rad per unit action (default 0.25 0.5 0.5); with --resume from another scale the checkpoint "
                         "is rescaled to act as before (rescale_checkpoint)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--init-std", type=float)
    ap.add_argument("--beta0", type=float, default=5.0)
    ap.add_argument("--beta-hold", type=int, default=50)
    ap.add_argument("--beta-decay", type=int, default=300)
    ap.add_argument("--critic-warmup", type=int,
                    help="value-only updates before the actor moves (default 50 with --init, else 0); the anchor's hold "
                         "starts after them")
    ap.add_argument("--lr0", type=float,
                    help="the adaptive learning rate's start (default 1e-4 with --init, else 1e-3). A warm start's first "
                         "update at 1e-3 with an untrained critic can stop the warm-started trot from walking")
    ap.add_argument("--noise", default="n2", choices=("n1", "n2"))
    ap.add_argument("--masses", help="mass file in mp3_walk_policy/data (default: the terrain task's: paths.TERRAIN_MASSES, by MP3_ROBOT)")
    ap.add_argument("--terrain", default="rough_basic", choices=TERRAINS)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--heading", default="integral", choices=("integral", "loop"),
                    help="integral: the HeadingTrack penalty; loop: wz from a simulated heading loop, as on the robot")
    ap.add_argument("--alive", type=float, default=0.0, help="reward weight per second not terminated")
    ap.add_argument("--terminated", type=float, default=0.0,
                    help="reward weight on a fall (not a time-out); rewards are scaled by dt, so -250 is -5 per fall")
    ap.add_argument("--progress", type=float, default=0.0, help="weight of the linear progress term (per second)")
    ap.add_argument("--stumble", type=float, default=0.0, help="weight per second per foot pushed mostly sideways")
    ap.add_argument("--clearance", default="speed", choices=("speed", "apex"),
                    help="speed: |h - 30 mm| x foot speed; apex: the swing apex's deficit below 25 mm at touchdown, "
                         "against the terrain under and 3 cm ahead of the foot")
    ap.add_argument("--clearance-weight", type=float, default=-4.0, help="apex: per metre of deficit per touchdown")
    ap.add_argument("--clearance-target", type=float, default=0.025, help="apex: the swing apex wanted (m)")
    ap.add_argument("--privileged", action="store_true",
                    help="a teacher: the actor also observes the terrain (body scan, foot heights; simulation only); "
                         "with --resume from a blind run the checkpoint is widened (widen_for_privileged)")
    ap.add_argument("--tilt-limit", type=float, default=0.7,
                    help="rad: the tilt from vertical that ends an episode as a fall (a 27-degree flight pitches the "
                         "body by itself)")
    ap.add_argument("--stair-exit", action="store_true",
                    help="jittered-stair tiles end the episode once off the stairs (a success): promotion on that, "
                         "demotion on a fall or never leaving the platform (terrain_task.stair_exit)")
    ap.add_argument("--heading-law", choices=("p", "pi"), default="p",
                    help="the simulated heading loop: p = mjlab's clip(2 err, +-0.6); pi = the robot's (heading.pi_step: "
                         "PI with anti-windup, clipped at --heading-wz)")
    ap.add_argument("--heading-wz", type=float, default=0.4, help="pi: the loop's output limit (rad/s)")
    ap.add_argument("--cross-track", type=float, default=0.0,
                    help="weight per second of the clipped cross-track drift (|drift| / --cross-track-clip, at most 1; "
                         "the critic also observes the drift; a resumed critic is widened)")
    ap.add_argument("--cross-track-clip", type=float, default=0.10, help="m: the drift at which that term saturates")
    ap.add_argument("--yaw-catch", type=float, nargs=2, metavar=("LO", "HI"),
                    help="N m: yaw torque pulses on the base, |torque| U(LO, HI), random sign, 0.2-0.4 s every 3-6 s "
                         "(terrain_task.YawCatch: a foot catching a step edge)")
    ap.add_argument("--symmetry", action="store_true",
                    help="left-right mirror data augmentation (rsl_rl symmetry_cfg; mp3_walk_policy.symmetry.augment): "
                         "every PPO sample is trained on as observed and mirrored")
    ap.add_argument("--heading-obs", action="store_true",
                    help="the actor and critic observe the heading loop's error (terrain_task.obs_heading_error, the "
                         "last term: 46 floats per frame; needs --heading loop); a resumed or warm-started network is "
                         "widened with zero columns (widen_for_heading)")
    ap.add_argument("--heading-reward", type=float, default=0.0,
                    help="weight per second of heading_align, exp(-(err / 0.25)^2), on the heading loop's envs")
    ap.add_argument("--yaw-catch-after", type=int, default=0,
                    help="iterations of this run before the yaw pulses start")
    ap.add_argument("--foot-softness", type=float, nargs=2, help="the feet's solref timeconst range drawn per env (s)")
    ap.add_argument("--friction", type=float, nargs=2, help="the feet's friction range (default 0.5 1.3)")
    ap.add_argument("--leg-length-mm", type=float, default=0.0,
                    help="each shin, knee to foot, +- this (mm), drawn per env and foot (a robot unlike the model)")
    ap.add_argument("--foot-radius-scale", type=float, nargs=2, metavar=("LO", "HI"),
                    help="each foot sphere's radius times U(LO, HI), per env and foot (shoes, wear)")
    ap.add_argument("--servo-strength", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the servos' supply voltage times U(LO, HI), per env (stronger or weaker servos)")
    ap.add_argument("--vx-max", type=float, help="the forward command's top (default 0.15 m/s)")
    ap.add_argument("--abduction", type=float, default=0.0,
                    help="weight per second of the abductions' squared distance from the stand (rad^2)")
    ap.add_argument("--leg-contact", type=float, default=-0.5,
                    help="weight per second per thigh or shin capsule touching the terrain")
    ap.add_argument("--narrow-risers", type=float, nargs=2,
                    help="--terrain stairs_narrow: the flights' nominal riser range, difficulty 0 to 1 (m; default 0.010 0.035)")
    ap.add_argument("--narrow-treads", type=float, nargs=2,
                    help="--terrain stairs_narrow: the flights' tread range (m; default 0.045 0.080)")
    ap.add_argument("--narrow-share", type=float, default=0.20,
                    help="--terrain stairs_narrow: the flights' spawn proportion (default 0.20)")
    ap.add_argument("--stall-s", type=float, default=0.0,
                    help="a stall ends the episode as a failure: commanded forward and not 2 cm (--stall-m) further "
                         "along the heading in this many seconds (0: off; with --terminated it is penalised as a fall)")
    ap.add_argument("--stall-m", type=float, default=0.02, help="m: the advance that restarts the stall clock")
    ap.add_argument("--stall-stairs-axis", action="store_true",
                    help="on the narrow flights the stall is measured along the start heading (the flight), not the current one")
    ap.add_argument("--heading-fail", type=float, default=0.0,
                    help="rad: on the narrow flights, turned more than this from the start heading ends the episode as a failure")
    ap.add_argument("--upright-roll-only-on-stairs", action="store_true",
                    help="on stair tiles the upright term penalises only the roll (the pitch may follow the stairs)")
    ap.add_argument("--base-com-x", type=float, nargs=2, metavar=("LO", "HI"),
                    help="the base COM x offset drawn per robot, added to the mass file's (m; default -0.005 0.005)")
    ap.add_argument("--max-init-level", type=int, default=2,
                    help="envs start on terrain levels drawn from 0..this (levels are not saved in a checkpoint: a "
                         "resumed policy re-climbs from there)")
    ap.add_argument("--out", default=str(REPO / "runs/terrain"))
    ap.add_argument("--device", default="cuda:0", help="cuda:0; cpu for a slow check without a GPU")
    ap.add_argument("--tag", default="")
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard",
                    help="wandb also writes the TensorBoard events (needs `wandb login` and WANDB_MODE not disabled)")
    ap.add_argument("--wandb-project", default="mp3_walk_policy")
    ap.add_argument("--wandb-entity", help="the W&B user or team (default: the account's default entity)")
    ap.add_argument("--allow-unpatched", action="store_true",
                    help="train even if mujoco_warp still prints its heightfield-overflow warning")
    a = ap.parse_args()
    if not a.allow_unpatched:
        sys.path.insert(0, str(REPO / "tools"))
        from patch_mujoco_warp import is_patched
        if not is_patched():
            raise SystemExit("mujoco_warp is not patched: run python tools/patch_mujoco_warp.py (see README)")
    if a.logger == "wandb" and os.environ.get("WANDB_MODE") in ("disabled", "offline"):
        raise SystemExit("--logger wandb with WANDB_MODE=%s: unset it to log online" % os.environ["WANDB_MODE"])
    if a.wandb_entity:
        os.environ["WANDB_USERNAME"] = a.wandb_entity           # rsl_rl reads the entity from it
    if a.logger == "wandb":
        from mp3_walk_policy import wandb_compat
        wandb_compat.patch()
    v = dict(VARIANTS[a.variant])
    for k in ("arm", "history", "init"):
        if getattr(a, k) is not None:
            v[k] = getattr(a, k)
    if a.resume:
        src = json.loads((Path(a.resume).parent / "config.json").read_text())
        src_robot = src["task"].get("sources", {}).get("robot", "sku2")
        if src_robot != paths.ROBOT:
            raise SystemExit(f"--resume from a {src_robot} run with MP3_ROBOT={paths.ROBOT}: set MP3_ROBOT={src_robot}")
        v.update(arm=src["variant_settings"]["arm"], history=src["variant_settings"]["history"], init=None)
    init_std = a.init_std if a.init_std is not None else (0.1 if v["init"] else 0.5)
    warmup = a.critic_warmup if a.critic_warmup is not None else (50 if v["init"] else 0)
    lr0 = a.lr0 if a.lr0 is not None else (1.0e-4 if v["init"] or a.resume else 1.0e-3)

    opt = TerrainOptions(arm=v["arm"], num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True,
                         noise_end=noise_profile(a.noise), history=v["history"],
                         **({"masses_json": a.masses} if a.masses else {}),
                         terrain=a.terrain, rows=a.rows, terrain_seed=a.terrain_seed,
                         action_scale=tuple(a.action_scale) * 4 if a.action_scale else TERRAIN_ACTION_SCALE,
                         heading=a.heading, alive=a.alive,
                         terminated=a.terminated, progress=a.progress, stumble=a.stumble, clearance=a.clearance,
                         clearance_weight=a.clearance_weight, clearance_target=a.clearance_target,
                         stair_exit=a.stair_exit, privileged=a.privileged, heading_law=a.heading_law,
                         heading_wz=a.heading_wz, cross_track=a.cross_track, cross_track_clip=a.cross_track_clip,
                         yaw_catch=tuple(a.yaw_catch) if a.yaw_catch else None, tilt_limit=a.tilt_limit, abduction=a.abduction, leg_contact=a.leg_contact,
                         heading_obs=a.heading_obs or bool(a.resume and src["task"]["options"].get("heading_obs")),
                         heading_reward=a.heading_reward,
                         max_init_level=a.max_init_level,
                         narrow_risers=tuple(a.narrow_risers) if a.narrow_risers else None,
                         narrow_treads=tuple(a.narrow_treads) if a.narrow_treads else None,
                         narrow_share=a.narrow_share, stall_s=a.stall_s, stall_m=a.stall_m,
                         stall_stairs_axis=a.stall_stairs_axis, heading_fail=a.heading_fail,
                         upright_roll_only_on_stairs=a.upright_roll_only_on_stairs,
                         base_com_x=tuple(a.base_com_x) if a.base_com_x else None,
                         foot_softness=tuple(a.foot_softness) if a.foot_softness else None,
                         leg_length_m=a.leg_length_mm / 1000.0,
                         foot_radius_scale=tuple(a.foot_radius_scale) if a.foot_radius_scale else None,
                         servo_strength=tuple(a.servo_strength) if a.servo_strength else None,
                         **({"friction_range": tuple(a.friction)} if a.friction else {}),
                         **({"vx_max": a.vx_max} if a.vx_max is not None else {}))
    if a.yaw_catch:     # the env step counter continues from a resumed checkpoint (mjlab's runner restores it)
        ck = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else {}
        s0 = ((ck.get("infos") or {}).get("env_state") or {}).get("common_step_counter", 24 * int(ck.get("iter", 0)))
        opt.yaw_catch_start = int(s0) + 24 * a.yaw_catch_after    # 24 env steps per iteration (runner_cfg)
    env_cfg = make_terrain_env_cfg(opt)
    env_cfg.seed = a.seed
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{a.variant}_{v['arm']}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    log_dir = Path(a.out) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    anchored = v["init"] is not None
    cfg = asdict(runner_cfg(a.seed, a.iters, init_std, anchored, logger=a.logger, wandb_project=a.wandb_project,
                            privileged=a.privileged))
    cfg["algorithm"]["learning_rate"] = lr0
    if a.symmetry:
        cfg["algorithm"]["symmetry_cfg"] = {"use_data_augmentation": True, "use_mirror_loss": False,
                                            "data_augmentation_func": "mp3_walk_policy.symmetry:augment",
                                            "mirror_loss_coeff": 0.0}
    ckpt = None
    if anchored:
        # the anchor group has no heading term: an init policy that reads it cannot be the anchor
        init_meta = json.loads(str(np.load(REPO / v["init"] / "policy.npz")["meta"]))
        assert not has_heading_obs(init_meta), "--init from a --heading-obs policy: the anchor group has no heading term"
        ckpt = npz_to_checkpoint(REPO / v["init"], opt.action_scale, log_dir / "init_rescaled.pt", history=v["history"])
        # the anchor reads the plain 45-float observation group: its network keeps the original input layer
        anchor = ckpt if not v["history"] else npz_to_checkpoint(REPO / v["init"], opt.action_scale,
                                                                   log_dir / "anchor_rescaled.pt")
        cfg["algorithm"].update(anchor_path=str(anchor), anchor_sha256=sha256(anchor), anchor_beta0=a.beta0,
                                anchor_hold_iters=warmup + a.beta_hold, anchor_decay_iters=a.beta_decay,
                                anchor_beta_floor=0.0, anchor_obs_group="anchor", actor_warmup_iters=warmup)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
    venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(venv, cfg, str(log_dir), a.device)
    if anchored:
        start = ckpt
        if opt.heading_obs and not has_heading_obs(json.loads(str(np.load(REPO / v["init"] / "policy.npz")["meta"]))):
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)     # the actor only; the anchor keeps its 45
            pos, n = heading_columns(env)["actor"]
            ck["actor_state_dict"] = insert_zero_columns(ck["actor_state_dict"], pos, n)
            start = log_dir / "init_heading_widened.pt"
            torch.save(ck, start)
        sd = torch.load(start, map_location=a.device, weights_only=False)["actor_state_dict"]
        missing, unexpected = runner.alg.actor.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all("normalizer" in k or "std" in k for k in missing), missing
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(init_std)
    resume = a.resume
    if resume:
        old = tuple(src["task"]["options"]["action_scale"])
        if not np.allclose(old, opt.action_scale):
            resume = str(rescale_checkpoint(Path(resume), env, old, opt.action_scale, log_dir / "resume_rescaled.pt"))
        widened = False
        if opt.heading_obs and not src["task"]["options"].get("heading_obs", False):
            resume = str(widen_for_heading(Path(resume), env, log_dir / "resume_heading_widened.pt"))
            widened = True
        if opt.privileged and not src["task"]["options"].get("privileged", False):
            widened = True
            resume = str(widen_for_privileged(Path(resume), env, log_dir / "resume_widened.pt"))
        if opt.cross_track and not src["task"]["options"].get("cross_track", 0.0):
            # widen_critic appends; the critic's terms are [..., track_errors, heading_error], so a source that already
            # has the heading term would get the new columns after it, misaligned
            assert not src["task"]["options"].get("heading_obs", False), \
                "--cross-track widening from a --heading-obs run is not supported (column order)"
            resume = str(widen_critic(Path(resume), env, log_dir / "resume_critic_widened.pt", extra=3))
            widened = True
        runner.load(resume, map_location=a.device,           # strict: the same network, history and observations
                    load_cfg={"actor": True, "critic": True, "optimizer": not widened, "iteration": True})

    try:                     # a copy without git (a zipped bundle) or a PC without git: no commit recorded
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "terrain_task": True, "variant": a.variant, "variant_settings": v, "runner": cfg,
        "algorithm_class": f"{type(runner.alg).__module__}.{type(runner.alg).__name__}",
        "robot": paths.ROBOT,
        "masses_sha256": sha256(REPO / "mp3_walk_policy" / "data" / (opt.masses_json or paths.MASSES_JSON.name)),
        "init": v["init"], "init_rescaled_sha256": sha256(ckpt) if ckpt else None, "init_std": init_std,
        "resume": a.resume, "resume_sha256": sha256(Path(a.resume)) if a.resume else None,
        "resume_rescaled": resume if resume != a.resume else None,
        "critic_warmup": warmup, "lr0": lr0,
        "noise": a.noise, "repo_commit": commit, "repo_dirty": dirty, "argv": sys.argv}, indent=1, default=str))
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
