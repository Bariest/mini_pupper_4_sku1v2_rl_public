"""Export a policy for the robot: the actor's mean net as numpy arrays, with what the runner needs to use it.

``export(checkpoint, out)`` writes ``out/policy.npz``: the layers (``W0, b0, ...``, ELU between them), the action
scale, the contract's hash, the observation layout (``meta``: ``history``, ``omni``, ``obs_terms_extra``), and
fixtures (observations and the torch actions for them), so that the robot side can check its numpy forward pass (``NumpyPolicy``, numpy only) against training to 1e-5.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import CONTRACT_JSON, HEADING_TERM, obs_size


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]
        self.scale = d["scale"]
        self.meta = json.loads(str(d["meta"]))

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)     # ELU
        return x


def export(checkpoint, out_dir, scale, fixtures: int = 256, seed: int = 0, omni: bool = False,
           history: int = 0, privileged: bool = False, heading_obs: bool = False, robot: str = "sku2") -> Path:
    """``omni``: the policy observes the whole command (vx, vy, wz); the runner refuses vy / wz for one that does
    not. ``privileged``: a teacher whose input also holds the simulation-only terrain group (never for the robot).
    ``heading_obs``: the observation's 46th float is the heading loop's error (meta ``obs_terms_extra``). A
    distillation checkpoint exports its student."""
    import torch
    from .distill import load_actor
    from .paths import sha256
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck["student_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    arrays = {}
    for i, idx in enumerate(lin):
        arrays[f"W{i}"] = sd[f"mlp.{idx}.weight"].numpy().astype(np.float32)
        arrays[f"b{i}"] = sd[f"mlp.{idx}.bias"].numpy().astype(np.float32)
    actor = load_actor(checkpoint, device="cpu")
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 0.5, (fixtures, arrays["W0"].shape[1])).astype(np.float32)
    assert privileged or arrays["W0"].shape[1] == obs_size(heading_obs) * max(history, 1), arrays["W0"].shape
    with torch.no_grad():
        act = actor(torch.from_numpy(obs)).numpy()
    meta = {"checkpoint": Path(checkpoint).name,                    # the file name; the digest identifies it
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "contract_sha256": sha256(CONTRACT_JSON),
            "obs": "ang_vel 3, gravity 3, command 3, pos 12, vel 12, prev_action 12"
                   + (", heading_error 1" if heading_obs else "") + " (contract.py)",
            "activation": "elu", "omni": bool(omni), "history": int(history), "privileged": bool(privileged),
            "robot": robot}
    if heading_obs:
        meta["obs_terms_extra"] = [HEADING_TERM]
    path = out / "policy.npz"
    np.savez(path, n_layers=len(lin), scale=np.asarray(scale, dtype=np.float32), meta=json.dumps(meta),
             fixture_obs=obs, fixture_act=act, **arrays)
    p = NumpyPolicy(path)
    err = float(np.abs(p(obs) - act).max())
    assert err < 1e-5, err
    return path
