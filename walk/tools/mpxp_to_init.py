#!/usr/bin/env python
"""A training start from a robot policy file: an mpv2onlyRL ``.mpxp`` walk (float32 MLP, ELU) written as a checkpoint
that ``scripts/train.py --clone`` warm-starts from and anchors to (``actor_state_dict``: ``mlp.{0,2,4,6}`` and the
exploration std), checked against the file's own fixtures.

    python -B tools/mpxp_to_init.py default.mpxp runs/sprint/init/walk_s2.pt

The sprint bundle ships ``runs/sprint/init/walk_s2.pt``, made this way from mpv2onlyRL's ``default.mpxp``
("Walk - SKU1 v2 - 405 g", sku1v2_405_walk_s2) - the walk that is on the robot.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def read_mpxp(path):
    b = Path(path).read_bytes()
    magic = struct.unpack_from("<I", b, 0)[0]
    if magic != 0x5058504D:
        raise SystemExit(f"{path}: not an .mpxp file")
    name = struct.unpack_from("48s", b, 16)[0].split(b"\0")[0].decode()
    n_layers = struct.unpack_from("<I", b, 80)[0]
    dims = struct.unpack_from("<9I", b, 84)[: n_layers + 1]
    activation, _flags, n_fix, history = struct.unpack_from("<4I", b, 120)
    if activation != 1 or history not in (0, 1):
        raise SystemExit(f"{path}: activation {activation}, history {history}: not a 45-input ELU walk")
    scale = np.array(struct.unpack_from("<12f", b, 264), dtype=np.float32)
    off, layers = 640, []
    for i in range(n_layers):
        n_in, n_out = dims[i], dims[i + 1]
        w = np.frombuffer(b, "<f4", n_in * n_out, off).reshape(n_out, n_in).copy()
        off += 4 * n_in * n_out
        bias = np.frombuffer(b, "<f4", n_out, off).copy()
        off += 4 * n_out
        layers.append((w, bias))
    fix_obs = np.frombuffer(b, "<f4", n_fix * dims[0], off).reshape(n_fix, dims[0]).copy()
    off += 4 * n_fix * dims[0]
    fix_act = np.frombuffer(b, "<f4", n_fix * 12, off).reshape(n_fix, 12).copy()
    return {"name": name, "dims": list(dims), "layers": layers, "scale": scale,
            "fix_obs": fix_obs, "fix_act": fix_act}


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    p = read_mpxp(src)
    if p["dims"] != [45, 128, 128, 64, 12]:
        raise SystemExit(f"{src}: layers {p['dims']}, expected 45-128-128-64-12")
    sd = {"distribution.std_param": torch.full((12,), std)}
    for i, (w, bias) in enumerate(p["layers"]):
        sd[f"mlp.{2 * i}.weight"] = torch.from_numpy(w)
        sd[f"mlp.{2 * i}.bias"] = torch.from_numpy(bias)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": sd, "source": str(src.name), "policy": p["name"]}, dst)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mp3_walk_policy.distill import load_actor
    actor = load_actor(dst, device="cpu")
    with torch.no_grad():
        act = actor(torch.from_numpy(p["fix_obs"])).numpy()
    err = float(np.abs(act - p["fix_act"]).max())
    if err > 1e-5:
        raise SystemExit(f"{dst}: the checkpoint's actor differs from the file's fixtures by {err:.2e}")
    info = {"policy": p["name"], "source": src.name, "fixtures": int(len(p["fix_obs"])), "max_error": err,
            "action_scale": [float(v) for v in p["scale"]], "init_std": std}
    dst.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"{dst}: {p['name']}, {len(p['fix_obs'])} fixtures, max error {err:.1e}")


if __name__ == "__main__":
    main()
