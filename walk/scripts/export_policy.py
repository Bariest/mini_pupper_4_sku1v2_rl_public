#!/usr/bin/env python
"""Export a training checkpoint as a robot policy directory: ``policy.npz`` (numpy = torch to 1e-5), the run's
``config.json`` and the checkpoint's evaluations, if any.

    python scripts/export_policy.py runs/ppo/<run>/model_1299.pt \
        policies/<name>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3_walk_policy.export import export                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    a = ap.parse_args()
    ck, out = Path(a.ckpt), Path(a.out)
    cfg = json.loads((ck.parent / "config.json").read_text())
    opts = cfg["task"]["options"]
    # terrain policies observe the whole (vx, vy, wz) command, as omni ones do
    omni = bool(opts.get("omni", False)) or bool(cfg.get("terrain_task", False))
    path = export(ck, out, opts["action_scale"], omni=omni, history=int(opts.get("history", 0) or 0),
                  privileged=bool(opts.get("privileged", False)) and not cfg.get("distillation", False),
                  heading_obs=bool(opts.get("heading_obs", False)),
                  robot=cfg["task"].get("sources", {}).get("robot", "sku2"))
    shutil.copy(ck.parent / "config.json", out / "config.json")
    for ev in ck.parent.glob(f"eval_{ck.stem}*.json"):
        shutil.copy(ev, out / ev.name)
    print("EXPORTED", path)


if __name__ == "__main__":
    main()
