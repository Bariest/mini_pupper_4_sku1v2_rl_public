#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn a flat walk trained in walk/ (RLmp4, scripts/export_policy.py) into the recovery repository's walker base.

    python tools/walk_to_walker.py walk/policies/sku1v2_405_walk_s2 recovery/policies/sku1v2_walker_base \
        --contract recovery/src/mp4_recovery/data/sku1v2_contract.json

SKU2's recipe does not train its walker in the recovery repository either: it takes a finished omni walk and writes the
joint bounds and a 12 deg wider abduction stance into its metadata (recovery/scripts/recipes/walker.sh). This step makes
that walk readable by the recovery tools: the network, the action scale and the fixtures are copied unchanged, and the
metadata says what the recovery runner checks (the contract it runs on, the robot, the walking band). The walk and the
recovery contract must agree on the servo signs, the stand and the calibration; this is checked against the walk's
own contract (--walk-contract), so a walk trained on another calibration is refused instead of silently re-centred.

Then, as walker.sh:
    python -m mp4_recovery.export_policy --from-policy policies/sku1v2_walker_base policies/sku1v2_walker \
        --limits --stand-offset-abd 12
numpy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WALK_BAND_DEG = [20.0 if i % 3 == 0 else 60.0 for i in range(12)]     # abduction 20, hip and knee 60 (the walk's)
OBS = "ang_vel 3, gravity 3, command 3, pos 12, vel 12, prev_action 12 (contract.py)"


def lf_sha256(path: Path) -> str:
    """The recovery tools hash contracts with LF line endings (a Windows checkout must not change the hash)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def same(a, b, tol=1e-4) -> bool:
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walk", type=Path, help="the walk's policy directory (policy.npz from walk/scripts/export_policy.py)")
    ap.add_argument("out", type=Path, help="the walker base to write (a policy directory)")
    ap.add_argument("--contract", type=Path,
                    default=ROOT / "recovery/src/mp4_recovery/data/sku1v2_contract.json",
                    help="the recovery contract the walker runs on")
    ap.add_argument("--walk-contract", type=Path,
                    default=ROOT / "walk/mp3_walk_policy/data/sku2_contract.json",
                    help="the contract the walk was trained with (walk/ uses SKU2's signs and stand for SKU1 v2)")
    ap.add_argument("--config", type=Path, default=ROOT / "tools/templates/sku1v2_walker_config.json",
                    help="the config.json the recovery evaluations read (the 405 g SKU1 v2 walk's options)")
    ap.add_argument("--robot", default="sku1v2")
    a = ap.parse_args()

    src = a.walk / "policy.npz"
    if not src.exists():
        print(f"no {src}", file=sys.stderr)
        return 1
    d = dict(np.load(src, allow_pickle=False))
    meta = json.loads(str(d["meta"]))
    n = int(d["n_layers"])
    dims = [d["W0"].shape[1]] + [d[f"W{i}"].shape[0] for i in range(n)]
    hist = int(meta.get("history", 0) or 0)
    if hist or dims[0] != 45 or dims[-1] != 12:
        print(f"{a.walk}: network {dims}, history {hist}: the walker must be a flat walk (45 inputs, no history)",
              file=sys.stderr)
        return 1
    if meta.get("activation", "elu") != "elu":
        print(f"{a.walk}: activation {meta.get('activation')}: the recovery runner runs ELU", file=sys.stderr)
        return 1
    trained = meta.get("robot")
    if trained and trained != a.robot:
        print(f"{a.walk} was trained for {trained}, not {a.robot}", file=sys.stderr)
        return 1

    rc = json.loads(a.contract.read_text())
    wc = json.loads(a.walk_contract.read_text())
    for key in ("signs", "offsets_deg", "stand_dd", "stand_policy_rad"):
        if not same(rc[key], wc[key]):
            print(f"the contracts differ in {key}:\n  recovery {rc[key]}\n  walk     {wc[key]}\n"
                  "the walk would stand differently on the recovery runner; make the contracts agree first "
                  "(recovery/tools/make_contract.py)", file=sys.stderr)
            return 1

    hidden = [int(d[f"W{i}"].shape[0]) for i in range(n - 1)]
    new_meta = {
        "contract_sha256": lf_sha256(a.contract),
        "robot": a.robot,
        "obs": meta.get("obs", OBS),
        "activation": "elu",
        "omni": bool(meta.get("omni", True)),
        "history": 0,
        "kind": "walk",
        "band_deg": WALK_BAND_DEG,
        "limits": False,
        "bounds": None,
        "stand_offset_abd_deg": 0.0,
        "action_filter_alpha": 0.0,
        "hidden": hidden,
        "source": f"walk/{a.walk.as_posix().split('walk/', 1)[-1]} (RLmp4 flat omni walk, SKU2's recipe on SKU1 v2)",
        "source_policy_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": meta.get("checkpoint_sha256"),
        "trained": "walk/ (RLmp4): bam_t5ub06 servo model, the robot's mass file, 8-19 ms sensing chain",
    }
    d["meta"] = np.array(json.dumps(new_meta))
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "policy.npz", **d)
    shutil.copy(a.config, a.out / "config.json")
    print(f"WROTE {a.out / 'policy.npz'}: net {dims}, {d['fixture_obs'].shape[0]} fixtures, "
          f"contract {new_meta['contract_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
