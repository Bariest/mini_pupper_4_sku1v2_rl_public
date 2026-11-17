#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One table of every stage's simulation tests (train.ps1's last step), printed and optionally written as Markdown.

    python tools/report.py [--write out/REPORT.md]

Reads what the stages leave behind; a stage that has not run shows as "not run". numpy-free.
  walk    walk/runs/sku1v2_405_walk/ppo/<run>/eval_omni_model_1299.json   (scripts/evaluate_omni.py)
  stairs  walk/runs/eval/sku1v2_w405_stairs_<iter>_<layout>_seed<k>.json  (tools/terrain_eval.py)
  walker, get-up: recovery/tools/sku1v2_report.py                         (runs/motions, runs/eval)
  export  out/MANIFEST.json, out/choices.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WALK = ROOT / "walk"
REC = ROOT / "recovery"
OUT = ROOT / "out"
STAIRS_TAG = "w405_"
LAYOUTS = {"printed_r25_t50_x4": "4 x 25 mm risers, 50 mm treads", "printed_r20_25_25": "3 steps, 20 / 25 / 25 mm"}


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f} %"


def walk_rows() -> list[str]:
    ppo = WALK / "runs" / "sku1v2_405_walk" / "ppo"
    runs = sorted(ppo.glob("*_stops_s*_sku1v2")) if ppo.exists() else []
    runs = [r for r in runs if not r.name.endswith("_interrupted")]
    if not runs:
        return ["not run"]
    rows = ["| seed | omni gates | joystick falls | run |", "|---|---|---|---|"]
    for r in runs:
        ev = load(r / "eval_omni_model_1299.json")
        seed = r.name.split("_stops_s")[1].split("_")[0]
        if ev is None:
            rows.append(f"| {seed} | not tested | - | `{r.name}` |")
            continue
        rows.append(f"| {seed} | {'PASS' if ev.get('pass') else 'fail'} | {pct(ev.get('joystick', {}).get('fall_frac'))} "
                    f"| `{r.name}` |")
    return rows


def stairs_rows() -> list[str]:
    ev = WALK / "runs" / "eval"
    rows = ["| policy | staircase | success | fall | stall |", "|---|---|---|---|---|"]
    found = False
    for name, label in ((f"{STAIRS_TAG}stairs_4500", "SKU1 v2, iteration 4500"),
                        (f"{STAIRS_TAG}stairs_4897", "SKU1 v2, iteration 4897"),
                        (f"{STAIRS_TAG}sku2policy", "SKU2's stairs policy, unchanged (reference)")):
        for lay, lay_label in LAYOUTS.items():
            cells = [load(ev / f"sku1v2_{name}_{lay}_seed{s}.json") for s in (1, 2)]
            cells = [c["cells"][0] for c in cells if c and c.get("cells")]
            if not cells:
                continue
            found = True
            m = {k: sum(c[k] for c in cells) / len(cells) for k in ("success", "fall", "stall")}
            rows.append(f"| {label} | {lay_label} | {pct(m['success'])} | {pct(m['fall'])} | {pct(m['stall'])} |")
    return rows if found else ["not run"]


def recovery_text() -> str:
    env = dict(os.environ, MP4R_ROBOT="sku1v2")
    r = subprocess.run([sys.executable, "-B", "tools/sku1v2_report.py"], cwd=REC, env=env,
                       capture_output=True, text=True)
    return (r.stdout or r.stderr).rstrip() or "not run"


def export_rows() -> list[str]:
    m = load(OUT / "MANIFEST.json")
    if not m:
        return ["not run"]
    rows = ["| file | policy | from | sha256 |", "|---|---|---|---|"]
    for f, v in m.get("files", {}).items():
        rows.append(f"| `{f}` | {v['name']} | `{v['source']}` | `{v['sha256'][:16]}` |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", type=Path, help="also write the report here (Markdown)")
    a = ap.parse_args()
    choices = load(OUT / "choices.json") or {}
    parts = [
        "# SKU1 v2 training report", "",
        "Simulation tests of every stage; docs/PIPELINE.md says what each test does. Choices: "
        + (", ".join(f"{k} = {v}" for k, v in choices.items()) or "none yet"), "",
        "## 1. Walk (flat, omnidirectional)", "", *walk_rows(), "",
        "## 2. Stairs (the printed staircases, 2 seeds x 256 robots, 30 s)", "", *stairs_rows(), "",
        "## 3-4. Walker and get-up", "", "```", recovery_text(), "```", "",
        "## 5. Files for the robot (`out/`)", "", *export_rows(), "",
    ]
    text = "\n".join(parts)
    print(text)
    if a.write:
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
