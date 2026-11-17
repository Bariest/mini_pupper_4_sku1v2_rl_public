#!/usr/bin/env python
"""The get-up recipe's searched motions for SKU1 v2 (MP4R_ROBOT=sku1v2), from SKU2's.

    python -B tools/sku1v2_motions.py [--threshold 0.75] [--gens 25] [--robots 64] [--only back_a,...]

The recipe's clone plays six open-loop get-ups (``data/searched_motions``, found on SKU2) in the get-up environment
with the hip and knee track ends lifted 60 deg (the recipe's ``wide`` options) and keeps the trials that stand up. On SKU1
v2 (a shorter, lighter body; the same legs) each motion is first replayed as it is (``getup_search --replay``, ``--robots``
randomised robots, the clone's environment). One that stands up often enough (``--threshold``, the best sequence's
success) is kept unchanged; any other is searched again on SKU1 v2 from itself (``getup_search --init``: the cross-entropy
method, its own start, band, postures, trial length, shaping and current cost, ``--gens`` generations), replayed, and
the better of the two kept. The results go to ``data/searched_motions_sku1v2`` under the same names, which the recipe reads;
``runs/motions/summary.json`` has the table. A finished motion is skipped when the script runs again.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "searched_motions"
DST = REPO / "data" / "searched_motions_sku1v2"
RUNS = REPO / "runs" / "motions"
# the recipe's sources (scripts/recipes/getup.sh): name -> start
MOTIONS = {"search_left": "left", "search_nose": "nose", "search_right_a": "right", "search_right_b": "right",
           "search_back_a": "back", "search_back_b": "back"}
WIDEN = "60"


def run(args: list[str], log: Path) -> None:
    cmd = [sys.executable, "-B", "-m", "mp4_recovery.gpu.getup_search", *args]
    print("  " + " ".join(cmd[3:]), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
    if r.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise SystemExit(f"getup_search failed (log {log}):\n" + "\n".join(tail))


def replay(src: Path, start: str, robots: int, seed: int, tag: str) -> float:
    out = RUNS / f"replay_{tag}.json"
    if not out.exists():
        run(["--limits", "--widen-limits-deg", WIDEN, "--replay", str(src), "--start", start, "--robots", str(robots),
             "--seed", str(seed), "--json", str(out)], RUNS / f"replay_{tag}.log")
    return float(json.loads(out.read_text())["best"]["success"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.75, help="replay success that keeps SKU2's motion unchanged")
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--robots", type=int, default=64, help="robots per replay")
    ap.add_argument("--seed", type=int, default=12, help="the replays' seed")
    ap.add_argument("--only", default="", help="comma-separated motion names (default: all six)")
    ap.add_argument("--force-search", action="store_true", help="search every motion again, whatever its replay")
    a = ap.parse_args()
    if os.environ.get("MP4R_ROBOT", "").strip().lower() != "sku1v2":
        raise SystemExit("set MP4R_ROBOT=sku1v2 first")
    RUNS.mkdir(parents=True, exist_ok=True)
    DST.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(MOTIONS)
    summary_path = RUNS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names:
        start = MOTIONS[name]
        if (DST / f"{name}.json").exists() and name in summary:
            print(f"{name}: done ({summary[name]['kept']}, {summary[name]['success']:.2f})", flush=True)
            continue
        src = SRC / f"{name}.json"
        saved = json.loads(src.read_text())
        print(f"{name} ({start}): SKU2's motion on SKU1 v2", flush=True)
        s0 = replay(src, start, a.robots, a.seed, name)
        row = {"start": start, "sku2_success": s0, "searched": False}
        best_path, best_s = src, s0
        if s0 < a.threshold or a.force_search:
            print(f"  {s0:.2f} < {a.threshold:.2f}: searching it again from itself ({a.gens} generations)", flush=True)
            out = RUNS / f"search_{name}.json"
            if not out.exists():
                run(["--limits", "--widen-limits-deg", WIDEN, "--start", start, "--init", str(src),
                     "--band", *[str(v) for v in saved["band_deg"]], "--k", str(saved["k"]),
                     "--trial-s", str(saved.get("trial_s", 6.0)), "--shaping", str(saved.get("shaping", 0.0)),
                     "--current-cost", str(saved.get("current_cost", 0.1)), "--pop", str(saved.get("pop", 64)),
                     "--robots", str(saved.get("robots", 16)), "--gens", str(a.gens),
                     "--seed", str(saved.get("seed", 11)), "--json", str(out)], RUNS / f"search_{name}.log")
            s1 = replay(out, start, a.robots, a.seed, name + "_sku1v2")
            row.update(searched=True, sku1v2_success=s1)
            if s1 > s0:
                best_path, best_s = out, s1
        shutil.copy(best_path, DST / f"{name}.json")
        row.update(kept="SKU1 v2 search" if best_path != src else "SKU2's", success=best_s)
        summary[name] = row
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"  kept {row['kept']}: {best_s:.2f} of the robots stand up", flush=True)
    print("\nmotion            start  SKU2's  searched  kept")
    for name in MOTIONS:
        r = summary.get(name)
        if r:
            sk = f"{r['sku1v2_success']:.2f}" if r.get("searched") else "-"
            print(f"{name:<17} {r['start']:<6} {r['sku2_success']:.2f}    {sk:<8}  {r['kept']}")
    bad = [n for n in names if summary.get(n, {}).get("success", 0.0) == 0.0]
    for n in bad:
        print(f"WARNING: {n}: no robot stands up with it; the clone has no data from {MOTIONS[n]} from it")
    if any(all(summary.get(n, {}).get("success", 0.0) == 0.0 for n in MOTIONS if MOTIONS[n] == st)
           for st in set(MOTIONS[n] for n in names)):
        print("WARNING: a start has no motion that stands up: the clone learns nothing from it (PPO's `wide` step may still)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
