#!/usr/bin/env python
"""The SKU1 v2 run's results in one table (train.ps1's last step): the motions, the walker's shoves, and per seed the
get-up from every start and the walk / shove / fall / get-up / walk sequence. Reads runs/motions and runs/eval.

    python -B tools/sku1v2_report.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "runs" / "eval"


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def f(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def main() -> int:
    s = load(REPO / "runs" / "motions" / "summary.json") or {}
    if s:
        print("get-up motions on SKU1 v2 (share of robots standing up)")
        for name, r in s.items():
            print(f"  {name:<16} {r['start']:<6} SKU2's {r['sku2_success']:.2f}"
                  f"{'  searched ' + f(r.get('sku1v2_success')) if r.get('searched') else '':<16}  kept {r['kept']}")
    rows = []
    for w, label in (("sku1v2_walker_stance0", "the 405 g walk, own stance"), ("sku1v2_walker", "stance 12 deg wider")):
        d = load(EV / f"push_{w}.json")
        if d:
            rows.append((label, {k: v.get("fall_share") for k, v in d.get("commands", {}).items()}))
    if rows:
        cmds = list(rows[0][1])
        print("\nthe walker shoved (0.5-1.5 m/s every 2-4 s, 20 s): share of robots that fell")
        print("  " + " " * 26 + "".join(f"{c:>8}" for c in cmds))
        for label, r in rows:
            print(f"  {label:<26}" + "".join(f"{f(r.get(c)):>8}" for c in cmds))
    for g in sorted(EV.glob("getup_s*_s12.json")):
        seed = g.stem.split("_")[1]
        d = load(g)
        if not d:
            continue
        print(f"\nget-up seed {seed[1:]}: success / median seconds to a held stand, 256 robots per start")
        for st, r in d.get("starts", {}).items():
            sm = r.get("summary", {})
            print(f"  {st:<8} {f(sm.get('success'))}  {f(sm.get('t_success_p50'), '{:.1f} s')}")
        t = load(EV / f"thresholds_{seed}.json")
        if t is not None:
            print(f"  thresholds (the recovery repo's): {t.get('verdict')}")
        q = load(EV / f"sequence_{seed}_s12.json")
        sm = (q or {}).get("summary", {})
        if sm:
            o = sm.get("overall", {})
            print(f"  sequence (walk, shoves 0.5-1.5 m/s, 60 s, 256 robots): topple rate {f(o.get('topple_rate'))}, "
                  f"walking again within 8 s {f(o.get('within_8s'))}, robots down over 15 s {sm.get('robots_down_over_15s')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
