#!/usr/bin/env python
"""How well a parameter set predicts a unit it was not fitted on.

``transfer`` compares a unit with one or more reference units (typically the
one the set was fitted on) over the cells both recorded -- same payload and
arm, trajectory and kp -- and reports, per kp and trajectory class (powered,
small-signal, torque-off drops):

- the paired MAE difference unit - reference, with a 95 % interval from a
  bootstrap over trajectories (a block repeats each trajectory at several
  gains, so its cells are not independent), per trajectory, per block, and on
  the cells whose start temperatures are within ``--max-dt`` of each other;
- the log-vs-log difference between the units on the same commands, raw and
  with each pair's mean offset removed;
- each dataset's floor: its ``extra/repeat`` (and ``extra/small_repeat``) logs
  against their twins in ``processed/``, log vs log on the samples where both
  have torque on, with the model's MAE on the same samples;
- a torque-constant scan (``kt`` x s) on the powered cells and a friction scan
  (every friction term x f, at the best s) on the drops;
- the out-of-sample rescale test: s fitted on the unit's powered cells at
  ``--fit-kp``, and f on its drops there, applied to its other gains.

``q_offset`` is refitted on a grid, once per dataset, arm length and
comparison, over the cells that comparison uses (drops included), and then
held: refitted per scale, it trades against the scale.

``stream`` scores recorded target streams (``gait_hip``, ``gait_knee``) after
their lead-in: the model's MAE, a baseline that is the goal itself delayed by
the best of 0-60 ms (``--max-delay``), the skill 1 - model / baseline (and the same skill on the
dataset's standard powered logs of that payload and kp), log-vs-log floors
between logs of the same stream and payload, peak speed, overshoot, current,
and dwells (the output still for at least 40 ms with more than 0.1 rad of
error), measured against the model. The streams hold +0.5 rad, so dq < 0
moves toward the hanging vertical. ``--variant`` re-scores with a modified
model, e.g. ``kd_ratio*=0``, ``friction*=0.5`` (every friction term) or
``kd_form=filtered,KD_TAU=0.01`` (actuator attributes).

``holds`` measures oscillation while the goal holds still (the levels of
``steps`` by default): the rms of the output in a frequency band (3.5-8 Hz)
over each hold after it has settled, measured against the model's rollout of
the same log. A stick-slip limit cycle shows here; a smooth settle does not.

All three drop the first ``--trim`` samples of every log (3 by default):
``bam.process`` extrapolates each log back to t = 0, and on some logs the first
processed sample carries tens of rad/s.

    .venv/bin/python bam/mangdang/tools/unit_transfer.py transfer --params bam/params/md01/id_3/m6.json \\
        --unit datasets/md01/id_3/unit_C --ref datasets/md01/id_3/unit_A --ref datasets/md01/id_3/unit_B
    .venv/bin/python bam/mangdang/tools/unit_transfer.py stream \\
        --params bam/params/md01/id_1/m6.json bam/params/md01/id_2/m6.json bam/params/md01/id_3/m6.json \\
        --dataset datasets/md01/id_2/unit_B --dataset datasets/md01/id_2/unit_C
"""

import argparse
import glob
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DROPS = {"lift_and_drop", "abd_drop"}
SMALL = {"abd_steps", "abd_sts", "abd_sweep"}
STREAMS = {"gait_hip", "gait_knee"}
FRICTION = (
    "friction_base",
    "friction_stribeck",
    "load_friction_motor",
    "load_friction_external",
    "load_friction_motor_stribeck",
    "load_friction_external_stribeck",
    "load_friction_motor_quad",
    "load_friction_external_quad",
    "friction_viscous",
)
CLASSES = ("powered", "small", "drops")


def traj_class(trajectory):
    if trajectory in DROPS:
        return "drops"
    if trajectory in SMALL:
        return "small"
    if trajectory in STREAMS:
        return "stream"
    return "powered"


def block_of(log):
    """Payload and arm, with the payload to the nearest 5 g (units weigh the same payload set 0.1-0.5 g apart)."""
    return f"{int(round(log['mass'] * 1000 / 5) * 5)}g@{log['length']:.2f}m"


# -- rollouts (run in worker processes) ----------------------------------------


@lru_cache(maxsize=None)
def _log(path, trim):
    log = json.load(open(path))
    log["entries"] = log["entries"][trim:]
    return log


def parse_variant(spec):
    """``a*=0.5,b=1,kd_form=filtered`` -> ((key, op, value), ...)."""
    mods = []
    for part in filter(None, spec.split(",")):
        op = "*=" if "*=" in part else "="
        key, value = part.split(op)
        try:
            value = float(value)
        except ValueError:
            pass
        mods.append((key.strip(), op, value))
    return tuple(mods)


def build_model(params, mods=(), q=None, scale=1.0, friction=1.0):
    from bam.model import load_model

    model = load_model(params)
    if q is not None:
        model.q_offset.value = q
    model.kt.value *= scale
    for name in FRICTION:
        if hasattr(model, name):
            getattr(model, name).value *= friction
    for key, op, value in mods:
        names = [n for n in FRICTION if hasattr(model, n)] if key == "friction" else [key]
        for name in names:
            if hasattr(model, name):
                p = getattr(model, name)
                p.value = p.value * value if op == "*=" else value
            else:  # an actuator attribute (kd_form, KD_TAU, KD_GAIN)
                setattr(model.actuator, name, value)
    return model


def _rollout(params, mods, q, scale, friction, path, trim):
    from bam import simulate

    log = _log(path, trim)
    model = build_model(params, mods, q, scale, friction)
    pos, vel, ctrl = simulate.Simulator(model).rollout_log(log, simulate_control=True)
    n = min(len(pos), len(log["entries"]))
    ctrl = np.array([c if c is not None else 0.0 for c in ctrl[:n]], dtype=float)
    pos, vel = np.asarray(pos[:n], dtype=float), np.asarray(vel[:n], dtype=float)
    current = np.asarray(model.actuator.compute_current(ctrl, vel), dtype=float)
    return pos, vel, current


def mae_job(job):
    """Position MAE [mrad] over all samples, as ``eval_fits.py`` computes it."""
    params, mods, q, scale, friction, path, trim = job
    pos, _, _ = _rollout(params, mods, q, scale, friction, path, trim)
    ref = np.array([e["position"] for e in _log(path, trim)["entries"]])[: len(pos)]
    return float(np.mean(np.abs(pos - ref))) * 1000.0


def rollout_job(job):
    return _rollout(*job)


# -- datasets -----------------------------------------------------------------


class Dataset:
    def __init__(self, root, kps, kd):
        self.root = root.rstrip("/")
        self.name = "/".join(Path(self.root).parts[-2:])
        self.cells = {}  # (length, block, trajectory, kp) -> path
        self.meta = {}
        for path in sorted(glob.glob(f"{self.root}/processed/*.json")):
            log = json.load(open(path))
            if float(log.get("kd", 400.0)) != kd or int(log["kp"]) not in kps:
                continue
            key = (round(log["length"], 2), block_of(log), log["trajectory"], int(log["kp"]))
            if key in self.cells:
                print(f"   {self.name}: {Path(path).name} repeats cell {key}; kept {Path(self.cells[key]).name}")
                continue
            self.cells[key] = path
            self.meta[path] = {"temp": log.get("temp_live_start_c"), "trajectory": log["trajectory"]}

    def lengths(self):
        return sorted({k[0] for k in self.cells})

    def extras(self, *names):
        out = []
        for name in names:
            out += sorted(glob.glob(f"{self.root}/extra/{name}/*.json"))
        return out


def entries_array(path, trim, field):
    return np.array([e[field] for e in _log(path, trim)["entries"]], dtype=float)


# -- statistics -----------------------------------------------------------------


def boot_ci(diffs, n_boot, rng):
    """95 % interval of the mean of ``{key: value}`` from a bootstrap over the keys' trajectories (key[2])."""
    trajs = sorted({k[2] for k in diffs})
    by = {t: [v for k, v in diffs.items() if k[2] == t] for t in trajs}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(trajs), len(trajs))
        values = [v for i in pick for v in by[trajs[i]]]
        means.append(np.mean(values))
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarise(diffs, a, b, n_boot, rng, temps=None, max_dt=5.0):
    v = list(diffs.values())
    out = {
        "n": len(v),
        "delta": float(np.mean(v)),
        "median": float(np.median(v)),
        "ci95": boot_ci(diffs, n_boot, rng) if len({k[2] for k in diffs}) > 1 else None,  # one trajectory: no interval
        "unit": float(np.mean([a[k] for k in diffs])),
        "ref": float(np.mean([b[k] for k in diffs])),
        "per_trajectory": {t: float(np.mean([d for k, d in diffs.items() if k[2] == t])) for t in sorted({k[2] for k in diffs})},
        "per_block": {bl: float(np.mean([d for k, d in diffs.items() if k[1] == bl])) for bl in sorted({k[1] for k in diffs})},
    }
    if temps:
        matched = {k: d for k, d in diffs.items() if temps[k] is not None and abs(temps[k]) <= max_dt}
        out["temp_diff_c"] = {"min": float(min(t for t in temps.values() if t is not None)),
                              "max": float(max(t for t in temps.values() if t is not None))} if any(t is not None for t in temps.values()) else None
        out["matched_temp"] = {"n": len(matched), "delta": float(np.mean(list(matched.values())))} if matched else {"n": 0}
    return out


def fmt(s):
    ci = f"95% CI {s['ci95'][0]:+5.1f}..{s['ci95'][1]:+5.1f}" if s["ci95"] else "one trajectory, no CI"
    return (f"n={s['n']:2d} delta {s['delta']:+5.1f} (median {s['median']:+5.1f}, {ci}); "
            f"unit {s['unit']:5.1f} vs ref {s['ref']:5.1f}")


def grid(spec):
    lo, hi, step = (float(x) for x in spec.split(":"))
    return [round(x, 4) for x in np.arange(lo, hi + step / 2, step)]


def arange_shift(goal, d):
    return np.concatenate([np.full(d, goal[0]), goal[: len(goal) - d]]) if d else goal


def dwells(position, speed, goal, mask, dt, gap=0.1, still=0.15, min_s=0.04):
    """Runs spanning at least ``min_s`` (first to last sample) where the output is still (|speed| < ``still``)
    with more than ``gap`` rad of error."""
    hit = mask & (np.abs(speed) < still) & (np.abs(goal - position) > gap)
    runs, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j < len(hit) and hit[j]:
                j += 1
            if (j - 1 - i) * dt >= min_s - 1e-9:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def delayed_goal_mae(goal, measured, mask, max_samples):
    maes = [float(np.mean(np.abs(arange_shift(goal, d)[mask] - measured[mask]))) * 1000.0 for d in range(max_samples + 1)]
    best = int(np.argmin(maes))
    return maes[best], best, maes[0]


# -- transfer -------------------------------------------------------------------


def run_transfer(args, pool):
    rng = np.random.default_rng(0)
    kps = [int(k) for k in args.kps.split(",")]
    unit = Dataset(args.unit, kps, args.kd)
    refs = [Dataset(r, kps, args.kd) for r in args.ref]
    qs, scales, frictions = grid(args.q_grid), grid(args.scales), grid(args.frictions)
    P, T = args.params, args.trim
    print(f"params {P}; unit {unit.name} ({len(unit.cells)} cells); trim {T} samples; q grid {args.q_grid}")

    def maes(jobs):
        return list(pool.map(mae_job, jobs, chunksize=4))

    def cell_maes(ds, keys, q, scale=1.0, friction=1.0):
        return dict(zip(keys, maes([(P, (), q, scale, friction, ds.cells[k], T) for k in keys])))

    def grid_q(ds, keys):
        if args.q_offset is not None:
            return args.q_offset
        values = maes([(P, (), q, 1.0, 1.0, ds.cells[k], T) for q in qs for k in keys])
        costs = np.array(values).reshape(len(qs), len(keys)).mean(axis=1)
        best = qs[int(np.argmin(costs))]
        if abs(best) >= 0.1 - 1e-9:
            print(f"   WARNING q_offset at the grid bound ({best:+.3f}) on {ds.name}")
        return best

    def log_vs_log(pa, pb):
        """Mean |a - b| [mrad] on the samples where both have torque on, raw and offset-removed."""
        a, b = entries_array(pa, T, "position"), entries_array(pb, T, "position")
        ea, eb = entries_array(pa, T, "torque_enable"), entries_array(pb, T, "torque_enable")
        n = min(len(a), len(b))
        m = (ea[:n] > 0) & (eb[:n] > 0)
        d = a[:n][m] - b[:n][m]
        return float(np.mean(np.abs(d))) * 1000.0, float(np.mean(np.abs(d - d.mean()))) * 1000.0, m

    result = {"params": P, "unit": unit.name, "trim": T, "comparisons": [], "floors": {}, "scans": {}}
    q_of = {}  # (dataset, length) -> q used for its floor

    for ref in refs:
        pooled = {kp: {c: ({}, {}, {}, {}) for c in CLASSES} for kp in kps}
        arms = []  # (shared cells, unit q, reference MAEs) per arm length
        for length in unit.lengths():
            shared = sorted(k for k in unit.cells if k in ref.cells and k[0] == length)
            if not shared:
                continue
            qu, qr = grid_q(unit, shared), grid_q(ref, shared)
            q_of.setdefault((unit.name, length), qu)
            q_of.setdefault((ref.name, length), qr)
            mu, mr = cell_maes(unit, shared, qu), cell_maes(ref, shared, qr)
            arms.append((shared, qu, mr))
            temps = {}
            for k in shared:
                tu, tr = unit.meta[unit.cells[k]]["temp"], ref.meta[ref.cells[k]]["temp"]
                temps[k] = None if tu is None or tr is None else tu - tr
            comp = {"ref": ref.name, "length": length, "q_unit": qu, "q_ref": qr, "cells": len(shared), "rows": {},
                    "cells_detail": {"|".join(map(str, k)): {"unit": mu[k], "ref": mr[k], "dT": temps[k]} for k in shared}}
            print(f"-- vs {ref.name}, {length:.2f} m arm: q unit {qu:+.3f}, q ref {qr:+.3f}, {len(shared)} shared cells")
            for kp in kps:
                for cls in CLASSES:
                    diffs = {k: mu[k] - mr[k] for k in shared if k[3] == kp and traj_class(k[2]) == cls}
                    if not diffs:
                        continue
                    s = summarise(diffs, mu, mr, args.boot, rng, {k: temps[k] for k in diffs}, args.max_dt)
                    comp["rows"][f"{kp}/{cls}"] = s
                    mt = s["matched_temp"]
                    print(f"   kp {kp:3d} {cls:7s}: {fmt(s)}"
                          + (f"; |dT|<={args.max_dt:g} C: n={mt['n']} {mt['delta']:+5.1f}" if mt.get("n") and mt["n"] < s["n"] else ""))
                    print("      per trajectory " + ", ".join(f"{t} {d:+.1f}" for t, d in s["per_trajectory"].items()))
                    if len(s["per_block"]) > 1:
                        print("      per block " + ", ".join(f"{b} {d:+.1f}" for b, d in s["per_block"].items()))
                    for d, src in zip(pooled[kp][cls], (diffs, mu, mr, temps)):
                        d.update({k: src[k] for k in diffs})
            # unit vs reference, log vs log on identical commands (powered)
            lvl = {k: log_vs_log(unit.cells[k], ref.cells[k])[:2] for k in shared if traj_class(k[2]) == "powered"}
            if lvl:
                comp["log_vs_log"] = {"|".join(map(str, k)): v for k, v in lvl.items()}
                per_t = {t: (np.mean([v[0] for k, v in lvl.items() if k[2] == t]), np.mean([v[1] for k, v in lvl.items() if k[2] == t]))
                         for t in sorted({k[2] for k in lvl})}
                print("   log vs log (raw / offset removed): " + ", ".join(f"{t} {a:.1f}/{b:.1f}" for t, (a, b) in per_t.items()))
            # out-of-sample rescale: s from the unit's powered cells at fit_kp, f from its drops there
            fit_pow = [k for k in shared if k[3] == args.fit_kp and traj_class(k[2]) == "powered"]
            fit_drop = [k for k in shared if k[3] == args.fit_kp and traj_class(k[2]) == "drops"]
            if fit_pow:
                table = np.array(maes([(P, (), qu, s, 1.0, unit.cells[k], T) for s in scales for k in fit_pow])).reshape(len(scales), -1)
                s_fit = scales[int(np.argmin(table.mean(axis=1)))]
                f_fit = 1.0
                if fit_drop:
                    ft = np.array(maes([(P, (), qu, s_fit, f, unit.cells[k], T) for f in frictions for k in fit_drop])).reshape(len(frictions), -1)
                    f_fit = frictions[int(np.argmin(ft.mean(axis=1)))]
                comp["rescale"] = {"s_fit": s_fit, "f_fit": f_fit, "fit_kp": args.fit_kp, "rows": {}}
                for kp in kps:
                    if kp == args.fit_kp:
                        continue
                    for cls in ("powered", "small"):
                        keys = [k for k in shared if k[3] == kp and traj_class(k[2]) == cls]
                        if not keys:
                            continue
                        rs = cell_maes(unit, keys, qu, s_fit, 1.0)
                        rsf = cell_maes(unit, keys, qu, s_fit, f_fit)
                        a = summarise({k: rs[k] - mr[k] for k in keys}, rs, mr, args.boot, rng)
                        b = summarise({k: rsf[k] - mr[k] for k in keys}, rsf, mr, args.boot, rng)
                        comp["rescale"]["rows"][f"{kp}/{cls}"] = {"s": a, "s_f": b}
                        print(f"   rescale (s x{s_fit:.3f} at kp {args.fit_kp}) kp {kp} {cls:7s}: {fmt(a)}")
                        print(f"   rescale (s x{s_fit:.3f}, f x{f_fit:.2f})    kp {kp} {cls:7s}: {fmt(b)}")
            result["comparisons"].append(comp)
        if len([c for c in result["comparisons"] if c["ref"] == ref.name]) > 1:
            print(f"-- vs {ref.name}, all arms pooled")
            rows = {}
            for kp in kps:
                for cls in CLASSES:
                    diffs, mu, mr, temps = pooled[kp][cls]
                    if diffs:
                        s = summarise(diffs, mu, mr, args.boot, rng, temps, args.max_dt)
                        rows[f"{kp}/{cls}"] = s
                        print(f"   kp {kp:3d} {cls:7s}: {fmt(s)}")
            pooled_comp = {"ref": ref.name, "length": "all", "rows": rows}
            # one s for the unit, fitted on its powered cells at fit_kp on every arm, applied to its other gains
            fit = [(k, q) for shared, q, _ in arms for k in shared if k[3] == args.fit_kp and traj_class(k[2]) == "powered"]
            drops = [(k, q) for shared, q, _ in arms for k in shared if k[3] == args.fit_kp and traj_class(k[2]) == "drops"]
            if fit:
                table = np.array(maes([(P, (), q, s_, 1.0, unit.cells[k], T) for s_ in scales for k, q in fit])).reshape(len(scales), -1)
                s_fit = scales[int(np.argmin(table.mean(axis=1)))]
                f_fit = 1.0
                if drops:
                    ft = np.array(maes([(P, (), q, s_fit, f, unit.cells[k], T) for f in frictions for k, q in drops])).reshape(len(frictions), -1)
                    f_fit = frictions[int(np.argmin(ft.mean(axis=1)))]
                pooled_comp["rescale"] = {"s_fit": s_fit, "f_fit": f_fit, "fit_kp": args.fit_kp, "rows": {}}
                for kp in kps:
                    if kp == args.fit_kp:
                        continue
                    for cls in ("powered", "small"):
                        cells = [(k, q, mr) for shared, q, mr in arms for k in shared if k[3] == kp and traj_class(k[2]) == cls]
                        if not cells:
                            continue
                        rs = dict(zip([k for k, _, _ in cells], maes([(P, (), q, s_fit, 1.0, unit.cells[k], T) for k, q, _ in cells])))
                        rsf = dict(zip([k for k, _, _ in cells], maes([(P, (), q, s_fit, f_fit, unit.cells[k], T) for k, q, _ in cells])))
                        ref_mae = {k: mr[k] for k, _, mr in cells}
                        a = summarise({k: rs[k] - ref_mae[k] for k in rs}, rs, ref_mae, args.boot, rng)
                        b = summarise({k: rsf[k] - ref_mae[k] for k in rsf}, rsf, ref_mae, args.boot, rng)
                        pooled_comp["rescale"]["rows"][f"{kp}/{cls}"] = {"s": a, "s_f": b}
                        print(f"   rescale, one s on all arms (x{s_fit:.3f} at kp {args.fit_kp}) kp {kp} {cls:7s}: {fmt(a)}")
                        print(f"   rescale, one s on all arms (x{s_fit:.3f}, f x{f_fit:.2f})  kp {kp} {cls:7s}: {fmt(b)}")
            result["comparisons"].append(pooled_comp)

    # floors: each dataset's repeat logs against their matrix twins
    for ds in [unit] + refs:
        rows = []
        for path in ds.extras("repeat", "small_repeat"):
            log = _log(path, T)
            key = (round(log["length"], 2), block_of(log), log["trajectory"], int(log["kp"]))
            twin = ds.cells.get(key)
            q = q_of.get((ds.name, key[0]))
            if twin is None or q is None:
                continue
            raw, _, mask = log_vs_log(path, twin)
            pos, _, _ = rollout_job((P, (), q, 1.0, 1.0, path, T))
            meas = entries_array(path, T, "position")
            n = min(len(pos), len(mask))
            model = float(np.mean(np.abs(pos[:n] - meas[:n])[mask[:n]])) * 1000.0
            rows.append({"file": Path(path).name, "trajectory": key[2], "block": key[1], "floor": raw, "model": model})
        if rows:
            result["floors"][ds.name] = rows
            print(f"-- floor {ds.name} (repeat vs twin, common torque-on samples): "
                  + ", ".join(f"{r['trajectory']} {r['floor']:.1f} / model {r['model']:.1f}" for r in rows)
                  + f"; mean ratio {np.mean([r['model'] for r in rows]) / np.mean([r['floor'] for r in rows]):.1f}x")

    # scans on each dataset, over the cells it shares with the unit, q held
    for ds in [unit] + refs:
        for length in unit.lengths():
            keys = sorted(k for k in ds.cells if k[0] == length and k in unit.cells)
            pk = [k for k in keys if traj_class(k[2]) == "powered"]
            dk = [k for k in keys if traj_class(k[2]) == "drops"]
            if not pk:
                continue
            q = grid_q(ds, keys)
            table = np.array(maes([(P, (), q, s, 1.0, ds.cells[k], T) for s in scales for k in pk])).reshape(len(scales), len(pk))
            curve = table.mean(axis=1)
            best = scales[int(np.argmin(curve))]
            within = [s for s, c in zip(scales, curve) if c <= curve.min() + 0.5]
            trajs = sorted({k[2] for k in pk})
            idx = {t: [i for i, k in enumerate(pk) if k[2] == t] for t in trajs}
            boots = []
            for _ in range(args.boot):
                cols = [i for j in rng.choice(len(trajs), len(trajs)) for i in idx[trajs[j]]]
                boots.append(scales[int(np.argmin(table[:, cols].mean(axis=1)))])
            scan = {"q": q, "cells": len(pk), "s_best": best, "s_5_95": [float(x) for x in np.percentile(boots, [5, 95])],
                    "s_within_0.5": [min(within), max(within)], "curve": dict(zip(map(str, scales), map(float, curve))),
                    "mae_at_1": float(curve[scales.index(1.0)]) if 1.0 in scales else None}
            sk = [k for k in keys if traj_class(k[2]) == "small"]
            if sk:
                st = np.array(maes([(P, (), q, s, 1.0, ds.cells[k], T) for s in scales for k in sk])).reshape(len(scales), len(sk))
                sc = st.mean(axis=1)
                sw = [s for s, c in zip(scales, sc) if c <= sc.min() + 0.5]
                scan.update({"s_best_small": scales[int(np.argmin(sc))], "s_within_0.5_small": [min(sw), max(sw)], "small_cells": len(sk)})
            if dk:
                ft = np.array(maes([(P, (), q, best, f, ds.cells[k], T) for f in frictions for k in dk])).reshape(len(frictions), len(dk))
                fc = ft.mean(axis=1)
                fw = [f for f, c in zip(frictions, fc) if c <= fc.min() + 0.5]
                scan.update({"f_best": frictions[int(np.argmin(fc))], "f_within_0.5": [min(fw), max(fw)], "drop_cells": len(dk),
                             "f_curve": dict(zip(map(str, frictions), map(float, fc)))})
            result["scans"][f"{ds.name}@{length:.2f}"] = scan
            print(f"-- scan {ds.name}, {length:.2f} m (q {q:+.3f}, {len(pk)} powered cells): s best x{best:.3f} "
                  f"(5-95 % x{scan['s_5_95'][0]:.3f}-x{scan['s_5_95'][1]:.3f}; +0.5 mrad x{min(within):.3f}-x{max(within):.3f}); "
                  f"MAE {curve.min():.1f} vs {scan['mae_at_1']:.1f} at x1"
                  + (f"; small-signal s best x{scan['s_best_small']:.3f} (+0.5 mrad x{scan['s_within_0.5_small'][0]:.3f}-"
                     f"x{scan['s_within_0.5_small'][1]:.3f})" if sk else "")
                  + (f"; drops f best x{scan['f_best']:.2f} (+0.5 mrad x{scan['f_within_0.5'][0]:.2f}-x{scan['f_within_0.5'][1]:.2f})" if dk else ""))
    return result


# -- stream ---------------------------------------------------------------------


def run_stream(args, pool):
    from bam.mangdang.trajectories import STREAMS as STREAM_DEFS

    T = args.trim
    variants = [("as fitted", ())] + [(v, parse_variant(v)) for v in args.variant]
    qs = grid(args.q_grid)
    result = {"trim": T, "datasets": {}}
    for root in args.dataset:
        root = root.rstrip("/")
        name = "/".join(Path(root).parts[-2:])
        streams = [p for p in sorted(glob.glob(f"{root}/extra/*/*.json")) if json.load(open(p))["trajectory"] in STREAMS]
        if not streams:
            continue
        info = {p: _log(p, T) for p in streams}
        blocks = {}
        for l in info.values():
            blocks.setdefault((round(l["length"], 2), block_of(l), int(l["kp"])), set()).add(l.get("zero_dd"))
        # standard logs of the stream payloads on the same mount (same zero_dd), for q_offset and the skill reference
        std, fallback = [], []
        for p in sorted(glob.glob(f"{root}/processed/*.json") + glob.glob(f"{root}/extra/*/*.json")):
            l = json.load(open(p))
            key = (round(l["length"], 2), block_of(l), int(l["kp"]))
            if l["trajectory"] in STREAMS or key not in blocks or float(l.get("kd", 400)) != args.kd:
                continue
            (std if l.get("zero_dd") in blocks[key] else fallback).append(p)
        if not std:
            std = [p for p in fallback if "/processed/" in p]
            print(f"   {name}: no standard logs on the streams' mount; q_offset and the skill reference come from processed/")
        std_pow = [p for p in std if traj_class(json.load(open(p))["trajectory"]) == "powered"]
        out = {"streams": {}, "floors": [], "standard": {}, "variants": {}}
        # floors: pairs of logs of the same stream and block
        groups = {}
        for p, l in info.items():
            groups.setdefault((block_of(l), l["trajectory"]), []).append(p)
        for (block, traj), ps in sorted(groups.items()):
            for a, b in itertools.combinations(ps, 2):
                qa, qb = entries_array(a, T, "position"), entries_array(b, T, "position")
                ta = entries_array(a, T, "timestamp")
                n = min(len(qa), len(qb))
                w = ta[:n] >= STREAM_DEFS[traj].lead_in
                out["floors"].append({"block": block, "trajectory": traj, "a": Path(a).name, "b": Path(b).name,
                                      "floor": float(np.mean(np.abs(qa[:n][w] - qb[:n][w]))) * 1000.0})
        print(f"== {name}: {len(streams)} stream logs; floors " + ", ".join(
            f"{f['trajectory']}@{f['block']} {f['floor']:.1f}" for f in out["floors"]))
        for params in args.params:
            label = f"{Path(params).parent.name}/{Path(params).stem}"
            # q: grid over the dataset's standard logs of the stream payloads, as fitted
            if args.q_offset is not None:
                q = args.q_offset
            else:
                vals = list(pool.map(mae_job, [(params, (), q, 1.0, 1.0, p, T) for q in qs for p in std], chunksize=4))
                q = qs[int(np.argmin(np.array(vals).reshape(len(qs), len(std)).mean(axis=1)))]
            for vname, mods in variants:
                jobs = [(params, mods, q, 1.0, 1.0, p, T) for p in streams + std_pow]
                rolls = dict(zip(streams + std_pow, pool.map(rollout_job, jobs, chunksize=2)))
                rows = {}
                for p in streams:
                    l = info[p]
                    pos, vel, cur = rolls[p]
                    n = len(pos)
                    t = entries_array(p, T, "timestamp")[:n]
                    meas = entries_array(p, T, "position")[:n]
                    goal = entries_array(p, T, "goal_position")[:n]
                    amps = np.abs(entries_array(p, T, "load")[:n]) / 1000.0
                    w = t >= STREAM_DEFS[l["trajectory"]].lead_in
                    dt = float(l["dt"])
                    base, delay, tracking = delayed_goal_mae(goal, meas, w, int(round(args.max_delay / dt)))
                    err = np.abs(pos - meas) * 1000.0
                    dq = np.gradient(meas, dt)
                    split = {"dq<0": w & (dq < -0.3), "dq>0": w & (dq > 0.3), "still": w & (np.abs(dq) <= 0.3)}
                    dm, dmod = dwells(meas, dq, goal, w, dt), dwells(pos, vel, goal, w, dt)
                    rows[Path(p).name] = {
                        "trajectory": l["trajectory"], "block": block_of(l), "kp": int(l["kp"]),
                        "model": float(err[w].mean()), "signed": float(((pos - meas) * 1000.0)[w].mean()),
                        "delayed_goal": base, "best_delay_ms": delay * dt * 1000.0, "delay_at_edge": delay == int(round(args.max_delay / dt)),
                        "tracking": tracking, "skill": 1.0 - float(err[w].mean()) / base if base > 0 else float("nan"),
                        "split": {k: float(err[m].mean()) for k, m in split.items() if m.any()},
                        "dwells": {"measured": len(dm), "model": len(dmod),
                                   "measured_ms": float(np.mean([(j - 1 - i) * dt * 1000 for i, j in dm])) if dm else None,
                                   "measured_current_mA": float(np.mean([amps[i:j].mean() for i, j in dm]) * 1000) if dm else None},
                        "peak_speed": {"measured": float(np.abs(dq[w]).max()), "model": float(np.abs(vel[w]).max()),
                                       "measured_p99": float(np.percentile(np.abs(dq[w]), 99)),
                                       "model_p99": float(np.percentile(np.abs(vel[w]), 99))},
                        "below_goal": {"measured": float((meas - goal)[w].min()), "model": float((pos - goal)[w].min())},
                        "above_goal": {"measured": float((meas - goal)[w].max()), "model": float((pos - goal)[w].max())},
                        "current_mA": {"measured": float(amps[w].mean() * 1000), "model": float(np.abs(cur[w]).mean() * 1000),
                                       "measured_p95": float(np.percentile(amps[w], 95) * 1000),
                                       "model_p95": float(np.percentile(np.abs(cur[w]), 95) * 1000)},
                    }
                skills = {}
                for p in std_pow:
                    pos, _, _ = rolls[p]
                    n = len(pos)
                    meas = entries_array(p, T, "position")[:n]
                    goal = entries_array(p, T, "goal_position")[:n]
                    m = np.ones(n, dtype=bool)
                    base, _, _ = delayed_goal_mae(goal, meas, m, int(round(args.max_delay / float(_log(p, T)["dt"]))))
                    l = _log(p, T)
                    skills.setdefault(block_of(l), []).append((l["trajectory"], float(np.mean(np.abs(pos - meas))) * 1000.0, base))
                std_summary = {b: {"mae": float(np.mean([v[1] for v in vs])), "skill": float(np.mean([1 - v[1] / v[2] for v in vs])),
                                   "per_trajectory": {v[0]: {"mae": v[1], "skill": 1 - v[1] / v[2]} for v in vs}} for b, vs in skills.items()}
                if vname == "as fitted":
                    out["streams"][label] = {"q": q, "logs": rows}
                    out["standard"][label] = std_summary
                    print(f"-- {label} (q {q:+.3f}); standard powered logs of the stream payloads: "
                          + ", ".join(f"{b} MAE {s['mae']:.1f} skill {s['skill']:+.2f}" for b, s in std_summary.items()))
                    for fname, r in rows.items():
                        floors = [f["floor"] for f in out["floors"] if f["block"] == r["block"] and f["trajectory"] == r["trajectory"]]
                        g1 = r["skill"] >= std_summary[r["block"]]["skill"] - 0.1 if r["block"] in std_summary else None
                        g2 = r["model"] <= 3 * float(np.median(floors)) if floors else None
                        r.update({"G1": g1, "G2": g2, "floor_median": float(np.median(floors)) if floors else None})
                        print(f"   {r['trajectory']:9s} {r['block']:11s} model {r['model']:5.1f} (signed {r['signed']:+5.1f}) "
                              f"delayed goal {r['delayed_goal']:5.1f} @ {r['best_delay_ms']:3.0f} ms{'!' if r['delay_at_edge'] else ''} "
                              f"skill {r['skill']:+.2f} G1 {'-' if g1 is None else 'pass' if g1 else 'FAIL'} "
                              f"G2 {'-' if g2 is None else 'pass' if g2 else 'FAIL'} | measured/model: speed p99 "
                              f"{r['peak_speed']['measured_p99']:3.1f}/{r['peak_speed']['model_p99']:3.1f} rad/s, below goal "
                              f"{r['below_goal']['measured']:+.3f}/{r['below_goal']['model']:+.3f}, above "
                              f"{r['above_goal']['measured']:+.3f}/{r['above_goal']['model']:+.3f} rad, "
                              f"|I| {r['current_mA']['measured']:3.0f}/{r['current_mA']['model']:3.0f} mA, dwells "
                              f"{r['dwells']['measured']}/{r['dwells']['model']}"
                              + (f" ({r['dwells']['measured_ms']:.0f} ms at {r['dwells']['measured_current_mA']:.0f} mA)" if r['dwells']['measured'] else ""))
                else:
                    out["variants"].setdefault(vname, {})[label] = {
                        "stream": {f: r["model"] for f, r in rows.items()},
                        "stream_mean": float(np.mean([r["model"] for r in rows.values()])),
                        "standard_mean": float(np.mean([s["mae"] for s in std_summary.values()])),
                    }
            if len(variants) > 1:
                ref_stream = float(np.mean([r["model"] for r in out["streams"][label]["logs"].values()]))
                ref_std = float(np.mean([s["mae"] for s in out["standard"][label].values()]))
                print(f"   variants ({label}; stream mean / standard mean, as fitted {ref_stream:.1f} / {ref_std:.1f}): "
                      + ", ".join(f"{v} {out['variants'][v][label]['stream_mean']:.1f}/{out['variants'][v][label]['standard_mean']:.1f}"
                                  for v, _ in variants[1:]))
        result["datasets"][name] = out
    return result


# -- holds ----------------------------------------------------------------------


def band_rms(t, x, lo, hi):
    """rms [same unit as x] of x in [lo, hi] Hz after a quadratic detrend, and the band's peak frequency."""
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    dt = float(np.median(np.diff(t)))
    # the peak from a zero-padded spectrum: a hold lasts about 1 s, so the plain FFT has ~1 Hz bins
    padded = np.abs(np.fft.rfft(x, 8192))
    fp = np.fft.rfftfreq(8192, dt)
    band = (fp >= lo) & (fp <= hi)
    peak = float(fp[band][int(np.argmax(padded[band]))]) if band.any() else float("nan")
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), dt)
    outside = (f < lo) | (f > hi)
    spectrum[outside] = 0
    return float(np.std(np.fft.irfft(spectrum, len(x)))), peak


def hold_masks(t, goal, min_hold, settle):
    """One mask per stretch where the goal is constant for at least ``min_hold`` s, from ``settle`` s in."""
    masks, i = [], 0
    while i < len(goal):
        j = i
        while j < len(goal) and abs(goal[j] - goal[i]) < 1e-4:
            j += 1
        if t[j - 1] - t[i] >= min_hold:
            masks.append((t >= t[i] + settle) & (t < t[j - 1]))
        i = j
    return masks


def run_holds(args, pool):
    T = args.trim
    kps = [int(k) for k in args.kps.split(",")]
    lo, hi = (float(x) for x in args.band.split(":"))
    qs = grid(args.q_grid)
    result = {"params": args.params, "band_hz": [lo, hi], "trim": T, "datasets": {}}
    print(f"params {args.params}; {args.trajectory} holds, {lo:g}-{hi:g} Hz rms after {args.settle:g} s; trim {T}")
    for root in args.dataset:
        ds = Dataset(root, kps, args.kd)
        rows = {}
        for length in ds.lengths():
            keys = sorted(k for k in ds.cells if k[0] == length)
            if args.q_offset is not None:
                q = args.q_offset
            else:
                vals = list(pool.map(mae_job, [(args.params, (), q, 1.0, 1.0, ds.cells[k], T) for q in qs for k in keys], chunksize=4))
                q = qs[int(np.argmin(np.array(vals).reshape(len(qs), len(keys)).mean(axis=1)))]
            cells = [k for k in keys if k[2] == args.trajectory]
            rolls = list(pool.map(rollout_job, [(args.params, (), q, 1.0, 1.0, ds.cells[k], T) for k in cells]))
            for k, (pos, _, _) in zip(cells, rolls):
                path = ds.cells[k]
                n = len(pos)
                t = entries_array(path, T, "timestamp")[:n]
                meas = entries_array(path, T, "position")[:n]
                goal = entries_array(path, T, "goal_position")[:n]
                m_rms, s_rms, peaks = [], [], []
                for m in hold_masks(t, goal, args.min_hold, args.settle):
                    a, f = band_rms(t[m], meas[m], lo, hi)
                    b, _ = band_rms(t[m], pos[m], lo, hi)
                    m_rms.append(a * 1000.0)
                    s_rms.append(b * 1000.0)
                    peaks.append(f)
                if m_rms:
                    rows[f"{k[1]}|{k[3]}"] = {"block": k[1], "kp": k[3], "q": q, "holds": len(m_rms), "measured_mrad": float(np.mean(m_rms)),
                                              "model_mrad": float(np.mean(s_rms)), "peak_hz": float(np.median(peaks))}
        result["datasets"][ds.name] = rows
        for r in sorted(rows.values(), key=lambda r: (r["block"], r["kp"])):
            print(f"   {ds.name:12s} {r['block']:11s} kp {r['kp']:3d}: measured {r['measured_mrad']:5.1f} mrad "
                  f"(peak {r['peak_hz']:.1f} Hz), model {r['model_mrad']:4.1f} mrad, {r['holds']} holds")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trim", type=int, default=3, help="processed samples dropped at the start of every log")
    common.add_argument("--kd", type=float, default=400.0, help="only logs at this kd_position")
    common.add_argument("--q-grid", default="-0.10:0.10:0.005", help="q_offset grid lo:hi:step [rad]")
    common.add_argument("--q-offset", type=float, default=None, help="hold q_offset at this value instead of the grid refit")
    common.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    common.add_argument("--json", default=None)
    t = sub.add_parser("transfer", parents=[common])
    t.add_argument("--params", required=True, help="one params JSON")
    t.add_argument("--unit", required=True, help="dataset folder of the unit under test (holds processed/, extra/)")
    t.add_argument("--ref", action="append", required=True, help="reference dataset folder (repeatable)")
    t.add_argument("--kps", default="80,120")
    t.add_argument("--fit-kp", type=int, default=80, help="the gain the rescale test fits s and f at")
    t.add_argument("--scales", default="0.70:1.30:0.025", help="kt scale grid lo:hi:step")
    t.add_argument("--frictions", default="0.5:2.0:0.1", help="friction scale grid lo:hi:step")
    t.add_argument("--boot", type=int, default=2000)
    t.add_argument("--max-dt", type=float, default=5.0, help="temperature match for the matched-cell delta [C]")
    s = sub.add_parser("stream", parents=[common])
    s.add_argument("--params", nargs="+", required=True)
    s.add_argument("--dataset", action="append", required=True, help="dataset folder; its extra/*/ stream logs are scored")
    s.add_argument("--variant", action="append", default=[], help="model modification, e.g. kd_ratio*=0 (repeatable)")
    s.add_argument("--max-delay", type=float, default=0.06, help="longest delay of the delayed-goal baseline [s]")
    h = sub.add_parser("holds", parents=[common])
    h.add_argument("--params", required=True, help="one params JSON")
    h.add_argument("--dataset", action="append", required=True, help="dataset folder; its processed/ logs are scored")
    h.add_argument("--trajectory", default="steps")
    h.add_argument("--kps", default="80,120")
    h.add_argument("--band", default="3.5:8", help="frequency band lo:hi [Hz]")
    h.add_argument("--min-hold", type=float, default=0.9, help="shortest hold scored [s]")
    h.add_argument("--settle", type=float, default=0.4, help="skipped at the start of each hold [s]")
    args = ap.parse_args()
    run = {"transfer": run_transfer, "stream": run_stream, "holds": run_holds}[args.cmd]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        result = run(args, pool)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.json, "w"), indent=1, default=float)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
