#!/usr/bin/env python
"""Evaluate a directory of fitted BAM params on a log directory.

For every ``m*.json`` in ``--fits``: print the parameters with a flag when a
value sits within 2 % of its optimisation bound, then the position MAE [mrad]
split into train / validation (``--validation_kp``) and per (mass, length,
trajectory) block, using the reference simulator or, with ``--mujoco``, the
MuJoCo CPU backend.

    .venv/bin/python bam/mangdang/tools/eval_fits.py --fits bam/params/md01/id_3 --logdir datasets/md01/id_3/unit_A/processed \
        --validation_kp 120 --current
"""

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

parser = argparse.ArgumentParser()
parser.add_argument("--fits", required=True)
parser.add_argument("--logdir", required=True)
parser.add_argument("--validation_kp", type=float, default=140.0)
parser.add_argument("--mujoco", action="store_true")
parser.add_argument("--json", default=None, help="write the summary here")
parser.add_argument("--extra", action="append", default=[], help="extra log directories evaluated separately (repeatable)")
parser.add_argument("--current", action="store_true", help="also report the simulated-vs-measured current MAE [mA]")
parser.add_argument(
    "--trim",
    type=int,
    default=0,
    help="drop this many samples at the start of every log (bam.process extrapolates a log back to t = 0, "
    "so its first processed samples can carry a large speed)",
)
parser.add_argument("--duty", action="store_true", help="also report the simulated-vs-recorded duty MAE on logs that carry pwm_duty (probe logs)")
parser.add_argument(
    "--refit-q-offset",
    action="store_true",
    help="fit q_offset alone on the --logdir logs before scoring: it is the rig's gravity zero, not the "
    "actuator, so a model evaluated on another rig (another servo, another mount) must not be charged for it",
)
parser.add_argument(
    "--register",
    default=None,
    help="python file to import first (registers extra actuators, e.g. a scratch class)",
)
args = parser.parse_args()

if args.register:
    import importlib.util

    spec = importlib.util.spec_from_file_location("extra_actuators", args.register)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

from bam import simulate
from bam.actuators import actuators
from bam.logs import Logs
from bam.model import load_model, models

if args.mujoco:
    from bam import mujoco as mujoco_backend

def load_logs(directory):
    logs = Logs(directory)
    for log in logs.logs:
        log["entries"] = log["entries"][args.trim:]
    return logs


logs = load_logs(args.logdir)


def simulated_current(model, controls, dq):
    """Motor current [A] from the control the simulator returned (amps, or volts through the plant)."""
    if model.actuator.control_unit() == "amps":
        return controls
    # Voltage law: the control is already in volts (vin * duty). A class that
    # models its own plant (md01v: back-EMF constant, soft knee) says how the
    # current follows; the fallback below is the plain DC motor with a hard
    # brush-drop dead band and kt as the back-EMF constant.
    if hasattr(model.actuator, "compute_current"):
        return np.asarray(model.actuator.compute_current(controls, dq), dtype=float)
    V0 = model.V0.value if hasattr(model, "V0") else 0.0
    x = controls - model.kt.value * dq
    return np.sign(x) * np.maximum(np.abs(x) - V0, 0.0) / model.R.value


def rollout(model, log):
    """Position MAE [mrad] and, with --current / --duty, the current [mA] / duty MAE while torque is on."""
    if args.mujoco:
        sim = mujoco_backend.Simulator(model, command_delay=True)
        positions, _, controls = sim.rollout_log(log)
    else:
        sim = simulate.Simulator(model)
        positions, _, controls = sim.rollout_log(log, simulate_control=True)
    ref = np.array([e["position"] for e in log["entries"]])
    n = min(len(ref), len(positions))
    pmae = float(np.mean(np.abs(np.array(positions[:n]) - ref[:n]))) * 1000.0
    if not (args.current or args.duty):
        return pmae, None, None
    en = np.array([e["torque_enable"] for e in log["entries"]])[:n]
    ctrl = np.array([c if c is not None else 0.0 for c in controls[:n]], dtype=float)
    imae = dmae = None
    if args.current:
        meas = np.array([e["load"] for e in log["entries"]])[:n] / 1000.0
        dq = np.gradient(ref[:n], log["dt"])
        cur = simulated_current(model, ctrl, dq)
        sign = np.sign(np.sum(meas[en] * cur[en])) or 1.0
        imae = float(np.mean(np.abs(sign * cur[en] - meas[en]))) * 1000.0
    if args.duty and "pwm_duty" in log["entries"][0] and model.actuator.control_unit() != "amps":
        rec = np.array([e["pwm_duty"] for e in log["entries"]])[:n]
        # the class takes its rail from supply_v when the log carries it
        rail = log.get("supply_v") or log.get("vin", 12.0)
        sim_duty = ctrl / float(np.asarray(rail).reshape(-1)[0])
        sign = np.sign(np.sum(rec[en] * sim_duty[en])) or 1.0
        dmae = float(np.mean(np.abs(sign * sim_duty[en] - rec[en])))
    return pmae, imae, dmae


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float("nan")


def refit_q_offset(model, logs_list, lo, hi, iterations=24):
    """Golden-section search of the mean position MAE over ``q_offset`` in [lo, hi]."""
    def cost(q):
        model.q_offset.value = q
        return float(np.mean([rollout(model, log)[0] for log in logs_list]))

    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = cost(d)
    best = c if fc < fd else d
    model.q_offset.value = best
    return best


summary = {}
for params_file in sorted(Path(args.fits).glob("m*.json")):
    data = json.load(open(params_file))
    if not data:
        print(f"== {params_file.name}: empty (fit not started?)")
        continue
    model = load_model(str(params_file))

    # Bounds from a freshly built model of the same variant.
    fresh = models[data["model"]]()
    fresh.set_actuator(actuators[data["actuator"]]())
    bounds = {k: (p.min, p.max) for k, p in fresh.get_parameters().items()}

    print(f"== {params_file.name}  ({data['model']}, actuator {data['actuator']})")
    for key, value in data.items():
        if key in bounds:
            lo, hi = bounds[key]
            near = "  <-- at bound" if (value - lo) < 0.02 * (hi - lo) or (hi - value) < 0.02 * (hi - lo) else ""
            print(f"   {key:32s} {value:12.5g}   [{lo:g}, {hi:g}]{near}")

    if args.refit_q_offset:
        fitted_from = model.q_offset.value
        lo, hi = bounds.get("q_offset", (-0.1, 0.1))
        fitted = refit_q_offset(model, logs.logs, lo, hi)
        print(f"   q_offset refit on {args.logdir}: {fitted_from:+.4f} -> {fitted:+.4f} rad")

    per_block = collections.defaultdict(list)
    train, val, cur, dut = [], [], [], []
    rows = []
    for log in logs.logs:
        mae, imae, dmae = rollout(model, log)
        rows.append({"file": os.path.basename(log["filename"]), "trajectory": log["trajectory"], "mass": log["mass"],
                     "length": log["length"], "kp": float(log["kp"]), "kd": float(log.get("kd", 400.0)),
                     "temp_start_c": log.get("temp_live_start_c"), "mae": mae, "current_mae_mA": imae})
        per_block[(log["mass"], log["length"], log["trajectory"])].append(mae)
        (val if float(log["kp"]) == args.validation_kp else train).append(mae)
        if imae is not None and log["mass"] > 0:
            cur.append(imae)
        if dmae is not None:
            dut.append(dmae)
    print(f"   MAE train {mean_or_nan(train):6.1f} mrad   validation(kp={args.validation_kp:g}) "
          f"{mean_or_nan(val):6.1f} mrad   all {mean_or_nan(train + val):6.1f} mrad"
          + (f"   current MAE {np.mean(cur):5.0f} mA (loaded logs)" if cur else "")
          + (f"   duty MAE {np.mean(dut):6.3f}" if dut else ""))
    extras = {}
    for extra_dir in args.extra:
        e_pos, e_cur, e_dut = [], [], []
        for log in load_logs(extra_dir).logs:
            mae, imae, dmae = rollout(model, log)
            e_pos.append(mae)
            if imae is not None and log["mass"] > 0:  # loaded logs, as in the main set
                e_cur.append(imae)
            if dmae is not None:
                e_dut.append(dmae)
        extras[os.path.basename(extra_dir.rstrip("/"))] = float(np.mean(e_pos))
        print(f"   extra {os.path.basename(extra_dir.rstrip('/')):8s} {np.mean(e_pos):6.1f} mrad"
              + (f"   current {np.mean(e_cur):5.0f} mA" if e_cur else "")
              + (f"   duty {np.mean(e_dut):6.3f}" if e_dut else ""))
    blocks = collections.defaultdict(list)
    for (mass, length, traj), v in sorted(per_block.items()):
        blocks[(mass, length)] += v
        print(f"     m={mass:.3f} l={length:.3f} {traj:16s} {np.mean(v):6.1f}")
    for (mass, length), v in sorted(blocks.items()):
        print(f"     m={mass:.3f} l={length:.3f} {'(block)':16s} {np.mean(v):6.1f}")
    summary[params_file.stem] = {
        "train": mean_or_nan(train),
        "validation": mean_or_nan(val),
        "all": mean_or_nan(train + val),
        "q_offset": float(model.q_offset.value),
        "blocks": {f"m{m:.3f}_l{l:.3f}": float(np.mean(v)) for (m, l), v in blocks.items()},
        "current_mae_mA": float(np.mean(cur)) if cur else None,
        "extras": extras,
        "params": {k: v for k, v in data.items() if k in bounds},
        "logs": rows,
    }

if args.json:
    json.dump(summary, open(args.json, "w"), indent=2)
    print(f"wrote {args.json}")
