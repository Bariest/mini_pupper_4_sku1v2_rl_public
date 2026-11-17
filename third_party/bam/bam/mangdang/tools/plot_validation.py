"""Held-out-set overview: one panel per (block, trajectory), position only.

    .venv/bin/python bam/mangdang/tools/plot_validation.py --logdir datasets/md01/id_3/unit_A/processed --kp 120 \
        --params bam/params/md01/id_3/m1.json bam/params/md01/id_3/m5.json bam/params/md01/id_3/m6.json \
        --out validation_kp120.png

Goal, measured and each model's rollout (reference simulator, control
recomputed from the model) are overlaid; the panel title carries each model's
MAE so the picture and the number agree.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bam import simulate
from bam.logs import Logs
from bam.model import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--logdir", required=True)
ap.add_argument("--kp", type=float, required=True)
ap.add_argument("--validation_kp", type=float, default=120.0, help="the held-out gain, for the title")
ap.add_argument("--params", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trajectories", default="lift_and_drop,up_and_down,sin_time_square,sin_sin,steps")
ap.add_argument("--labels", nargs="+", default=None, help="series labels, one per --params file")
ap.add_argument("--q-offset", type=float, nargs="+", default=None,
                help="roll out at this q_offset instead of each params file's own (one value, or one per --params file)")
ap.add_argument("--trim", type=int, default=0, help="drop this many samples at the start of every log")
ap.add_argument("--unseen", action="store_true", help="the logs are from a unit the params were not fitted on")
args = ap.parse_args()

MEASURED, GOAL = "#2a78d6", "#555555"
SERIES = ["#eb6834", "#1baf7a", "#4a3aa7"]  # fixed slots: 1st, 2nd, 3rd params file
stems = [Path(p).stem for p in args.params]
# the parent folder tells same-named files (m6.json of different sets) apart
names = [f"{Path(p).parent.name}/{s}" if stems.count(s) > 1 else s for p, s in zip(args.params, stems)]
if args.labels:
    if len(args.labels) != len(args.params):
        raise SystemExit("--labels needs one label per --params file")
    names = args.labels
models = [(name, load_model(p)) for name, p in zip(names, args.params)]
if args.q_offset is not None:
    if len(args.q_offset) not in (1, len(models)):
        raise SystemExit("--q-offset needs one value or one per --params file")
    for (_, model), q in zip(models, args.q_offset * len(models) if len(args.q_offset) == 1 else args.q_offset):
        model.q_offset.value = q
trajs = args.trajectories.split(",")
logs = [l for l in Logs(args.logdir).logs if float(l["kp"]) == args.kp and l["trajectory"] in trajs]
for l in logs:
    l["entries"] = l["entries"][args.trim:]
blocks = sorted({(l["mass"], l["length"]) for l in logs}, key=lambda b: (-b[1], b[0]))
fig, axes = plt.subplots(len(blocks), len(trajs), figsize=(3.4 * len(trajs), 2.1 * len(blocks)), sharex=True)
for i, (mass, length) in enumerate(blocks):
    tau = (mass * length + 0.0196 * length / 2) * 9.81 if length > 0.12 else (mass * length + 0.016 * length / 2) * 9.81
    for j, tr in enumerate(trajs):
        ax = axes[i, j]
        log = next((l for l in logs if l["mass"] == mass and l["length"] == length and l["trajectory"] == tr), None)
        if log is None:
            ax.set_visible(False); continue
        t = np.array([e["timestamp"] for e in log["entries"]])
        q = np.array([e["position"] for e in log["entries"]])
        g = np.array([e["goal_position"] for e in log["entries"]])
        en = np.array([e["torque_enable"] for e in log["entries"]])
        g_plot = np.where(en, g, np.nan)
        ax.plot(t, g_plot, color=GOAL, lw=1.0, ls="--", label="goal")
        ax.plot(t, q, color=MEASURED, lw=2.0, label="measured")
        maes = []
        for k, (name, model) in enumerate(models):
            sim = simulate.Simulator(model)
            pos, _, _ = sim.rollout_log(log, simulate_control=True)
            pos = np.array(pos[: len(q)])
            maes.append(np.mean(np.abs(pos - q)) * 1e3)
            ax.plot(t[: len(pos)], pos, color=SERIES[k % len(SERIES)], lw=1.4, label=name)
        ax.set_title(f"{tr.replace('_', ' ')}   " + "  ".join(f"{n} {m:.0f}" for (n, _), m in zip(models, maes)) + " mrad",
                     fontsize=8, loc="left")
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel(f"{mass * 1000:.0f} g @ {length * 100:.0f} cm\n{tau:.2f} Nm\nangle [rad]", fontsize=8)
        if i == len(blocks) - 1:
            ax.set_xlabel("time [s]", fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9, frameon=False)
if args.unseen:
    label = "a unit the fit never saw"
else:
    label = "held-out set, never seen by the fit" if args.kp == args.validation_kp else "training set"
# channel n of a module is output id_n
outputs = {f"id_{(int(l['servo_id']) - 1) % 3 + 1}" for l in logs if "servo_id" in l}
who = f"MD01 {outputs.pop()}" if len(outputs) == 1 else "MD01"
fig.suptitle(f"{who} — kp {args.kp:g} ({label}): measured vs model rollouts", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
print(f"saved {args.out} ({len(logs)} logs)")
