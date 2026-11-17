"""Per-log detail: angle, speed and CURRENT for one or more models.

    .venv/bin/python bam/mangdang/tools/plot_detail.py --logdir datasets/md01/id_3/unit_A/processed --kp 120 \
        --params bam/params/md01/id_3/m1.json bam/params/md01/id_3/m5.json bam/params/md01/id_3/m6.json \
        --select up_and_down:0.3034:0.10 --out-dir figures

`bam.plot` draws the actuator's control signal in its own unit, which for a
voltage law is volts; for the MD01 the quantity worth looking at is the
current, because the fit never sees it. Here the third row is the measured
`load` against each model's `compute_current` along its own rollout, so the
panel answers "does this model draw what the motor drew".

`--select` takes `trajectory:mass:length` triples (mass and length as they
appear in the log, to 5 and 2 decimals); omit it to plot every log at `--kp`.
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
ap.add_argument("--params", nargs="+", required=True)
ap.add_argument("--select", nargs="*", default=[], help="trajectory:mass:length")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--labels", nargs="+", default=None, help="series labels, one per --params file")
ap.add_argument("--q-offset", type=float, nargs="+", default=None,
                help="roll out at this q_offset instead of each params file's own (one value, or one per --params file)")
ap.add_argument("--trim", type=int, default=0, help="drop this many samples at the start of every log")
ap.add_argument("--unseen", action="store_true", help="the logs are from a unit the params were not fitted on")
args = ap.parse_args()

MEAS, GOAL = "#2a78d6", "#555555"
SERIES = ["#eb6834", "#1baf7a", "#4a3aa7"]
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
want = {tuple(s.split(":")) for s in args.select}
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

for log in Logs(args.logdir).logs:
    if float(log["kp"]) != args.kp:
        continue
    log["entries"] = log["entries"][args.trim:]
    key = (log["trajectory"], f"{log['mass']:.5f}", f"{log['length']:.2f}")
    if want and key not in {(t, f"{float(m):.5f}", f"{float(l):.2f}") for t, m, l in want}:
        continue
    t = np.arange(len(log["entries"])) * log["dt"]
    q = np.array([e["position"] for e in log["entries"]])
    g = np.array([e["goal_position"] for e in log["entries"]])
    cur = np.array([e["load"] for e in log["entries"]]) / 1000.0
    en = np.array([e["torque_enable"] for e in log["entries"]], dtype=bool)

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, g, "--", color=GOAL, lw=1.1, label="goal")
    ax[0].plot(t, q, color=MEAS, lw=1.6, label="measured")
    ax[1].plot(t, np.gradient(q, log["dt"]), color=MEAS, lw=0.8, alpha=.7, label="measured")
    ax[2].plot(t, np.abs(cur) * 1000, color=MEAS, lw=1.2, label="measured |I|")

    for k, (name, model) in enumerate(models):
        sim = simulate.Simulator(model)
        pos, vel, ctrl = sim.rollout_log(log, simulate_control=True)
        n = min(len(pos), len(t))
        pos, vel = np.array(pos[:n]), np.array(vel[:n])
        mae = np.mean(np.abs(pos - q[:n])) * 1e3
        c = SERIES[k % len(SERIES)]
        ax[0].plot(t[:n], pos, color=c, lw=1.3, label=f"{name} ({mae:.0f} mrad)")
        ax[1].plot(t[:n], vel, color=c, lw=1.1, label=name)
        volts = np.array([x if x is not None else 0.0 for x in ctrl[:n]], dtype=float)
        isim = model.actuator.compute_current(volts, vel) * en[:n]
        imae = np.mean(np.abs(np.abs(isim[en[:n]]) - np.abs(cur[:n][en[:n]]))) * 1000
        ax[2].plot(t[:n], np.abs(isim) * 1000, color=c, lw=1.2, label=f"{name} ({imae:.0f} mA)")

    for a in ax:
        a.grid(alpha=.25)
        a.legend(fontsize=7.5, ncol=2)
    if (~en).any():
        for a in ax:
            a.fill_between(t, *a.get_ylim(), where=~en, color="#d62728", alpha=.08, lw=0)
    ax[0].set_ylabel("angle [rad]"); ax[1].set_ylabel("speed [rad/s]"); ax[2].set_ylabel("|current| [mA]")
    ax[2].set_xlabel("time [s]   (shaded: torque off)")
    grams = int(round(log["mass"] * 1000)); cm = int(round(log["length"] * 100))
    # channel n of a module is output id_n
    output = f"id_{(int(log['servo_id']) - 1) % 3 + 1}" if "servo_id" in log else "?"
    ax[0].set_title(f"{log['trajectory']}, {grams} g at {cm} cm, kp {args.kp:g} "
                    f"({'unit never fitted' if args.unseen else 'held out'})   {output}", fontsize=10, loc="left")
    out = Path(args.out_dir) / f"{log['trajectory']}_{grams}g_{cm}cm_kp{args.kp:g}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")
