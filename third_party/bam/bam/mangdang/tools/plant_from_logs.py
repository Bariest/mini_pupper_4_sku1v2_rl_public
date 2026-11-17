"""Plant constants of one MD01 channel from bench data, without a fit.

Three measurements, each from a different kind of data, so that two shafts or
two motors can be compared on the same footing:

``ladder``     blocked-output ladder (``bam.mangdang.stall`` CSV, mode 1): per
               level the motor voltage ``|duty| * vin`` against the reported
               current; R and V0 from the levels above ``--min-a`` (``V = R*I +
               V0``), the knee from the levels below, and the stall current.
               Motor-terminal quantities: they do not involve the gearbox.
``holds``      quasi-static samples of the loaded recorder logs (smoothed
               |speed| below ``--still``; the trajectories are smooth, so these
               are the turning points): ``|I| = |tau_gravity| / kt + d * f / kt``
               with d = +1 when the arm rose into the point and -1 when it
               fell, so the static friction band f cancels between the two
               directions. Output torque constant, includes the ratio. The
               absolute value depends on the friction assumption (0.66 on
               the id_3 dataset through the origin, 0.45 with a free intercept,
               because the band grows with load), so compare channels with the
               same method on the same blocks.
``saturated``  duty-saturated samples in recorder logs: ``dI/d(omega)`` at a
               fixed applied voltage is ``-kt_emf / R`` (the back-EMF constant
               at the output, over R). Includes the ratio. The speed is a
               central difference over +-``--half`` samples (the logged speed is
               noisy enough to bias the slope toward zero), and two selections
               are printed: all samples whose P term exceeds the ceiling
               (-41.6 mA/(rad/s) on the id_3 dataset), and those that stay
               saturated after the kd-400 D term and a margin (-37 to -39 there):
               the ~10 % between them is the method's own spread, so compare
               channels with the same selection.

    .venv/bin/python bam/mangdang/tools/plant_from_logs.py ladder datasets/md01/id_3/unit_A/raw/at32_stall_test.csv \\
        --label "shipped kff, bench PSU 12V"
    .venv/bin/python bam/mangdang/tools/plant_from_logs.py holds datasets/md01/id_3/unit_A/processed
    .venv/bin/python bam/mangdang/tools/plant_from_logs.py saturated datasets/md01/id_3/unit_A/processed

Currents are the AT32's reported values (a bench-supply reading at duty 0.99,
where the supply current is the motor current, gave ~1.1 A against 1.03 A
reported).
Signs are taken as they come: ``reverse_motor`` differs between
channels, so every estimate works on magnitudes or on sign-aligned pairs.
"""

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

G = 9.81

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("ladder")
p.add_argument("csv")
p.add_argument("--label", default="", help="substring of the CSV's label column to select one ladder")
p.add_argument("--vin", type=float, default=12.0)
p.add_argument("--min-a", type=float, default=0.25, help="levels above this current set R and V0")
p = sub.add_parser("holds")
p.add_argument("logdir")
p.add_argument("--trajectory", default=None, help="only this trajectory (default: all loaded logs)")
p.add_argument("--still", type=float, default=0.05, help="smoothed |speed| below this is quasi-static [rad/s]")
p.add_argument("--half", type=int, default=5, help="speed = central difference over +-half samples")
p.add_argument("--min-torque", type=float, default=0.05, help="ignore samples below this gravity torque [Nm]")
p = sub.add_parser("saturated")
p.add_argument("logdir")
p.add_argument("--margin", type=float, default=0.05, help="duty beyond the ceiling required to count as saturated")
p.add_argument("--max-ma", type=float, default=1250.0, help="drop samples near the 1306 mA reading ceiling")
p.add_argument("--R", type=float, default=None, help="convert the slope to kt_emf with this R [ohm]")
p.add_argument("--half", type=int, default=2, help="speed = central difference over +-half samples")
args = parser.parse_args()


def load_logs(logdir):
    for path in sorted(glob.glob(os.path.join(logdir, "*.json"))):
        log = json.load(open(path))
        if isinstance(log, dict) and "entries" in log:
            yield path, log


def ladder():
    rows = [r for r in csv.DictReader(open(args.csv)) if args.label in r["label"]]
    if not rows:
        raise SystemExit(f"no rows with label containing {args.label!r}")
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["label"], float(r["level_deg"])), []).append(r)
    table = []
    for (label, level), rs in groups.items():
        t = np.array([float(r["t"]) for r in rs])
        tail = [r for r, tt in zip(rs, t) if tt - t[0] >= (t[-1] - t[0]) / 2.0]
        duty = abs(np.mean([float(r["pwm_duty"]) for r in tail]))
        cur = abs(np.mean([float(r["present_cur_mA"]) for r in tail])) / 1000.0
        err = abs(np.mean([float(r["error_pos_deg"]) for r in tail]))
        temp = np.mean([float(r["temp_c"]) for r in tail])
        table.append((label, level, err, duty, duty * args.vin, cur, temp))
    print(f"{'level':>6s} {'err deg':>8s} {'duty':>6s} {'V':>6s} {'I A':>7s} {'V/I ohm':>8s} {'T C':>5s}")
    for label, level, err, duty, v, i, temp in table:
        print(f"{level:6.2f} {err:8.2f} {duty:6.3f} {v:6.2f} {i:7.3f} {v / i if i > 0.005 else float('nan'):8.1f} {temp:5.1f}")
    hi = [(v, i) for _, _, _, _, v, i, _ in table if i >= args.min_a]
    if len(hi) >= 2:
        v, i = np.array(hi).T
        R, V0 = np.polyfit(i, v, 1)
        rms = np.sqrt(np.mean((i - (v - V0) / R) ** 2)) * 1000.0
        print(f"R {R:.2f} ohm, V0 {V0:.2f} V from {len(hi)} levels above {args.min_a} A ({rms:.1f} mA rms)")
    sat = [(d, i) for _, _, _, d, _, i, _ in table if d >= 0.97]
    if sat:
        print(f"stall current at duty {max(sat)[0]:.3f}: {max(sat)[1]:.3f} A")
    linear = [(err, duty) for _, _, err, duty, _, _, _ in table if duty < 0.97]
    if len(linear) >= 2:
        err, duty = np.array(linear).T
        print(f"law: {np.sum(err * duty) / np.sum(err * err):.4f} duty/deg below saturation (through the origin)")


def holds():
    rows, n_logs, h = [], 0, args.half
    for path, log in load_logs(args.logdir):
        if log["mass"] <= 0 or (args.trajectory and log["trajectory"] != args.trajectory):
            continue
        n_logs += 1
        e = log["entries"]
        dt = log.get("dt", 0.005)
        q = np.array([x["position"] for x in e])
        cur = np.array([x["load"] for x in e]) / 1000.0
        en = np.array([x["torque_enable"] for x in e], dtype=bool)
        v = np.full_like(q, np.inf)
        v[h:len(q) - h] = (q[2 * h:] - q[:len(q) - 2 * h]) / (2 * h * dt)
        tau = (log["mass"] + log["arm_mass"] / 2.0) * G * log["length"] * np.sin(q)
        back = int(0.2 / dt)
        m = en & (np.abs(v) < args.still) & (np.abs(tau) >= args.min_torque) & (np.abs(cur) > 0.02)
        m[:back] = False
        for k in np.where(m)[0]:
            rose = np.sign(q[k] - q[k - back]) == np.sign(tau[k])
            rows.append((abs(tau[k]), 1.0 if rose else -1.0, abs(cur[k])))
    if not rows:
        raise SystemExit("no quasi-static samples found")
    x = np.array(rows)
    (a, b), *_ = np.linalg.lstsq(x[:, :2], x[:, 2], rcond=None)
    (a2, b2, c2), *_ = np.linalg.lstsq(np.column_stack([x[:, :2], np.ones(len(x))]), x[:, 2], rcond=None)
    n_up = int(np.sum(x[:, 1] > 0))
    print(f"{len(x)} quasi-static samples in {n_logs} logs ({n_up} rose into the point, {len(x) - n_up} fell)")
    print(f"kt {1 / a:.3f} Nm/A, static band {b / a:.3f} Nm   (through the origin)")
    print(f"kt {1 / a2:.3f} Nm/A, static band {b2 / a2:.3f} Nm, offset {c2 * 1000:+.0f} mA   (free intercept)")


def saturated():
    sel = {"P beyond the ceiling": ([], []), "saturated after D and margin": ([], [])}
    n_logs = 0
    h = args.half
    for path, log in load_logs(args.logdir):
        e = log["entries"]
        dt = log.get("dt", 0.005)
        kff = log.get("at32_kff_current", 0.001)
        ceiling = log.get("at32_max_pwm_duty_cycle", 0.99)
        kd = log.get("at32_kd_position", log.get("kd", 0.0))
        q = np.array([x["position"] for x in e])
        err = np.degrees(np.array([x["goal_position"] for x in e]) - q)
        v = np.zeros_like(q)
        v[h:len(q) - h] = (q[2 * h:] - q[:len(q) - 2 * h]) / (2 * h * dt)
        cur = np.array([x["load"] for x in e])
        ok = np.array([x["torque_enable"] for x in e], dtype=bool) & (np.abs(cur) < args.max_ma)
        ok[:h] = ok[len(q) - h:] = False
        p = log["kp"] * kff * err
        masks = {
            "P beyond the ceiling": ok & (np.abs(p) >= ceiling),
            # the D term (2.87e-5 duty per rad/s per unit kd) pulls a marginal P back inside
            "saturated after D and margin": ok & (np.abs(p) >= ceiling + args.margin + kd * 2.87e-5 * np.abs(v)),
        }
        if masks["P beyond the ceiling"].any():
            n_logs += 1
        for name, m in masks.items():
            if not m.any():
                continue
            # drive direction = sign of the error; the reported current's sign follows
            # the motor polarity, so orient it by its correlation with the drive
            s = np.sign(p[m])
            c = cur[m] * np.sign(np.sum(cur[m] * s) or 1.0)
            sel[name][0].append(s * v[m])
            sel[name][1].append(s * c)
    if not sel["P beyond the ceiling"][0]:
        raise SystemExit("no saturated samples")
    print(f"logs with saturated samples: {n_logs}; speed = central difference over +-{h} samples")
    for name, (xs, ys) in sel.items():
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        slope, icpt = np.polyfit(x, y, 1)
        print(f"  {name:30s} n={len(x):6d}  dI/d(omega) {slope:+6.1f} mA/(rad/s)  intercept {icpt:5.0f} mA"
              + (f"  kt_emf {-slope / 1000.0 * args.R:.3f} V.s/rad at R {args.R}" if args.R else ""))


{"ladder": ladder, "holds": holds, "saturated": saturated}[args.cmd]()
