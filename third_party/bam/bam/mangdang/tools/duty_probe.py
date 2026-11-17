"""What the AT32's mode-1 duty is made of: regress the probed ``pwm_duty`` on error, velocity and target rate.

    .venv/bin/python bam/mangdang/tools/duty_probe.py --logdir datasets/md01/id_3/unit_A/raw/probe

Note: this free regression is only a first look. Error and velocity are collinear
in a servo response, so it splits the duty between them arbitrarily. The D-term
gain in ``MD01PwmActuator`` came from constant-velocity ramps instead, with the
P law subtracted and a kd-0 twin differenced (``bam.mangdang.ramp_probe``). Non-recorder files in the
directory (e.g. the ramp probes) are skipped.

Input: raw recorder logs made with ``--live pwm_duty`` (one GET_LIVE per sample). For
every log and pooled, least squares of

    duty = a * error_deg + b * dq + c * dgoal_dt + d

on the samples with torque on and |duty| below the saturation, over a lag scan (the
live read trails the position sample) and a velocity low-pass scan. Reads:
``a`` against kp * kff_current (the P law), ``b`` / kd = the D gain in duty per
(rad/s) per unit kd (``MD01PwmActuator.KD_GAIN``), ``c`` ~ -b means the D term acts on
the error rate (target rate included), ``c`` ~ 0 on the velocity, the best low-pass
time constant is ``KD_TAU``, the residual says how much of the duty is unexplained.
"""

import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logdir", required=True)
ap.add_argument("--lags", type=str, default="-3,-2,-1,0,1,2,3", help="duty shift in samples to scan")
ap.add_argument("--taus", type=str, default="0,0.005,0.01,0.02,0.04", help="velocity low-pass time constants [s]")
ap.add_argument("--sat", type=float, default=0.95, help="drop samples with |duty| above sat * max_pwm")
args = ap.parse_args()


def lowpass(x, t, tau):
    if tau <= 0:
        return x.copy()
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        a = dt / (tau + dt)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def fit(duty, err, v, dg, keep, lag):
    """Least squares at one lag. The shift is applied to the full series and
    the ``keep`` mask afterwards, so an excised stretch never pairs
    non-adjacent samples."""
    n = len(duty)
    if lag >= 0:
        y, sl = duty[lag:], slice(0, n - lag)
    else:
        y, sl = duty[:n + lag], slice(-lag, n)
    ok = keep[sl] & keep[lag:] if lag >= 0 else keep[sl] & keep[:n + lag]
    y = y[ok]
    A = np.column_stack([err[sl][ok], v[sl][ok], dg[sl][ok], np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1.0 - np.sum(res**2) / max(np.sum((y - y.mean())**2), 1e-12)
    return coef, float(np.sqrt(np.mean(res**2))), r2, len(y)


files = sorted(glob.glob(os.path.join(args.logdir, "*.json")))
lags = [int(x) for x in args.lags.split(",")]
taus = [float(x) for x in args.taus.split(",")]
pooled = []
print(f"{'log':26s} {'traj':14s} {'kp':>4s} {'kd':>4s} {'n':>5s} {'lag':>3s} {'tau':>5s} {'a[/deg]':>8s} {'kp*kff':>7s} "
      f"{'b[/rad/s]':>9s} {'b/kd':>9s} {'c':>8s} {'d':>7s} {'rms':>6s} {'R2':>5s}")
for f in files:
    d = json.load(open(f))
    if not isinstance(d, dict) or "entries" not in d:
        print(f"{os.path.basename(f)}: not a recorder log, skipped")
        continue
    e = d["entries"]
    if "pwm_duty" not in e[0]:
        print(f"{os.path.basename(f)}: no pwm_duty column, skipped")
        continue
    t = np.array([x["timestamp"] for x in e])
    q = np.array([x["position"] for x in e])
    g = np.array([x["goal_position"] for x in e])
    en = np.array([x["torque_enable"] for x in e], dtype=bool)
    duty = np.array([x["pwm_duty"] for x in e], dtype=float)
    # the AT32 occasionally reports a denormal float for pwm_duty; treat it as 0
    duty[np.abs(duty) < 1e-30] = 0.0
    kp, kd = float(d["kp"]), float(d.get("at32_kd_position", d.get("kd", 0.0)))
    kff = d.get("at32_kff_current")
    max_pwm = float(d.get("at32_max_pwm_duty_cycle", 1.0))
    err = np.degrees(g - q)
    v_raw = np.gradient(q, t)
    dg = np.gradient(g, t)
    best = None
    for tau in taus:
        v = lowpass(v_raw, t, tau)
        keep = en & (np.abs(duty) < args.sat * max_pwm)
        for lag in lags:
            coef, rms, r2, used = fit(duty, err, v, dg, keep, lag)
            if best is None or rms < best[2]:
                best = (lag, tau, rms, r2, coef, used)
    lag, tau, rms, r2, (a, b, c, dd), n = best
    pooled.append((f, kp, kd, kff, a, b, c, rms, r2, lag, tau))
    print(f"{os.path.basename(f)[:26]:26s} {d['trajectory'][:14]:14s} {kp:4.0f} {kd:4.0f} {n:5d} {lag:3d} {tau:5.3f} "
          f"{a:8.5f} {(kp * kff) if kff else float('nan'):7.4f} {b:9.5f} {(b / kd) if kd else float('nan'):9.2e} "
          f"{c:8.5f} {dd:7.4f} {rms:6.4f} {r2:5.3f}")
if pooled:
    print("\nreading: a/(kp*kff) = the P scale ratio; b/kd = KD_GAIN candidate (sign: negative = damping); "
          "c ~ -b -> error-rate D, c ~ 0 -> velocity D; lag = samples the duty trails the position sample.")
