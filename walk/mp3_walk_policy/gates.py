"""Rollouts and the simulation gates (after the Mini Pupper 2 walking policy's gates, re-implemented for this environment).

Metrics per rollout, over the environments that did not fall, after a ``settle_s`` start:
- ``speed``: forward speed in the world x (every episode starts heading +x), m/s;
- ``lateral_mm``, ``yaw_deg``: the drift at the end;
- ``f0_hz``: the gait fundamental, the spectral peak (0.5-6 Hz) of the four hips' servo positions;
- ``clearance_mm``: the 95th percentile of each foot's height above the lowest foot;
- ``pitch_deg``: the RMS body pitch;
- ``fall_frac``: environments that terminated before the end.
The gates (``gate_table``) compare these with the command and, for a clone, with its teacher.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .contract import CONTROL_DT
from .env_cfg import TaskOptions, make_env_cfg
from .mdp import _foot_site_ids, _yaw, servo_action


def make_env(opt: TaskOptions, device: str = "cuda:0"):
    """An evaluation env. Rollouts must end before the episode's time-out (``rollout`` refuses otherwise)."""
    from mjlab.envs import ManagerBasedRlEnv
    return ManagerBasedRlEnv(cfg=make_env_cfg(opt), device=device)


@torch.no_grad()
def rollout(env, act_fn, seconds: float = 10.0, record_actions: bool = False):
    obs, _ = env.reset()
    n, steps = env.num_envs, int(round(seconds / CONTROL_DT))
    # The last step of an episode that times out records the auto-reset state, not the end of the walk.
    if steps >= env.max_episode_length:
        raise ValueError(f"rollout of {steps} steps in episodes of {env.max_episode_length}: lengthen episode_s")
    robot = env.scene["robot"]
    term = servo_action(env)
    feet = _foot_site_ids(env)
    rec = {k: [] for k in ("x", "y", "yaw", "pitch", "roll", "rel", "lift", "actions")}
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    for k in range(steps):
        a = act_fn(obs, env, k)
        if record_actions:
            rec["actions"].append(a.clone())
        obs, _, terminated, _, _ = env.step(a)
        fell |= terminated
        d = robot.data
        pos = d.root_link_pos_w - env.scene.env_origins
        rec["x"].append(pos[:, 0].clone())
        rec["y"].append(pos[:, 1].clone())
        rec["yaw"].append(_yaw(d.root_link_quat_w))
        g = d.projected_gravity_b
        rec["pitch"].append(torch.atan2(g[:, 0], -g[:, 2]))
        rec["roll"].append(torch.atan2(-g[:, 1], -g[:, 2]))
        rec["rel"].append(term.true_rel().clone())
        fz = d.site_pos_w[:, feet, 2]
        rec["lift"].append(fz - fz.amin(dim=1, keepdim=True))
    out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items() if v}
    out["fell"] = fell.cpu().numpy()
    return out


def metrics(r: dict, settle_s: float = 2.0) -> dict:
    k0 = int(round(settle_s / CONTROL_DT))
    ok = ~r["fell"]
    res = {"n": int(len(ok)), "fall_frac": float(1.0 - ok.mean())}
    if not ok.any():
        return res
    T = (r["x"].shape[0] - 1 - k0) * CONTROL_DT          # x[-1] - x[k0] spans N - 1 - k0 steps
    x, y = r["x"][:, ok], r["y"][:, ok]
    speed = (x[-1] - x[k0]) / T
    hips = r["rel"][k0:, :, 1::3][:, ok]                    # (t, env, 4)
    sig = hips - hips.mean(0, keepdims=True)
    spec = np.abs(np.fft.rfft(sig, axis=0)) ** 2
    freqs = np.fft.rfftfreq(sig.shape[0], CONTROL_DT)
    band = (freqs > 0.5) & (freqs < 6.0)
    power = spec.sum(axis=2)                                # (f, env)
    f0 = freqs[band][np.argmax(power[band], axis=0)]
    lift = r["lift"][k0:, ok]
    res.update({
        "speed": float(speed.mean()), "speed_sd": float(speed.std()),
        "lateral_mm": float(np.abs(y[-1] - y[0]).mean() * 1000.0),
        "yaw_deg": float(np.degrees(np.abs(r["yaw"][-1, ok])).mean()),
        "f0_hz": float(np.median(f0)),
        "clearance_mm": float(np.percentile(lift, 95, axis=0).mean() * 1000.0),
        "pitch_deg": float(np.degrees(np.sqrt((r["pitch"][k0:, ok] ** 2).mean()))),
    })
    return res


def teacher_fn(teacher, env, omni: bool = False):
    """``omni``: an omni teacher, fed the whole command."""
    term = servo_action(env)

    def act(obs, env_, k):
        cmd = env_.command_manager.get_command("twist")
        rel = teacher.target_rel(cmd if omni else cmd[:, 0], env_.episode_length_buf)
        return (rel / term.scale).clamp(-1.0, 1.0)
    return act


def policy_fn(actor):
    def act(obs, env, k):
        return actor(obs["actor"])
    return act


def replay_fn(actions):
    """Play a recorded action stream blind (MP2's open-loop bar)."""
    def act(obs, env, k):
        return actions[min(k, len(actions) - 1)]
    return act


FIRMWARE_F0_HZ = 1.0 / 0.36          # the firmware trot's cycle: 24 ticks x 15 ms


def gate_table(m: dict, cmd: float, teacher: dict | None = None) -> dict:
    """MP2's gates, for this robot: speed within 20 % of the command (of the teacher's, for a clone); the fundamental
    within 0.2 Hz of the teacher's for a clone, within 0.3 Hz of the firmware trot's 2.78 Hz for a policy (the gait
    family, as MP2's f0 gate); a clearance of at least 8 mm; an RMS pitch under 3 deg; no falls."""
    ref_speed = teacher["speed"] if teacher else cmd
    g = {
        "no_falls": m.get("fall_frac", 1.0) == 0.0,
        "speed": "speed" in m and abs(m["speed"] - ref_speed) <= 0.2 * ref_speed,
        "clearance": m.get("clearance_mm", 0.0) >= 8.0,
        "pitch": m.get("pitch_deg", 99.0) < 3.0,
    }
    if teacher:
        g["f0"] = "f0_hz" in m and abs(m["f0_hz"] - teacher["f0_hz"]) <= 0.2
    else:
        g["f0"] = "f0_hz" in m and abs(m["f0_hz"] - FIRMWARE_F0_HZ) <= 0.3
    g["pass"] = all(g.values())
    return g


# ------------------------------------------------------------------------------------------------------------ omni
# The omni evaluation set: eight directions at 0.10 m/s holding the heading, the corners of the speed range, turning
# in place both ways, and two curves.
OMNI_EVAL_COMMANDS = tuple(
    [(round(0.10 * math.cos(math.radians(a)), 4), round(0.10 * math.sin(math.radians(a)), 4), 0.0)
     for a in range(0, 360, 45)]
    + [(0.05, 0.0, 0.0), (0.12, 0.0, 0.0), (0.0, 0.12, 0.0), (0.0, -0.12, 0.0),
       (0.0, 0.0, 0.4), (0.0, 0.0, -0.4), (0.10, 0.0, 0.4), (0.0, 0.10, -0.4)])


def _command_name(c) -> str:
    return f"{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.2f}"


def omni_metrics(r: dict, commands, settle_s: float = 2.0) -> dict:
    """Per command (env i ran ``commands[i % len(commands)]``), over its environments that did not fall:
    ``along`` / ``cross``: the mean body-frame velocity along and across the commanded direction (m/s; for a turn in
    place, ``drift``: the body's mean speed); ``yaw_rate`` (rad/s); ``heading_deg``: the yaw change after the settle
    for a command without turning; ``f0_hz``, ``clearance_mm``, ``pitch_deg``, ``roll_deg`` as ``metrics``."""
    k0 = int(round(settle_s / CONTROL_DT))
    m = len(commands)
    n = r["x"].shape[1]
    yaw = np.unwrap(r["yaw"], axis=0)
    dx, dy = np.diff(r["x"], axis=0), np.diff(r["y"], axis=0)
    ym = 0.5 * (yaw[1:] + yaw[:-1])
    vbx = (np.cos(ym) * dx + np.sin(ym) * dy) / CONTROL_DT
    vby = (-np.sin(ym) * dx + np.cos(ym) * dy) / CONTROL_DT
    T = (yaw.shape[0] - 1 - k0) * CONTROL_DT
    out = {}
    for j, c in enumerate(commands):
        ids = np.arange(j, n, m)
        ok = ids[~r["fell"][ids]]
        res = {"command": list(c), "n": int(len(ids)), "fall_frac": float(1.0 - len(ok) / max(len(ids), 1))}
        if len(ok):
            sub = {k: v[:, ok] for k, v in r.items() if k not in ("fell", "actions")}
            sub["fell"] = np.zeros(len(ok), dtype=bool)
            base = metrics(sub, settle_s)
            v = np.stack([vbx[k0:, ok].mean(0), vby[k0:, ok].mean(0)], axis=-1)        # (env, 2)
            sp = math.hypot(c[0], c[1])
            if sp > 1e-6:
                u = np.array([c[0], c[1]]) / sp
                along, cross = v @ u, v[:, 0] * -u[1] + v[:, 1] * u[0]
                res.update(along=float(along.mean()), cross=float(cross.mean()))
            else:
                res.update(drift=float(np.linalg.norm(v, axis=-1).mean()))
            rate = (yaw[-1, ok] - yaw[k0, ok]) / T
            res.update(yaw_rate=float(rate.mean()), heading_deg=float(np.degrees(np.abs(yaw[-1, ok] - yaw[k0, ok])).mean()),
                       f0_hz=base["f0_hz"], clearance_mm=base["clearance_mm"], pitch_deg=base["pitch_deg"],
                       roll_deg=float(np.degrees(np.sqrt((r["roll"][k0:, ok] ** 2).mean()))))
        out[_command_name(c)] = res
    return out


# ----------------------------------------------------------------------------------------------- stops (takes)
TAKES_JSON = Path(__file__).resolve().parent / "data" / "takes_stops.json"
STOP_LIFT_MM = 3.0           # a stopped robot: every foot within this of the lowest (95th percentile)
STOP_DRIFT = 0.01            # and the body under this speed (m/s)


def take_schedules(path: Path = TAKES_JSON) -> dict:
    """The recorded takes' keyframe schedules (walks, turns in place, stops), by take name."""
    return {k: v["keyframes"] for k, v in json.loads(Path(path).read_text())["takes"].items()}


@torch.no_grad()
def play_schedules(schedules, arm, actor=None, n_per=1, randomise=False, noisy=False, lead_s=1.0, noise_end=None):
    """Each schedule on ``n_per`` envs, after ``lead_s`` standing: a policy (``actor``) with the command following the
    schedule (zero while stopped), or, ``actor=None``, the firmware playing it open loop (``takes.firmware_rel``)."""
    from .env_cfg import omni_options
    from .takes import duration_ms, firmware_rel
    steps = max(duration_ms(k) for k in schedules) // 20 + 50
    lead = int(round(lead_s / CONTROL_DT))
    total = lead + steps
    cmds, rels = [], []
    for kfs in schedules:
        c, r = firmware_rel(kfs, steps)
        cmds.append(np.concatenate([np.zeros((lead, 3)), c]))
        rels.append(np.concatenate([np.repeat(r[:1], lead, 0), r]))
    kw = {} if noise_end is None else {"noise_end": noise_end}
    env = make_env(omni_options(arm=arm, num_envs=len(schedules) * n_per, noisy=noisy, randomise=randomise,
                                noise_ramp=False, fixed_command=((0.0, 0.0, 0.0),),
                                episode_s=total * CONTROL_DT + 2.0, **kw))
    dev = env.device
    term = servo_action(env)
    cmd_term = env.command_manager.get_term("twist")
    C = torch.tensor(np.stack(cmds), dtype=torch.float32, device=dev).repeat_interleave(n_per, 0)    # (env, t, 3)
    R = torch.tensor(np.stack(rels), dtype=torch.float32, device=dev).repeat_interleave(n_per, 0)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    feet = _foot_site_ids(env)
    log = {k: [] for k in ("x", "y", "yaw", "lift")}
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    for k in range(total):
        cmd_term.vel_command_b[:] = C[:, k]
        if actor is None:
            a = (R[:, k] / term.scale).clamp(-1.0, 1.0)
        else:
            o = obs["actor"].clone()
            o[:, 6:9] = C[:, k]                          # the command in the observation (contract.py)
            a = actor(o)
        obs, _, terminated, _, _ = env.step(a)
        fell |= terminated
        d = robot.data
        pos = d.root_link_pos_w - env.scene.env_origins
        log["x"].append(pos[:, 0].clone())
        log["y"].append(pos[:, 1].clone())
        log["yaw"].append(_yaw(d.root_link_quat_w))
        fz = d.site_pos_w[:, feet, 2]
        log["lift"].append((fz - fz.amin(dim=1, keepdim=True)).amax(dim=1))
    env.close()
    out = {k: torch.stack(v).cpu().numpy() for k, v in log.items()}
    out["fell"] = fell.cpu().numpy()
    out["lead"] = lead
    out["n_per"] = n_per
    return out


def schedule_metrics(kfs, out, j: int) -> list:
    """Per keyframe hold of schedule ``j`` (after up to 0.5 s of settling), over its environments that did not fall:
    the command, ``along`` / ``cross`` (travel) or ``drift`` (none), ``yaw_rate``, and ``lift_mm`` (the 95th
    percentile of the highest foot above the lowest)."""
    from .takes import command_of
    n = out["n_per"]
    ids = np.arange(j * n, (j + 1) * n)
    ok = ids[~out["fell"][ids]]
    rows, t0 = [], 0
    if not len(ok):
        return [{"kf": i, "fall_frac": 1.0} for i in range(len(kfs))]
    yaw = np.unwrap(out["yaw"][:, ok], axis=0)
    dx, dy = np.diff(out["x"][:, ok], axis=0), np.diff(out["y"][:, ok], axis=0)
    ym = 0.5 * (yaw[1:] + yaw[:-1])
    vbx = (np.cos(ym) * dx + np.sin(ym) * dy) / CONTROL_DT
    vby = (-np.sin(ym) * dx + np.cos(ym) * dy) / CONTROL_DT
    for i, k in enumerate(kfs):
        settle = min(500, k["hold_ms"] // 2)
        a = out["lead"] + (t0 + k["ramp_ms"] + settle) // 20
        b = out["lead"] + (t0 + k["ramp_ms"] + k["hold_ms"]) // 20
        t0 += k["ramp_ms"] + k["hold_ms"]
        c = command_of(k)
        v = np.array([vbx[a:b].mean(), vby[a:b].mean()])
        row = {"kf": i, "stand": bool(k["stand"]), "command": c.round(4).tolist(),
               "fall_frac": float(1.0 - len(ok) / len(ids)),
               "yaw_rate": float((yaw[b] - yaw[a]).mean() / ((b - a) * CONTROL_DT)),
               "lift_mm": float(np.percentile(out["lift"][a:b, ok], 95) * 1000.0)}
        sp = math.hypot(c[0], c[1])
        if sp > 1e-6:
            u = c[:2] / sp
            row.update(along=float(v @ u), cross=float(-v[0] * u[1] + v[1] * u[0]))
        else:
            row["drift"] = float(np.hypot(*v))
        rows.append(row)
    return rows


def stop_gate(row: dict, teacher: dict | None = None) -> dict:
    """One keyframe hold of a take. A stop: every foot down (``STOP_LIFT_MM``) and the body still (``STOP_DRIFT``).
    Travel: the speed along within 20 % of the command (of the teacher's, for a clone), across under 20 % of it. A
    turn: the rate within 20 % of the command's (of the teacher's). Moving at all: the feet lifting (8 mm)."""
    g = {"no_falls": row.get("fall_frac", 1.0) == 0.0}
    if not g["no_falls"]:
        g["pass"] = False
        return g
    c = row["command"]
    sp = math.hypot(c[0], c[1])
    if row["stand"]:
        g["still"] = row["lift_mm"] <= STOP_LIFT_MM and row["drift"] <= STOP_DRIFT
    else:
        if sp > 1e-6:
            ref = teacher["along"] if teacher else sp
            g["speed"] = abs(row["along"] - ref) <= 0.2 * abs(ref)
            g["direction"] = abs(row["cross"]) <= 0.2 * sp
        if abs(c[2]) > 1e-6:
            ref = teacher["yaw_rate"] if teacher else c[2]
            g["yaw"] = abs(row["yaw_rate"] - ref) <= max(0.2 * abs(ref), 0.05)
        g["stepping"] = row["lift_mm"] >= 8.0
    g["pass"] = all(g.values())
    return g


def takes_eval(arm, actor, n_per=2, randomise=False, noisy=False, teacher_rows=None, noise_end=None):
    """Every recorded take, played by ``actor`` (``None``: the firmware open loop): per take, its keyframes' metrics
    and gates (against ``teacher_rows`` for a clone), and ``pass``."""
    S = take_schedules()
    out = play_schedules(list(S.values()), arm, actor, n_per, randomise, noisy, noise_end=noise_end)
    res, ok = {}, True
    for j, (name, kfs) in enumerate(S.items()):
        rows = schedule_metrics(kfs, out, j)
        for i, r in enumerate(rows):
            r["gates"] = stop_gate(r, None if teacher_rows is None else teacher_rows[name][i])
        res[name] = rows
        ok &= all(r["gates"]["pass"] for r in rows)
    return {"takes": res, "pass": bool(ok)}


def print_takes(res: dict) -> None:
    for name, rows in res["takes"].items():
        print(f"== take {name}")
        for r in rows:
            c = r.get("command", [0, 0, 0])
            if "along" in r:
                s = f"along {r['along']:+.3f} cross {r['cross']:+.3f}"
            elif "drift" in r:
                s = f"drift {r['drift']:.3f}{'':12s}"
            else:
                s = f"{'':26s}"
            fails = [x for x, v in r["gates"].items() if not v and x != "pass"]
            print(f"  kf{r['kf']:<2} {'stop' if r.get('stand') else 'move'} cmd {c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f} | "
                  f"{s} yaw {r.get('yaw_rate', 0):+.2f} lift {r.get('lift_mm', 0):4.1f} mm "
                  f"falls {r.get('fall_frac', 1):.2f}  {'ok' if not fails else 'fails ' + ','.join(fails)}")
    print("takes", "PASS" if res["pass"] else "FAIL")


def omni_gate(m: dict, teacher: dict | None = None) -> dict:
    """One command's gates. Travel: the speed along the command within 20 % of it (of the teacher's, for a clone) and
    the speed across it under 20 % of it. Yaw: the rate within 20 % (at least 0.05 rad/s) of a commanded turn (of the
    teacher's, for a clone); without one, the heading kept within 10 deg after the settle (not for a clone). A turn in
    place: the body
    drifting under 0.03 m/s. The fundamental, clearance and pitch as ``gate_table``; the RMS roll under 3 deg."""
    c = m["command"]
    sp = math.hypot(c[0], c[1])
    g = {"no_falls": m.get("fall_frac", 1.0) == 0.0}
    if "yaw_rate" not in m:
        g["pass"] = False
        return g
    if sp > 1e-6:
        ref = teacher["along"] if teacher else sp
        g["speed"] = abs(m["along"] - ref) <= 0.2 * abs(ref)
        g["direction"] = abs(m["cross"]) <= 0.2 * sp
    else:
        g["drift"] = m["drift"] < 0.03
    if abs(c[2]) > 1e-6:
        ref = teacher["yaw_rate"] if teacher else c[2]
        g["yaw"] = abs(m["yaw_rate"] - ref) <= max(0.2 * abs(ref), 0.05)
    elif not teacher:                                   # the open-loop teacher does not hold its heading either
        g["heading"] = m["heading_deg"] <= 10.0
    g["clearance"] = m["clearance_mm"] >= 8.0
    g["pitch"] = m["pitch_deg"] < 3.0
    g["roll"] = m["roll_deg"] < 3.0
    if teacher:
        g["f0"] = abs(m["f0_hz"] - teacher["f0_hz"]) <= 0.2
    else:
        g["f0"] = abs(m["f0_hz"] - FIRMWARE_F0_HZ) <= 0.3
    g["pass"] = all(g.values())
    return g
