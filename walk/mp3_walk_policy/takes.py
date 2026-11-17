"""Keyframe schedules (mpx-recording takes, or drawn at random) played through the firmware's own leg phases.

A take (``mpxrec-*.csv``) lists keyframes ``vx% vy% wz% height_mm clearance_mm ramp_ms hold_ms ease [stand]``: full
stick is 200 mm/s and 2 rad/s, and each keyframe ramps from the previous one over ramp_ms (eased) and holds for hold_ms.
A ``stand`` keyframe is no command: the legs leave the gait (mpx-recording ``gait_task``: ``walk = false``), the feet ease
home from mid-stride over ``RAMP_OUT_MS`` (a quarter sine) and hold the stance, and the next walking keyframe restarts
the gait from that stance (``stanford_gait_reset``), its command ramping up from zero.

``firmware_rel(kfs, steps)`` plays a schedule through the host-built gait (``firmware_gait``) on the firmware's 15 ms
ticks and samples it at the policy's 20 ms steps as ``teacher.py`` does (tick floor(4 n / 3), a zero-order hold): per
step the command the policy observes (vx, vy m/s, wz rad/s; zero while standing) and the teacher's target (rel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import Contract

VX_MAX_MM_S, VY_MAX_MM_S, WZ_MAX_RAD_S = 200.0, 200.0, 2.0
RAMP_OUT_MS = 400
TICK_MS = 15
HEIGHT_MM, CLEARANCE_MM = 70.0, 20.0
SERVOS = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]


def keyframe(vx=0.0, vy=0.0, wz=0.0, ramp_ms=400, hold_ms=2000, stand=False, h=HEIGHT_MM, clr=CLEARANCE_MM,
             smooth=True) -> dict:
    """A keyframe in the take's units: % of full stick."""
    return {"vx": float(vx), "vy": float(vy), "wz": float(wz), "h": float(h), "clr": float(clr),
            "ramp_ms": int(ramp_ms), "hold_ms": int(hold_ms), "smooth": bool(smooth), "stand": bool(stand)}


def read_take(path):
    """The take's header (``trim_deg``, ``height_mm``, ``keyframes``) and its rows (``t`` ms, and per servo ``cmd`` /
    ``act`` ticks, ``deg`` the raw AT32 angle, ``cur`` mA), servos in id order."""
    head, rows, kfs, cols = {}, [], [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            s = line[1:].strip()
            if s.startswith("trim_deg"):
                head["trim_deg"] = np.array([float(v) for v in s.split()[1:]])
            elif s.startswith("stance height_mm"):
                head["height_mm"] = float(s.split()[1].split("=")[1])
            elif s[:2] == "kf" and s[2].isdigit():
                p = s.split()
                kfs.append(keyframe(p[1], p[2], p[3], h=float(p[4]), clr=float(p[5]), ramp_ms=int(p[6]),
                                    hold_ms=int(p[7]), smooth=p[8] == "smooth", stand=len(p) > 9 and p[9] == "1"))
            continue
        if line.startswith("t_ms"):
            cols = line.split(",")
            continue
        rows.append([float(v) for v in line.split(",")])
    a = np.array(rows)
    idx = {c: i for i, c in enumerate(cols)}
    take = {"t": a[:, 0].astype(int)}
    for f in ("cmd", "act", "deg", "cur"):
        take[f] = np.stack([a[:, idx[f"{s}_{f}"]] for s in SERVOS], axis=1)
    head["keyframes"] = kfs
    return head, take


def _ease(f, smooth):
    f = min(max(f, 0.0), 1.0)
    return f * f * (3.0 - 2.0 * f) if smooth else f


def schedule(kfs, el, height=HEIGHT_MM):
    """The motion (vx%, vy%, wz%, h, clr, stand) at ``el`` ms after the first walking tick, or None after the last
    keyframe."""
    prev = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "h": height, "clr": kfs[0]["clr"]}
    for k in kfs:
        n = k["ramp_ms"] + k["hold_ms"]
        if el < n:
            if k["stand"]:
                return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "h": k["h"], "clr": k["clr"], "stand": True}
            f = _ease(el / k["ramp_ms"], k["smooth"]) if k["ramp_ms"] else 1.0
            return {**{c: prev[c] + (k[c] - prev[c]) * f for c in prev}, "stand": False}
        el -= n
        prev = {c: k[c] for c in prev}
    return None


def duration_ms(kfs) -> int:
    return int(sum(k["ramp_ms"] + k["hold_ms"] for k in kfs))


class FirmwareLegs:
    """gait_task's leg phases (walk / ramp_out / stand) around the host gait, one 15 ms tick per ``tick``."""

    def __init__(self, gait, height=HEIGHT_MM):
        self.g = gait
        self.height = height
        self.walking = False
        self.feet = np.zeros((4, 3))
        self.feet[:, 2] = height
        self.ramp_t0, self.ramp_feet = 0, self.feet.copy()

    def tick(self, t_ms, m):
        """The feet at ``t_ms`` for motion ``m`` (``schedule``'s, or None: stand)."""
        if m is None or m["stand"]:
            h = self.height if m is None else m["h"]
            if self.walking:
                self.walking, self.ramp_t0, self.ramp_feet = False, t_ms, self.feet.copy()
            f = np.sin(min(t_ms - self.ramp_t0, RAMP_OUT_MS) * np.pi / 2 / RAMP_OUT_MS)
            feet = self.ramp_feet.copy()
            feet[:, :2] *= 1.0 - f
            feet[:, 2] += (h - self.ramp_feet[:, 2]) * f
        else:
            if not self.walking:
                self.g.reset(self.feet[0, 2])
                self.walking = True
            feet = self.g.step(m["vx"] * VX_MAX_MM_S / 100, m["vy"] * VY_MAX_MM_S / 100, m["wz"] * WZ_MAX_RAD_S / 100,
                               m["h"], m["clr"])
        self.feet = feet
        return feet


def command_of(m) -> np.ndarray:
    """The policy's command (vx, vy m/s, wz rad/s) for a motion; zero while standing."""
    if m is None or m["stand"]:
        return np.zeros(3)
    return np.array([m["vx"] * VX_MAX_MM_S / 1e5, m["vy"] * VY_MAX_MM_S / 1e5, m["wz"] * WZ_MAX_RAD_S / 100])


def schedule_commands(kfs, steps: int) -> np.ndarray:
    """``firmware_rel``'s commands alone (steps, 3), vectorised: the motion of tick floor(4 n / 3) at each step."""
    t = ((4 * np.arange(steps)) // 3) * TICK_MS
    out = np.zeros((steps, 3))
    prev = np.zeros(3)
    t0 = 0
    for k in kfs:
        tgt = command_of({**k, "stand": False})
        n = k["ramp_ms"] + k["hold_ms"]
        sel = (t >= t0) & (t < t0 + n)
        if k["stand"]:
            out[sel] = 0.0
            prev = np.zeros(3)
        else:
            f = np.clip((t[sel] - t0) / k["ramp_ms"], 0.0, 1.0) if k["ramp_ms"] else np.ones(sel.sum())
            if k["smooth"]:
                f = f * f * (3.0 - 2.0 * f)
            out[sel] = prev + (tgt - prev) * f[:, None]
            prev = tgt
        t0 += n
    return out


# Random take-like schedules: what the clone copies and what PPO is commanded. Walking at |(vx, vy)| in SPEED_RANGE
# at a uniform angle (half of them also turning, |wz| <= WALK_WZ_MAX), turning in place (|wz| in SPIN_RANGE, the
# takes' 60 % stick is 1.2 rad/s), and stops (stand keyframes), each ramped in over RAMP_MS (smooth, as the takes).
SPEED_RANGE = (0.05, 0.12)
WALK_WZ_MAX = 0.6
SPIN_RANGE = (0.2, 1.2)
REL_TURNING = 0.5
P_SPIN, P_STAND, P_START_STANDING = 0.15, 0.35, 0.3
WALK_HOLD_S, SPIN_HOLD_S, STAND_HOLD_S, START_STAND_S = (1.0, 3.5), (0.5, 2.5), (0.4, 2.0), (0.2, 1.5)
RAMP_MS = (200, 600)


def random_schedule(rng: np.random.Generator, seconds: float = 10.0) -> list:
    """A keyframe list covering at least ``seconds``: walks, turns in place and stops (never two stops in a row)."""
    kfs, total, standing = [], 0, False
    if rng.random() < P_START_STANDING:
        kfs.append(keyframe(ramp_ms=0, hold_ms=int(1000 * rng.uniform(*START_STAND_S)), stand=True))
        total, standing = kfs[-1]["hold_ms"], True
    while total < 1000 * seconds:
        ramp = int(rng.uniform(*RAMP_MS))
        if not standing and kfs and rng.random() < P_STAND:
            k = keyframe(ramp_ms=ramp, hold_ms=int(1000 * rng.uniform(*STAND_HOLD_S)), stand=True)
        elif rng.random() < P_SPIN:
            wz = rng.choice((-1.0, 1.0)) * rng.uniform(*SPIN_RANGE)
            k = keyframe(wz=wz * 100 / WZ_MAX_RAD_S, ramp_ms=ramp, hold_ms=int(1000 * rng.uniform(*SPIN_HOLD_S)))
        else:
            sp, th = rng.uniform(*SPEED_RANGE), rng.uniform(-np.pi, np.pi)
            wz = rng.uniform(-WALK_WZ_MAX, WALK_WZ_MAX) if rng.random() < REL_TURNING else 0.0
            k = keyframe(vx=sp * np.cos(th) * 1e5 / VX_MAX_MM_S, vy=sp * np.sin(th) * 1e5 / VY_MAX_MM_S,
                         wz=wz * 100 / WZ_MAX_RAD_S, ramp_ms=ramp, hold_ms=int(1000 * rng.uniform(*WALK_HOLD_S)))
        kfs.append(k)
        standing = k["stand"]
        total += k["ramp_ms"] + k["hold_ms"]
    return kfs


def schedule_bank(n: int, steps: int, seed: int = 0, with_rel: bool = False):
    """``n`` random schedules of ``steps`` policy steps: the commands (n, steps, 3) and, ``with_rel``, the firmware's
    targets (n, steps, 12)."""
    rng = np.random.default_rng(seed)
    cmd = np.zeros((n, steps, 3), dtype=np.float32)
    rel = np.zeros((n, steps, 12), dtype=np.float32) if with_rel else None
    if with_rel:
        from . import firmware_gait as fg
        c = Contract.load()
        g = fg.Gait(c.offsets_deg)
    for i in range(n):
        kfs = random_schedule(rng, steps * 0.020)
        if with_rel:
            cmd[i], rel[i] = firmware_rel(kfs, steps, c, g)
        else:
            cmd[i] = schedule_commands(kfs, steps)
    return cmd, rel


def firmware_rel(kfs, steps: int, contract: Contract | None = None, gait=None):
    """Per policy step (20 ms) from the first walking tick: the command (steps, 3) and the firmware's target as the
    policy's rel (steps, 12), with SKU2's calibration (the contract's offsets). After the last keyframe: the stand."""
    from . import firmware_gait as fg
    c = contract or Contract.load()
    g = gait or fg.Gait(c.offsets_deg)
    legs = FirmwareLegs(g)
    n_ticks = (4 * (steps - 1)) // 3 + 1
    cmd_t = np.zeros((n_ticks, 3))
    ticks = np.zeros((n_ticks, 12), dtype=int)
    for j in range(n_ticks):
        m = schedule(kfs, j * TICK_MS)
        cmd_t[j] = command_of(m)
        ticks[j] = g.ticks(legs.tick(j * TICK_MS, m))
    pick = (4 * np.arange(steps)) // 3
    return cmd_t[pick], c.rel_from_ticks(ticks[pick])
