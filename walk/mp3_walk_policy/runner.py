"""The robot side: run an exported policy on SKU2 from its Raspberry Pi 5, through mp3-link (numpy and mp3link only).

On the Pi (nothing installed; ``~/walk`` holds this package's contract, export and runner, its data and the policies):

    cd ~/walk && PYTHONPATH=~/walk:~/mp3link_calib python3 -m mp3_walk_policy.runner <command> ...

Commands:
- ``check --policy DIR``: no port. The policy's fixtures under this numpy, its contract hash, its input width against
  its meta's layout, the inference time.
- ``imu --seconds S``: the servos stay idle. Prints the tilt and the gravity direction in each IMU-mounting candidate;
  logs the STATE stream (the tilt test).
- ``replay --policy DIR --actions FILE``: plays a recorded 50 Hz raw-action stream open loop through the same decoder.
- ``run --policy DIR --vx V [--vy V --wz W] --seconds S [--mount M] [--dry]``: the policy closed loop (``--vy``,
  ``--wz``: omni policies only). ``--dry`` never arms (every frame has enable 0, no ARM): real STATE, IMU and timing,
  the servos idle. A policy with the heading term (meta ``obs_terms_extra``) also observes the heading loop's error
  (``contract.heading_obs``; 0 without ``--heading-loop``).
- ``gait --vx V [--vy V --wz W] [--heading-loop] --seconds S``: the firmware trot, minipupperesp's own Stanford
  trot and IK (``firmware_gait``: its C sources compiled unchanged), stepped live every 15 ms (3 STATE ticks, the
  firmware's tick) from the command, with the same heading loop, safety, stops and logs as ``run``. The command can
  change during the walk: ``<log>.cmd`` (``vx vy wz`` in m/s and rad/s; ``tools/pi_take.sh --cmd``) is read every
  0.1 s.

The loop implements ``contract.py`` as the simulator does (``mdp.py``): one policy step per 4 STATE ticks (20 ms,
locked to the ESP32's clock); every STATE frame into the velocity filter in tick order (a missing tick repeats the
last sample, and is counted); the observation from the newest frame (positions from its dd; the gyro in rad/s, its
bias removed; gravity = -acc/|acc|; both rotated from the sensor into the body frame by the mounting); the policy;
the decoder (clip, scale, band, the 20 deg step limit, firmware ticks); one SET_ALL.

Safety (every arming command): it refuses to start over a latched fault, without the identified servo configuration
on all twelve servos, without a valid IMU, or above 45 degC; it warns if the robot's calibration offsets differ from
the contract's. During motion: a low-passed tilt (0.1 s) above 35 deg disarms at once (the body drops); an invalid IMU
frame (the flag cleared, or the firmware's zeros), a non-finite action, a STATE gap, two late steps in a row, a servo
at 55 degC or Enter ramp to the stand, hold, and disarm; Ctrl-C or a link fault disarms at once.

Logs: ``<log>.npz`` (mp3link's ``save_log``: every STATE and SET_ALL), ``<log>.policy.npz`` (per step: the STATE tick
used, times, observation, raw action, applied target, dd) and ``<log>.stand.json`` (the stand and the robot's
offsets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import select
import sys
import time
from pathlib import Path

import numpy as np

from .contract import (CONTRACT_JSON, CONTROL_DT, ActionDecoder, Contract, HistoryStacker, VelocityFilter,
                       assemble_observation, has_heading_obs, heading_obs, obs_size)
from .export import NumpyPolicy
from .heading import ANTIWINDUP, I_MAX, KI, KP, WZ_MAX, pi_step

TICKS_PER_STEP = int(round(CONTROL_DT / 0.005))          # 4 STATE ticks per policy step
DEG = math.pi / 180.0
TILT_LIMIT_DEG = 35.0
TILT_TAU_S = 0.1               # the tilt that disarms is low-passed: single frames read up to 176 deg under vibration
TEMP_STOP_C = 55.0
TEMP_START_C = 45.0
ACC_DROPOUT_MG = 1.0          # the firmware sends exact zeros without a fresh sample; real frames reach 25 mg under vibration
CAP_MA = 2000                  # the frame cap (mode 1: duty <= min(max_pwm, kff x cap)); SKU2's recordings used 2000
GAIT_DT = 0.015                # the firmware trot's tick (minipupperesp stanford_gait.h SG_DT)
GAIT_TICKS = int(round(GAIT_DT / 0.005))    # 3 STATE ticks per gait tick
CMD_POLL_STEPS = 7             # the command file is read every 7 gait ticks (0.105 s)
GAIT_VX_MAX, GAIT_WZ_MAX = 0.20, 1.0        # m/s (the firmware's full stick), rad/s: a command beyond is clipped
IDENTIFIED = {"control_mode": 1.0, "kp_position": 80.0, "kd_position": 400.0, "kff_current": 0.001,
              "max_pwm_duty_cycle": 0.99}

# The IMU's mounting, sensor -> body. Only the z flip is known (the accelerometer reads -1 g on its z upright); the
# four proper rotations with z flipped are the candidates the tilt test decides between.
IMU_MOUNTS = {
    "flip_x": np.diag([1.0, -1.0, -1.0]),
    "flip_y": np.diag([-1.0, 1.0, -1.0]),
    "swap_xy": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    "swap_xy_neg": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}
IMU_MOUNT_JSON = Path(__file__).resolve().parent / "data" / "imu_mount.json"


class StopRun(Exception):
    """A graceful stop: ramp to the stand, hold, disarm."""


class Disarm(Exception):
    """An immediate stop: disarm now."""


def sha256(path) -> str:
    """As ``paths.sha256``: a JSON file is hashed with LF line endings (a Windows checkout's copy matches)."""
    b = Path(path).read_bytes()
    if Path(path).suffix.lower() == ".json":
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def load_mount(name: str | None) -> tuple[str, np.ndarray]:
    if name is None:
        if not IMU_MOUNT_JSON.exists():
            raise SystemExit("no measured IMU mounting (data/imu_mount.json): run the tilt test, or pass --mount")
        name = json.loads(IMU_MOUNT_JSON.read_text())["mount"]
    if name not in IMU_MOUNTS:
        raise SystemExit(f"unknown mount {name!r}; candidates: {sorted(IMU_MOUNTS)}")
    return name, IMU_MOUNTS[name]


# ------------------------------------------------------------------------------------------------ the observation
def tilt_deg(acc_mg) -> float:
    """The body's tilt from upright. Needs only the known z flip: body z = -sensor z for every candidate."""
    a = np.asarray(acc_mg, dtype=float)
    n = np.linalg.norm(a)
    if n <= 0.0:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, -a[2] / n))))


def imu_valid(st, imu_ok_flag: int) -> bool:
    """The IMU flag set and not the firmware's dropout zeros. The magnitude itself is no test: on SKU2's floor trot
    single frames read 0.35-2.1 g (1st-99th percentile), 7 % of them outside 0.5-1.5 g, in runs of up to 11 frames."""
    n = float(np.linalg.norm(np.asarray(st.acc_mg, dtype=float)))
    return bool(st.flags & imu_ok_flag) and n > ACC_DROPOUT_MG


def read_command_file(path: Path, command, mtime, out=print):
    """``vx vy wz`` (m/s, rad/s) from ``path`` when it changed since ``mtime``; else the same command. A malformed
    file keeps the command (and says so)."""
    try:
        m = path.stat().st_mtime_ns
    except FileNotFoundError:
        return command, mtime
    if m == mtime:
        return command, mtime
    try:
        v = [float(x) for x in path.read_text().split()]
        if len(v) != 3 or not all(math.isfinite(x) for x in v):
            raise ValueError(f"need three numbers, got {v}")
    except (OSError, ValueError) as exc:
        out(f"command file ignored: {exc}")
        return command, m
    new = (float(np.clip(v[0], -GAIT_VX_MAX, GAIT_VX_MAX)), float(np.clip(v[1], -GAIT_VX_MAX, GAIT_VX_MAX)),
           float(np.clip(v[2], -GAIT_WZ_MAX, GAIT_WZ_MAX)))
    out(f"command vx {new[0]:+.3f} m/s, vy {new[1]:+.3f} m/s, wz {new[2]:+.2f} rad/s")
    return new, m


class TiltFilter:
    """The tilt from a first-order low-pass of the accelerometer (time constant TILT_TAU_S, fed every STATE frame).
    On SKU2's logs it peaks at 8.2 deg on the floor trot and 32.8 deg when the front was lifted by hand."""

    def __init__(self, tau_s: float = TILT_TAU_S, dt: float = 0.005):
        self.k = dt / tau_s
        self.acc = None

    def push(self, acc_mg) -> float:
        a = np.asarray(acc_mg, dtype=float)
        self.acc = a.copy() if self.acc is None else self.acc + self.k * (a - self.acc)
        return tilt_deg(self.acc)


class ObservationBuilder:
    """STATE frames -> the policy's observation, as ``mdp.py`` builds it in simulation."""

    def __init__(self, contract: Contract, mount: np.ndarray, gyro_bias_rad_s=(0.0, 0.0, 0.0)):
        self.c = contract
        self.R = np.asarray(mount, dtype=float)
        self.bias = np.asarray(gyro_bias_rad_s, dtype=float)       # sensor frame
        self.vf = VelocityFilter()
        self.last_tick: int | None = None
        self.missing = 0
        self.latest = None

    def reset(self) -> None:
        self.vf.reset()
        self.last_tick = None
        self.latest = None

    def push(self, st) -> None:
        """Every STATE frame, in order. A missing tick repeats the previous sample (the simulator never misses)."""
        rel = self.c.rel_from_dd(np.asarray(st.pos_dd, dtype=float))
        if self.last_tick is not None:
            gap = (int(st.tick) - self.last_tick - 1) & 0xFFFFFFFF
            if 0 < gap < 1000:
                self.missing += gap
                for _ in range(min(gap, 6)):
                    self.vf.push(self.vf.hist[-1])
        self.vf.push(rel)
        self.last_tick = int(st.tick)
        self.latest = st

    def gyro_body(self, st) -> np.ndarray:
        g = np.asarray(st.gyr_ddps, dtype=float) * 0.1 * DEG - self.bias
        return self.R @ g

    def gravity_body(self, st) -> np.ndarray:
        a = self.R @ np.asarray(st.acc_mg, dtype=float)
        return -a / np.linalg.norm(a)

    def observe(self, command, prev_action, heading_error=None) -> np.ndarray:
        """``command``: vx, or (vx, vy, wz) for an omni policy. ``heading_error``: the observed heading term
        (``contract.heading_obs``) for a policy that has it."""
        st = self.latest
        pos = self.c.rel_from_dd(np.asarray(st.pos_dd, dtype=float))
        return assemble_observation(self.gyro_body(st), self.gravity_body(st), command, pos, self.vf.velocity(),
                                    prev_action, heading_error)


class HeadingLoop:
    """The outer heading loop for policies that observe a yaw-rate command: the heading is the gyro's rate about
    gravity, integrated over every STATE frame (gravity low-passed, the accelerometer being vibration-dominated while
    walking; valid on a slope); each policy step the error to the heading at the start becomes a yaw-rate command
    (PI, clipped; the law is ``heading.pi_step``, shared with the evaluation tools), added to the commanded wz. A
    policy with the heading term (``contract.has_heading_obs``) also observes ``error()``, the error the loop used."""

    def __init__(self, kp: float = KP, ki: float = KI, wz_max: float = WZ_MAX, i_max: float = I_MAX,
                 tau_g_s: float = 1.0, frame_dt: float = 0.005, antiwindup: str = "cond"):
        self.kp, self.ki, self.wz_max, self.i_max, self.antiwindup = kp, ki, wz_max, i_max, antiwindup
        self.tau_g_s = tau_g_s
        self.alpha = frame_dt / tau_g_s
        self.dt = frame_dt
        self.reset()

    def reset(self) -> None:
        self.psi, self.integ, self.g_lp = 0.0, 0.0, None
        self.err = 0.0

    def push(self, gyro_body, gravity_body) -> None:
        g = np.asarray(gravity_body, dtype=float)
        self.g_lp = g.copy() if self.g_lp is None else self.g_lp + self.alpha * (g - self.g_lp)
        up = -self.g_lp / max(np.linalg.norm(self.g_lp), 1e-9)
        self.psi += float(np.dot(np.asarray(gyro_body, dtype=float), up)) * self.dt

    def error(self) -> float:
        """The heading error (rad, + = turn left to correct): back to the starting heading, wrapped to [-pi, pi)."""
        return float(-((self.psi + np.pi) % (2 * np.pi) - np.pi))

    def wz(self, dt: float = CONTROL_DT) -> float:
        self.err = self.error()
        wz, integ = pi_step(self.err, self.integ, dt, self.kp, self.ki, self.wz_max, self.i_max, self.antiwindup)
        self.integ = float(integ)
        return float(wz)

    def describe(self) -> dict:
        """What the take's meta records, so a log says which loop ran."""
        return {"kp": self.kp, "ki": self.ki, "wz_max": self.wz_max, "i_max": self.i_max, "tau_g_s": self.tau_g_s,
                "antiwindup": self.antiwindup}


def heading_from_args(a) -> HeadingLoop:
    return HeadingLoop(kp=a.heading_kp, ki=a.heading_ki, wz_max=a.heading_wz_max, antiwindup=a.heading_aw)


# --------------------------------------------------------------------------------------------------- the runner
class Runner:
    def __init__(self, link, contract: Contract, policy: NumpyPolicy | None, mount: np.ndarray, *, dry: bool = False,
                 cap: int = CAP_MA, stdin=None, out=print, stop_file: Path | None = None):
        from mp3link import protocol as p
        self.p = p
        self.link, self.c, self.policy, self.dry, self.cap = link, contract, policy, dry, cap
        self.mount = mount
        self.stdin = stdin
        self.stop_file = stop_file                        # its appearance is a graceful stop, as Enter
        self.out = out
        self.stand_dd = np.asarray(contract.stand_dd, dtype=int)
        self.armed = False
        self.tilt_limit = TILT_LIMIT_DEG                  # 0: no tilt cut (--tilt-cut 0)
        self.clear = False                                # --clear: the first arming frame also clears a latched fault
        self.last_dd = None
        self.steps: dict[str, list] = {k: [] for k in ("tick", "t_rx", "t_send", "seq", "obs", "raw", "rel", "dd",
                                                       "imu_valid", "tilt_deg", "late", "heading_psi", "heading_integ",
                                                       "heading_err", "wz_cmd")}
        self.stop_reason = ""
        self.warnings: list[str] = []
        self.gyro_bias = np.zeros(3)
        self.robot_offsets = None
        self.tilt = TiltFilter()
        self._tilt_tick = None

    # ---- checks ---------------------------------------------------------------------------------------------------
    def preflight(self, clear: bool = False) -> None:
        from mp3link.cli import preflight, present, read_offsets
        st = self.link.wait_state(1.0)
        preflight(self.link, clear)
        self.clear = clear
        present(self.link)                                          # four fresh boards, plausible positions
        if not st.flags & self.p.ST_IMU_OK:
            raise SystemExit("IMU absent: power-cycle the board and check ping before a policy run")
        if not self.dry:
            for i in range(1, 13):
                for name, want in IDENTIFIED.items():
                    got = self.link.get_param(i, name)
                    if abs(got - want) > 1e-6 + 1e-3 * abs(want):
                        raise SystemExit(f"servo {i} {name} = {got:g}, not {want:g}: run `mp3link config` first")
        offs, stored = read_offsets(self.link)
        self.robot_offsets = offs
        if not all(stored) or np.any(np.abs(offs - self.c.offsets_deg) > 1e-3):
            msg = (f"WARNING: the robot's calibration offsets {np.round(offs, 2).tolist()} differ from the contract's "
                   f"{np.round(self.c.offsets_deg, 2).tolist()}; the policy runs on the contract's stand")
            self.warnings.append(msg)
            self.out(msg)
        temps = np.asarray(st.temp_dc, dtype=float) / 10.0
        if temps.max() >= TEMP_START_C:
            raise SystemExit(f"a servo is at {temps.max():.1f} degC: wait below {TEMP_START_C:.0f} degC")

    def measure_gyro_bias(self, seconds: float = 1.0) -> np.ndarray:
        """The gyro's mean over ``seconds`` with the servos idle (sensor frame, rad/s)."""
        end = time.monotonic() + seconds
        st, g = None, []
        while time.monotonic() < end:
            st = self.link.wait_state(0.1, after=st)
            if imu_valid(st, self.p.ST_IMU_OK):
                g.append(np.asarray(st.gyr_ddps, dtype=float) * 0.1 * DEG)
        if len(g) < 0.5 * seconds / 0.005:
            raise SystemExit("too few valid IMU frames to measure the gyro bias")
        self.gyro_bias = np.mean(g, axis=0)
        return self.gyro_bias

    # ---- sending --------------------------------------------------------------------------------------------------
    def send(self, dd) -> int:
        dd = np.asarray(dd)
        if not np.all(np.isfinite(dd)):
            raise StopRun("non-finite target")
        dd = np.clip(np.rint(dd), 0, 3100).astype(int)
        if self.dry:
            seq = self.link.set_all(dd, enable=0, flags=0, cur_cap_ma=0)
        else:
            flags = 0 if self.armed else self.p.SA_ARM | (self.p.SA_CLEAR if self.clear else 0)
            seq = self.link.set_all(dd, enable=0x0FFF, flags=flags, cur_cap_ma=self.cap)
            self.armed = True
        self.last_dd = dd
        return seq

    def watch(self) -> None:
        """Between sends: link faults and temperatures."""
        from mp3link.cli import check_flags
        try:
            self.link.check()
            if not self.dry:
                st = self.link.latest
                if st.flags & (self.p.ST_FAULTS | self.p.ST_WATCHDOG | self.p.ST_REJECTED):
                    check_flags(self.link)
        except Exception as exc:
            raise Disarm(f"link: {exc}") from exc
        st = self.link.latest
        if not self.dry:
            if not imu_valid(st, self.p.ST_IMU_OK):
                raise StopRun("IMU frame invalid")
            self.check_tilt(st)
        temps = np.asarray(st.temp_dc, dtype=float) / 10.0
        if temps.max() >= TEMP_STOP_C:
            raise StopRun(f"servo at {temps.max():.1f} degC")
        if self.enter_pressed():
            raise StopRun("Enter")
        if self.stop_file is not None and self.stop_file.exists():
            raise StopRun("stop file")

    def check_tilt(self, st) -> None:
        if self._tilt_tick is not None and ((int(st.tick) - self._tilt_tick) & 0xFFFFFFFF) - 1 >= 0x7FFFFFFF:
            return                                        # a frame already fed (not newer than the last)
        self._tilt_tick = int(st.tick)
        t = self.tilt.push(st.acc_mg)
        if self.tilt_limit and t > self.tilt_limit and not self.dry:
            raise Disarm(f"tilt {t:.0f} deg")

    def enter_pressed(self) -> bool:
        if self.stdin is None:
            return False
        r, _, _ = select.select([self.stdin], [], [], 0)
        if r:
            self.stdin.readline()
            return True
        return False

    def ramp(self, to_dd, seconds: float, frm=None, stoppable: bool = True) -> None:
        from mp3link.cli import Pacer, smoothstep
        frm = np.asarray(self.last_dd if frm is None else frm, dtype=float)
        to = np.asarray(to_dd, dtype=float)
        pacer = Pacer(CONTROL_DT)
        n = max(1, int(round(seconds / CONTROL_DT)))
        for k in range(n):
            self.send(frm + (to - frm) * smoothstep((k + 1) / n))
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def hold(self, dd, seconds: float, stoppable: bool = True) -> None:
        from mp3link.cli import Pacer
        pacer = Pacer(CONTROL_DT)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(dd)
            pacer.wait()
            try:
                self.watch()
            except StopRun:
                if stoppable:
                    raise

    def safe_end(self, hold_after: float) -> None:
        """Ramp to the stand, hold, disarm (a graceful stop)."""
        try:
            if self.armed and self.last_dd is not None:
                self.ramp(self.stand_dd, 1.0, stoppable=False)         # already stopping: finish the way down
                self.hold(self.stand_dd, hold_after, stoppable=False)
        except Disarm as exc:
            self.stop_reason += f"; {exc}"
        finally:
            self.link.disarm()
            self.armed = False

    # ---- the loop -------------------------------------------------------------------------------------------------
    def loop(self, act_fn, command, seconds: float, ob: ObservationBuilder, decoder: ActionDecoder,
             heading: HeadingLoop | None = None, heading_term: bool = False) -> None:
        """Policy steps locked to the STATE ticks; ``act_fn(obs, k)`` returns the raw action. With ``heading``, the
        command's wz gets the heading loop's correction every step. ``heading_term``: the observation ends with the
        loop's error as observed (``contract.heading_obs``; 0 without a loop, as in training's other envs)."""
        link = self.link
        rec = link.record
        idx = len(rec) if rec is not None else 0
        st = link.wait_state(0.05)
        ob.reset()
        decoder.reset()
        ob.push(st)
        if heading is not None:
            heading.reset()
        next_tick = int(st.tick) + TICKS_PER_STEP
        late_run = 0
        n_steps = int(round(seconds / CONTROL_DT))
        for k in range(n_steps):
            while True:                       # every frame up to the step's tick, in order
                try:
                    st = link.wait_state(0.025, after=st)
                except Exception as exc:
                    raise StopRun(f"STATE gap: {exc}") from exc
                new = rec[idx:] if rec is not None else [st]
                idx = len(rec) if rec is not None else idx
                for s in new:
                    if ob.last_tick is None or ((int(s.tick) - ob.last_tick) & 0xFFFFFFFF) < 1000:
                        if int(s.tick) != ob.last_tick:
                            ob.push(s)
                            if heading is not None and imu_valid(s, self.p.ST_IMU_OK):
                                heading.push(ob.gyro_body(s), ob.gravity_body(s))
                    if not imu_valid(s, self.p.ST_IMU_OK) and not self.dry:
                        raise StopRun("IMU frame invalid")
                    self.check_tilt(s)
                if ((int(st.tick) - next_tick) & 0xFFFFFFFF) < 0x80000000:
                    break
            late = (int(st.tick) - next_tick) & 0xFFFFFFFF
            late_run = late_run + 1 if late > 0 else 0
            if late_run >= 2:
                raise StopRun("two late steps in a row")
            next_tick = int(st.tick) + TICKS_PER_STEP
            cmd = command
            if heading is not None:
                cmd = (command[0], command[1], command[2] + heading.wz())
                self.steps["heading_psi"].append(heading.psi)
                self.steps["heading_integ"].append(heading.integ)
                self.steps["heading_err"].append(heading.err)
                self.steps["wz_cmd"].append(cmd[2])
            he = None
            if heading_term:
                he = heading_obs(heading.err) if heading is not None else 0.0
            obs = ob.observe(cmd, decoder.prev_action, he)
            raw = np.asarray(act_fn(obs, k), dtype=float)
            if not np.all(np.isfinite(raw)):
                raise StopRun("non-finite action")
            d = decoder(raw)
            t_send = time.monotonic()
            seq = self.send(d["dd"])
            self.watch()
            for key, val in (("tick", int(st.tick)), ("t_rx", st.t_rx), ("t_send", t_send), ("seq", seq),
                             ("obs", obs), ("raw", raw), ("rel", d["rel"]), ("dd", d["dd"]),
                             ("imu_valid", imu_valid(st, self.p.ST_IMU_OK)), ("tilt_deg", tilt_deg(st.acc_mg)),
                             ("late", late)):
                self.steps[key].append(val)

    def gait_loop(self, gait, command, seconds: float, ob: ObservationBuilder, heading: HeadingLoop | None = None,
                  cmd_file: Path | None = None) -> None:
        """The firmware trot locked to the STATE ticks: one gait tick per ``GAIT_TICKS`` frames (15 ms, the firmware's
        SG_DT), its feet through minipupperesp's IK and tick formula, then the contract's dd, clipped to the band
        about the stand. ``command`` is (vx, vy, wz) in m/s and rad/s; with ``heading`` the loop adds its wz; a
        ``cmd_file`` replaces the command when it changes."""
        from mp3link.cli import clip_band
        link = self.link
        rec = link.record
        idx = len(rec) if rec is not None else 0
        st = link.wait_state(0.05)
        ob.reset()
        ob.push(st)
        if heading is not None:
            heading.reset()
        gait.reset()
        command = tuple(float(v) for v in command)
        cmd_mtime, cmd_poll = None, 0
        next_tick = int(st.tick) + GAIT_TICKS
        late_run = 0
        for k in range(int(round(seconds / GAIT_DT))):
            while True:
                try:
                    st = link.wait_state(0.025, after=st)
                except Exception as exc:
                    raise StopRun(f"STATE gap: {exc}") from exc
                new = rec[idx:] if rec is not None else [st]
                idx = len(rec) if rec is not None else idx
                for s in new:
                    if ob.last_tick is None or ((int(s.tick) - ob.last_tick) & 0xFFFFFFFF) < 1000:
                        if int(s.tick) != ob.last_tick:
                            ob.push(s)
                            if heading is not None and imu_valid(s, self.p.ST_IMU_OK):
                                heading.push(ob.gyro_body(s), ob.gravity_body(s))
                    if not imu_valid(s, self.p.ST_IMU_OK) and not self.dry:
                        raise StopRun("IMU frame invalid")
                    self.check_tilt(s)
                if ((int(st.tick) - next_tick) & 0xFFFFFFFF) < 0x80000000:
                    break
            late = (int(st.tick) - next_tick) & 0xFFFFFFFF
            late_run = late_run + 1 if late > 0 else 0
            if late_run >= 2:
                raise StopRun("two late steps in a row")
            next_tick = int(st.tick) + GAIT_TICKS
            if cmd_file is not None and k >= cmd_poll:
                cmd_poll = k + CMD_POLL_STEPS
                command, cmd_mtime = read_command_file(cmd_file, command, cmd_mtime, self.out)
            vx, vy, wz = command
            if heading is not None:
                wz = wz + heading.wz(GAIT_DT)
                self.steps["heading_psi"].append(heading.psi)
                self.steps["heading_integ"].append(heading.integ)
            self.steps["wz_cmd"].append(wz)
            feet = gait.step(vx * 1000.0, vy * 1000.0, wz)
            dd = self.c.dd_from_ticks(gait.ticks(feet)).astype(float)
            safe = clip_band(self.stand_dd.astype(float), dd)
            t_send = time.monotonic()
            seq = self.send(safe)
            self.watch()
            for key, val in (("tick", int(st.tick)), ("t_rx", st.t_rx), ("t_send", t_send), ("seq", seq),
                             ("rel", self.c.rel_from_dd(safe)), ("dd", safe), ("command", (vx, vy, wz)),
                             ("clipped", bool(np.any(safe != dd))), ("imu_valid", imu_valid(st, self.p.ST_IMU_OK)),
                             ("tilt_deg", tilt_deg(st.acc_mg)), ("late", late)):
                self.steps.setdefault(key, []).append(val)

    def save(self, log: str, meta: dict) -> None:
        from mp3link.cli import save_log
        base = Path(log)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(meta, stop_reason=self.stop_reason, warnings=self.warnings, gyro_bias_rad_s=self.gyro_bias.tolist(),
                    stand=self.stand_dd.tolist(), cap=self.cap, dry=self.dry)
        save_log(self.link, str(base.with_suffix(".npz")), None, meta)
        arrays = {k: np.asarray(v) for k, v in self.steps.items()}
        np.savez_compressed(base.with_suffix(".policy.npz"), meta=json.dumps(meta), **arrays)
        offs = self.robot_offsets if self.robot_offsets is not None else self.c.offsets_deg
        base.with_suffix(".stand.json").write_text(json.dumps(
            {"stand_dd": self.stand_dd.tolist(), "stand_ticks": np.asarray(self.c.stand_ticks).tolist(),
             "offsets_deg": np.asarray(offs, dtype=float).tolist(),
             "source": "mp3_walk_policy contract (sku2_contract.json); offsets read from the robot"}, indent=1))


# ---------------------------------------------------------------------------------------------------- commands
def cmd_check(a) -> int:
    pol = NumpyPolicy(Path(a.policy) / "policy.npz")
    d = np.load(Path(a.policy) / "policy.npz")
    err = float(np.abs(pol(d["fixture_obs"]) - d["fixture_act"]).max())
    want = pol.meta.get("contract_sha256")
    have = sha256(CONTRACT_JSON)
    # a recalibration changes only the offsets and the stand in the contract; it lists the files it replaced
    recal = [c for c in json.loads(CONTRACT_JSON.read_text()).get("calibrations_superseded", []) if c["sha256"] == want]
    # the input layout the meta names (history, the heading term) against the first layer's width
    he, hist = has_heading_obs(pol.meta), int(pol.meta.get("history", 0) or 0)
    want_in = obs_size(he) * max(hist, 1)
    have_in = int(pol.layers[0][0].shape[1])
    layout_ok = pol.meta.get("privileged", False) or have_in == want_in
    x = d["fixture_obs"][0]
    t0 = time.perf_counter()
    for _ in range(1000):
        pol(x)
    us = (time.perf_counter() - t0) * 1e3
    print(f"fixtures: max |numpy - torch| {err:.2e} ({'ok' if err < 1e-5 else 'FAIL'})")
    same = want == have or bool(recal)
    print(f"contract: {'ok' if want == have else 'ok (exported with an earlier calibration; only the offsets and the stand differ)' if recal else 'DIFFERS from the policy'} ({have[:12]})")
    print(f"inputs: {have_in} = {obs_size(he)} x {max(hist, 1)}{' (with the heading error)' if he else ''} "
          f"({'ok' if layout_ok else f'FAIL: the meta says {want_in}'})")
    print(f"inference: {us:.1f} us per step; numpy {np.__version__}")
    return 0 if err < 1e-5 and same and layout_ok else 1


def cmd_imu(link, a) -> int:
    from mp3link import protocol as p
    link.record = []
    end = time.monotonic() + a.seconds
    st, last_print = None, 0.0
    while time.monotonic() < end:
        st = link.wait_state(0.5, after=st)
        if time.monotonic() - last_print > 0.2:
            last_print = time.monotonic()
            acc = np.asarray(st.acc_mg, dtype=float)
            cands = "  ".join(f"{k}: g=({', '.join(f'{v:+.2f}' for v in -(R @ acc) / np.linalg.norm(acc))})"
                              for k, R in IMU_MOUNTS.items()) if np.linalg.norm(acc) > 0 else "acc 0"
            print(f"tilt {tilt_deg(acc):5.1f}  imu {'ok' if st.flags & p.ST_IMU_OK else 'ABSENT'}  "
                  f"gyro dps {np.round(np.asarray(st.gyr_ddps) * 0.1, 1).tolist()}  {cands}", flush=True)
    if a.log:
        from mp3link.cli import save_log
        save_log(link, a.log, None, {"cmd": "imu", "seconds": a.seconds})
    return 0


def _run_common(link, a, act_fn, meta) -> int:
    contract = Contract.load()
    if not a.dry:                       # every armed command (run, replay) needs the measured or a named mounting
        mount_name, mount = load_mount(a.mount)
    else:
        mount_name, mount = load_mount(a.mount) if (a.mount or IMU_MOUNT_JSON.exists()) else ("flip_x", IMU_MOUNTS["flip_x"])
    d = np.load(Path(a.policy) / "policy.npz")
    scale = d["scale"]
    pmeta = json.loads(str(d["meta"]))
    omni = bool(pmeta.get("omni", False))
    heading_term = has_heading_obs(pmeta)
    heading = None
    if omni:
        command = (a.vx, a.vy, a.wz)
        if getattr(a, "heading_loop", False):
            heading = heading_from_args(a)
    elif a.vy or a.wz or getattr(a, "heading_loop", False):
        print("refused: --vy / --wz / --heading-loop need a policy that observes the yaw-rate command "
              "(this one observes vx only)", flush=True)
        return 2
    else:
        command = a.vx
    stop_file = Path(f"{a.log}.stop") if a.log else None     # tools/pi_take.sh --soft-stop creates it
    if stop_file is not None:
        stop_file.unlink(missing_ok=True)
    runner = Runner(link, contract, None, mount, dry=a.dry, cap=a.cap,
                    stdin=sys.stdin if sys.stdin.isatty() else None, stop_file=stop_file)
    runner.tilt_limit = getattr(a, "tilt_cut", TILT_LIMIT_DEG)
    runner.preflight(a.clear)
    link.record = []
    runner.measure_gyro_bias(1.0)
    ob = ObservationBuilder(contract, mount, runner.gyro_bias)
    decoder = ActionDecoder(contract, scale)
    meta.update(tilt_cut_deg=getattr(a, "tilt_cut", TILT_LIMIT_DEG), heading_loop=None if heading is None else heading.describe(),
                heading_obs=heading_term,
                mount=mount_name, vx=a.vx, vy=a.vy, wz=a.wz, omni=omni, seconds=a.seconds, policy=str(Path(a.policy).resolve()),
                policy_sha256=sha256(Path(a.policy) / "policy.npz"), contract_sha256=sha256(CONTRACT_JSON))
    try:
        from mp3link.cli import present
        start = present(link)
        if not a.dry:
            runner.ramp(runner.stand_dd, 2.0, frm=start)
            print(f"holding the stand {a.hold_before:.0f} s (Enter stops)", flush=True)
            runner.hold(runner.stand_dd, a.hold_before)
        print("policy" if a.cmd == "run" else "replay", "start", flush=True)
        runner.loop(act_fn, command, a.seconds, ob, decoder, heading, heading_term)
        runner.stop_reason = "completed"
        runner.safe_end(a.hold_after)
    except StopRun as exc:
        runner.stop_reason = f"stop: {exc}"
        print(runner.stop_reason, flush=True)
        runner.safe_end(a.hold_after)
    except (Disarm, KeyboardInterrupt) as exc:
        runner.stop_reason = f"disarm: {str(exc) or 'Ctrl-C'}"
        link.disarm()
        print(runner.stop_reason, flush=True)
    except BaseException:
        link.disarm()
        raise
    finally:
        if a.log:
            meta["missing_ticks"] = ob.missing
            runner.save(a.log, meta)
    lat = np.asarray(link.latency, dtype=float) * 1000
    if len(lat):
        print(f"SET_ALL -> STATE echo ms: p50 {np.percentile(lat, 50):.2f} p95 {np.percentile(lat, 95):.2f}; "
              f"missing ticks {ob.missing}; steps {len(runner.steps['tick'])}; {runner.stop_reason}")
    return 0


def cmd_gait(link, a) -> int:
    """The firmware trot (minipupperesp's Stanford trot) closed only through the heading loop; as ``run`` otherwise."""
    from . import firmware_gait
    contract = Contract.load()
    if not a.dry:
        mount_name, mount = load_mount(a.mount)
    else:
        mount_name, mount = load_mount(a.mount) if (a.mount or IMU_MOUNT_JSON.exists()) else ("flip_x", IMU_MOUNTS["flip_x"])
    gait = firmware_gait.Gait(contract.offsets_deg)
    stand = contract.dd_from_ticks(gait.ticks(firmware_gait.neutral_feet()))
    if not np.array_equal(stand, contract.stand_dd):
        raise SystemExit(f"the gait's neutral pose {stand.tolist()} is not the contract's stand: check the contract")
    command = (float(np.clip(a.vx, -GAIT_VX_MAX, GAIT_VX_MAX)), float(np.clip(a.vy, -GAIT_VX_MAX, GAIT_VX_MAX)),
               float(np.clip(a.wz, -GAIT_WZ_MAX, GAIT_WZ_MAX)))
    heading = heading_from_args(a) if a.heading_loop else None
    stop_file = Path(f"{a.log}.stop") if a.log else None
    cmd_file = Path(f"{a.log}.cmd") if a.log else None
    for p in (stop_file, cmd_file):
        if p is not None:
            p.unlink(missing_ok=True)
    runner = Runner(link, contract, None, mount, dry=a.dry, cap=a.cap,
                    stdin=sys.stdin if sys.stdin.isatty() else None, stop_file=stop_file)
    runner.tilt_limit = getattr(a, "tilt_cut", TILT_LIMIT_DEG)
    runner.steps.update(command=[], clipped=[])
    runner.preflight(a.clear)
    link.record = []
    runner.measure_gyro_bias(1.0)
    ob = ObservationBuilder(contract, mount, runner.gyro_bias)
    sources = {p.name: firmware_gait.paths.sha256(p) for p in firmware_gait.SOURCES}
    meta = {"cmd": "gait", "policy": "firmware_gait", "policy_sha256": None, "gait_sources": sources,
            "gait_dt": GAIT_DT, "tilt_cut_deg": getattr(a, "tilt_cut", TILT_LIMIT_DEG), "heading_loop": None if heading is None else
            heading.describe(),
            "mount": mount_name, "vx": a.vx, "vy": a.vy, "wz": a.wz, "omni": True, "seconds": a.seconds,
            "contract_sha256": sha256(CONTRACT_JSON)}
    ob.missing = 0
    try:
        from mp3link.cli import present
        start = present(link)
        if not a.dry:
            runner.ramp(runner.stand_dd, 2.0, frm=start)
            print(f"holding the stand {a.hold_before:.0f} s (Enter stops)", flush=True)
            runner.hold(runner.stand_dd, a.hold_before)
        print("gait start", flush=True)
        runner.gait_loop(gait, command, a.seconds, ob, heading, cmd_file)
        runner.stop_reason = "completed"
        runner.safe_end(a.hold_after)
    except StopRun as exc:
        runner.stop_reason = f"stop: {exc}"
        print(runner.stop_reason, flush=True)
        runner.safe_end(a.hold_after)
    except (Disarm, KeyboardInterrupt) as exc:
        runner.stop_reason = f"disarm: {str(exc) or 'Ctrl-C'}"
        link.disarm()
        print(runner.stop_reason, flush=True)
    except BaseException:
        link.disarm()
        raise
    finally:
        if a.log:
            meta["missing_ticks"] = ob.missing
            runner.save(a.log, meta)
    lat = np.asarray(link.latency, dtype=float) * 1000
    if len(lat):
        print(f"SET_ALL -> STATE echo ms: p50 {np.percentile(lat, 50):.2f} p95 {np.percentile(lat, 95):.2f}; "
              f"missing ticks {ob.missing}; gait ticks {len(runner.steps['tick'])}; "
              f"clipped {int(np.sum(runner.steps['clipped']))}; {runner.stop_reason}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mp3_walk_policy.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyAMA2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check")
    s.add_argument("--policy", required=True)
    s = sub.add_parser("imu")
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--log")
    for name in ("run", "replay", "gait"):
        s = sub.add_parser(name)
        if name != "gait":
            s.add_argument("--policy", required=True, help="a policies/<name> directory (policy.npz)")
        s.add_argument("--vx", type=float, default=0.10)
        s.add_argument("--vy", type=float, default=0.0, help="m/s, + = left (omni policies)")
        s.add_argument("--wz", type=float, default=0.0, help="rad/s, + = turn left (omni policies)")
        s.add_argument("--heading-loop", action="store_true",
                       help="hold the starting heading with the gyro (policies that observe wz)")
        s.add_argument("--heading-kp", type=float, default=KP)
        s.add_argument("--heading-ki", type=float, default=KI)
        s.add_argument("--heading-wz-max", type=float, default=WZ_MAX, help="rad/s: the loop's output limit")
        s.add_argument("--heading-aw", choices=ANTIWINDUP, default="cond",
                       help="the loop's anti-windup: cond = hold the integrator while the output is saturated (the "
                            "default); none = no anti-windup (only the integrator's clamp)")
        s.add_argument("--seconds", type=float, default=10.0)
        s.add_argument("--mount", choices=sorted(IMU_MOUNTS))
        s.add_argument("--dry", action="store_true", help="never arm: the servos stay idle")
        s.add_argument("--cap", type=int, default=CAP_MA)
        s.add_argument("--hold-before", type=float, default=10.0, help="s at the stand before (10 on the floor, 1 on a stand in the air)")
        s.add_argument("--hold-after", type=float, default=5.0)
        s.add_argument("--clear", action="store_true", help="clear a latched fault before arming")
        s.add_argument("--tilt-cut", type=float, default=TILT_LIMIT_DEG,
                       help="deg of low-passed accelerometer tilt that disarms; 0 = off (edge strikes read as tilt)")
        s.add_argument("--log")
        if name == "replay":
            s.add_argument("--actions", required=True, help="npz with raw_actions (N x 12) at 50 Hz")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    from mp3link.link import Link
    link = Link(a.port, a.baud)
    try:
        if a.cmd == "imu":
            return cmd_imu(link, a)
        if a.cmd == "gait":
            return cmd_gait(link, a)
        if a.cmd == "run":
            pol = NumpyPolicy(Path(a.policy) / "policy.npz")
            hist = int(pol.meta.get("history", 0) or 0)
            if not hist:
                return _run_common(link, a, lambda obs, k: pol(obs), {"cmd": "run"})
            stack = HistoryStacker(hist, has_heading_obs(pol.meta))   # the last ``hist`` observations per term

            def act(obs, k):
                if k == 0:
                    stack.reset()
                return pol(stack(obs))
            return _run_common(link, a, act, {"cmd": "run", "history": hist})
        acts = np.load(a.actions)["raw_actions"]
        a.seconds = min(a.seconds, len(acts) * CONTROL_DT)
        return _run_common(link, a, lambda obs, k: acts[min(k, len(acts) - 1)],
                           {"cmd": "replay", "file": str(a.actions)})
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
