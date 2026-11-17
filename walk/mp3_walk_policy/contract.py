"""The policy contract for SKU2: what the network sees and what its output does, in numpy.

This module is the robot side's reference (the Pi runner imports it; it needs numpy only). The training environment
implements the same arithmetic in torch (`mdp.py`); `tests/test_contract.py` checks that the two agree.

Coordinates. Per servo id 1..12 (FR, FL, RR, RL x abduction, hip, knee), the policy works in the servo's own angle in
radians, relative to the stand: ``rel_i = sign_i * radians(servo_deg_i)``, where ``servo_deg`` is the angle
minipupperesp's ``servo_write`` takes without the calibration offset (0 = the stand). In the MuJoCo model the abduction
and hip are the ``*_hip`` and ``*_thigh`` joints and the knee is the ``*_knee_drive`` tendon (the shin's angle); the
signs and the stand come from mpx-sim's mapping (``data/sku2_contract.json``).

The firmware's units (``minipupperesp`` ``servo_write``, main.c:2526; ``driver_board.c:175``; ``mp3link`` cli.py):
``tick = 511 + (int)((servo_deg + offset) * 10240 / 3100)``, ``dd = (int)(3100 - tick * 3100 / 1024)``, both in
float32. The exact inverse for a measured AT32 angle is ``servo_deg = (DD0 - dd) / 10 - offset``, DD0 the dd of tick
511 (convert with 1024, not 1023).

Observation (45, float32): base angular velocity (3, rad/s, body frame, the gyro bias removed), gravity direction (3,
-acc/|acc| in the body frame: (0, 0, -1) upright), command (3: vx, 0, 0 m/s; an omni policy: vx, vy m/s, wz rad/s, body
frame), position (12, rel, rad), velocity (12,
rad/s, ``VelocityFilter``), previous applied action (12).

A policy exported with ``heading_error`` in its meta's ``obs_terms_extra`` (``has_heading_obs``) observes 46: the
45, then the heading loop's error (``heading_obs``: wrap(target - heading), clipped to +-HEADING_CLIP rad, over
HEADING_CLIP: [-1, 1], + = turn left to correct; 0 without the loop). Every other policy's layout is unchanged.

Action (12): ``a`` -> clip to [-1, 1] -> ``scale * a`` (rel, rad) -> the band about the stand -> at most
``STEP_LIMIT_DEG`` from the previous applied target -> firmware ticks -> the dd sent. The previous action observed is
the applied target in action units.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MP3_CONTRACT=<a file in mp3_walk_policy/data> (read once, when Python starts): another servo contract, e.g.
# arena_sku1_contract.json (MPX Arena's Mini Pupper 4 / SKU1: the same signs, the firmware's mid-scale stand)
_CONTRACT_OVERRIDE = os.environ.get("MP3_CONTRACT", "").strip()
CONTRACT_JSON = Path(__file__).resolve().parent / "data" / (_CONTRACT_OVERRIDE or "sku2_contract.json")
if not CONTRACT_JSON.is_file():
    raise FileNotFoundError(f"MP3_CONTRACT={_CONTRACT_OVERRIDE}: no such file {CONTRACT_JSON}")

DD0 = 3100.0 - 511.0 * 3100.0 / 1024.0          # 1553.02734375, the dd of tick 511
STATE_DT = 0.005                                # the ESP32's tick: one STATE frame per 5 ms
CONTROL_DT = 0.020                              # the policy's step (50 Hz)
STEP_LIMIT_DEG = 20.0                           # per policy step, the same in simulation and on the robot
ABDUCTION = (0, 3, 6, 9)                        # indices of the abductions in servo order
BAND_DEG = np.array([20.0 if i in ABDUCTION else 60.0 for i in range(12)])   # mp3link cli.py MAX_DEV_DEG
OBS_SIZE = 45
ACTION_SIZE = 12
# The firmware trot's 360 ms cycle (24 ticks of 15 ms, stanford_gait.h) is 18 policy steps.
GAIT_CYCLE_STEPS = 18
F32 = np.float32


@dataclass
class Contract:
    signs: np.ndarray
    stand_rel_rad: np.ndarray            # the stand in the model's coordinates (joint angles, tendon lengths)
    stand_joint_rad: np.ndarray          # all twelve model joints at the stand (hip, thigh, calf per leg)
    offsets_deg: np.ndarray
    stand_dd: np.ndarray
    stand_ticks: np.ndarray
    joint_order: list
    policy_targets: list
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = CONTRACT_JSON) -> "Contract":
        d = json.loads(Path(path).read_text())
        return cls(signs=np.array(d["signs"], dtype=float), stand_rel_rad=np.array(d["stand_policy_rad"]),
                   stand_joint_rad=np.array(d["stand_joint_rad"]), offsets_deg=np.array(d["offsets_deg"]),
                   stand_dd=np.array(d["stand_dd"]), stand_ticks=np.array(d["stand_ticks"]),
                   joint_order=d["joint_order"], policy_targets=d["policy_targets"], meta=d)

    # ---- firmware units -------------------------------------------------------------------------------------------
    def ticks_from_servo_deg(self, servo_deg) -> np.ndarray:
        ang = (np.asarray(servo_deg, dtype=F32) + self.offsets_deg.astype(F32)).astype(F32)
        sig = 511 + np.trunc(ang * F32(10240.0) / F32(3100.0)).astype(np.int64)
        return np.clip(sig, 0, 1023)

    @staticmethod
    def dd_from_ticks(ticks) -> np.ndarray:
        t = np.asarray(ticks, dtype=F32)
        return np.trunc(F32(3100.0) - t * F32(3100.0) / F32(1024.0)).astype(np.int64)

    def servo_deg_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (dd) in servo degrees. Not a target's inverse: the dd of tick t reads up to 0.1 deg
        above t's angle, and C's truncation towards zero turns that into t + 1 for ticks below 511."""
        return (DD0 - np.asarray(dd, dtype=float)) / 10.0 - self.offsets_deg

    def servo_deg_of_tick(self, ticks) -> np.ndarray:
        """The target angle (servo degrees) whose firmware tick is exactly ``ticks``: the centre of the tick's
        truncation interval (k = tick - 511: [k, k+1) for k > 0, (k-1, k] for k < 0, (-1, 1) for k = 0)."""
        k = np.asarray(ticks, dtype=float) - 511.0
        return (k + 0.5 * np.sign(k)) * 3100.0 / 10240.0 - self.offsets_deg

    # ---- policy coordinates ---------------------------------------------------------------------------------------
    def rel_from_servo_deg(self, servo_deg) -> np.ndarray:
        return self.signs * np.radians(np.asarray(servo_deg, dtype=float))

    def servo_deg_from_rel(self, rel) -> np.ndarray:
        return np.degrees(np.asarray(rel, dtype=float)) * self.signs

    def rel_from_ticks(self, ticks) -> np.ndarray:
        """A policy target that the firmware path turns into exactly ``ticks`` (the teacher's targets)."""
        return self.rel_from_servo_deg(self.servo_deg_of_tick(ticks))

    def rel_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (STATE ``pos_dd``) as a policy position (rad, relative to the stand)."""
        return self.rel_from_servo_deg(self.servo_deg_from_dd(dd))

    def quantise_rel(self, rel) -> tuple[np.ndarray, np.ndarray]:
        """(the dd the firmware path would send for a target, that dd back in policy coordinates)."""
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(self.servo_deg_from_rel(rel)))
        return dd, self.rel_from_dd(dd)


class ActionDecoder:
    """Stateful: the step limit acts against the previous applied target. ``reset()`` at the stand."""

    def __init__(self, contract: Contract, scale, quantise: bool = True):
        self.c = contract
        self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (ACTION_SIZE,)).copy()
        self.quantise = quantise
        self.band = np.radians(BAND_DEG)
        self.limit = np.radians(STEP_LIMIT_DEG)
        self.reset()

    def reset(self) -> None:
        self.prev_rel = np.zeros(ACTION_SIZE)
        self.prev_action = np.zeros(ACTION_SIZE)

    def __call__(self, raw_action) -> dict:
        a = np.clip(np.asarray(raw_action, dtype=float), -1.0, 1.0)
        rel = np.clip(self.scale * a, -self.band, self.band)
        rel = np.clip(rel, self.prev_rel - self.limit, self.prev_rel + self.limit)
        if self.quantise:
            dd, rel = self.c.quantise_rel(rel)
        else:
            dd = None
        self.prev_rel = rel
        self.prev_action = rel / self.scale
        return {"dd": dd, "rel": rel, "action": self.prev_action.copy()}


class VelocityFilter:
    """Joint velocity from the 200 Hz positions: a median of 3, then the least-squares slope over the last 4.

    Fed every STATE frame (5 ms). Until six samples have arrived, the history is filled with the first one."""

    WEIGHTS = np.array([-1.5, -0.5, 0.5, 1.5]) / (5.0 * STATE_DT)

    def __init__(self, n: int = ACTION_SIZE):
        self.hist = None
        self.n = n

    def reset(self) -> None:
        self.hist = None

    def push(self, x) -> None:
        x = np.asarray(x, dtype=float)
        if self.hist is None:
            self.hist = np.repeat(x[None, :], 6, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], x[None, :]], axis=0)

    def velocity(self) -> np.ndarray:
        if self.hist is None:
            return np.zeros(self.n)
        h = self.hist
        med = np.stack([np.median(h[k:k + 3], axis=0) for k in range(4)])
        return self.WEIGHTS @ med


def gravity_from_accel(acc_body) -> np.ndarray:
    acc = np.asarray(acc_body, dtype=float)
    return -acc / np.linalg.norm(acc)


# The observation's terms (name, slice), in order: what a history stacks per term (mjlab's observation manager).
OBS_TERMS = (("ang_vel", slice(0, 3)), ("gravity", slice(3, 6)), ("command", slice(6, 9)), ("pos", slice(9, 21)),
             ("vel", slice(21, 33)), ("prev_action", slice(33, 45)))
HEADING_TERM = "heading_error"          # the optional 46th float, appended last (its own block in a history)
HEADING_CLIP = 0.8                      # rad: the observed error saturates here (the loop's wz from 0.2: kp 2, 0.4)


def has_heading_obs(meta: dict) -> bool:
    """Does an exported policy observe the heading error (its meta's ``obs_terms_extra``)?"""
    return HEADING_TERM in (meta.get("obs_terms_extra") or ())


def obs_terms(heading: bool = False) -> tuple:
    return OBS_TERMS + ((HEADING_TERM, slice(OBS_SIZE, OBS_SIZE + 1)),) if heading else OBS_TERMS


def obs_size(heading: bool = False) -> int:
    return OBS_SIZE + 1 if heading else OBS_SIZE


def heading_obs(err) -> float:
    """The observed heading error: ``err`` (rad, + = turn left to correct) clipped to +-HEADING_CLIP, over it."""
    return float(np.clip(err, -HEADING_CLIP, HEADING_CLIP)) / HEADING_CLIP


class HistoryStacker:
    """The actor's observation with a history of ``length`` steps, as mjlab builds it: each term's frames oldest to
    newest, the terms concatenated in order; the first observation after a reset fills every slot. ``heading``: the
    observation has the heading error (46 floats), its frames the last block."""

    def __init__(self, length: int, heading: bool = False):
        self.length = int(length)
        self.terms = obs_terms(heading)
        self.frames: list | None = None

    def reset(self) -> None:
        self.frames = None

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=F32)
        if self.frames is None:
            self.frames = [obs] * self.length
        else:
            self.frames = self.frames[1:] + [obs]
        h = np.stack(self.frames)                                   # (length, 45 or 46), oldest first
        assert h.shape[1] == self.terms[-1][1].stop, h.shape
        return np.concatenate([h[:, sl].reshape(-1) for _, sl in self.terms]).astype(F32)


def assemble_observation(ang_vel, gravity, command, pos_rel, vel, prev_action, heading_error=None) -> np.ndarray:
    """``command``: vx (a forward policy) or (vx, vy, wz) (an omni policy). ``heading_error``: the observed heading
    term (``heading_obs``) of a policy that has it, appended as the 46th float; None: the 45."""
    obs = np.empty(obs_size(heading_error is not None), dtype=F32)
    obs[0:3] = ang_vel
    obs[3:6] = gravity
    obs[6:9] = (command, 0.0, 0.0) if np.ndim(command) == 0 else command
    obs[9:21] = pos_rel
    obs[21:33] = vel
    obs[33:45] = prev_action
    if heading_error is not None:
        obs[45] = heading_error
    return obs
