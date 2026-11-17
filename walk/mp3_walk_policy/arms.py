"""The actuator arms: what turns a policy target into a joint torque.

An arm is a set of mjlab actuator configs plus each config's own command latency. The sensing-to-command chain delay
(the same for every arm) is added at reset by ``mdp.reset_command_delay``; every actuator holds its lag for the whole
episode (``delay_hold_prob = 1``: mjlab otherwise resamples a lag at every physics step, and actuator groups with
different transmissions keep separate buffers).

``stock``: mpx-sim's ``sim`` servo profile (model/servo_profiles.json): a MuJoCo position
actuator kp 8 N m/rad, kv 0.05, +-0.8 N m; damping 0.0381 and armature 2e-4 on the actuated joint, or on the knee's
tendon; its ``latency_ms`` (5) as the command latency. The XML's frictionloss stays (0.003 on the abduction and hip
joints, 0.001 on the passive calf hinge, none on the tendon), and so do the calf hinge's own damping and armature, as
in mpx-sim (servo.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType

from . import paths

PHYSICS_DT = 0.001
MAX_LAG_STEPS = 60          # the delay buffers' capacity (ms at 1 ms physics); lags are set per episode below it
JOINT_TARGETS = (r".*_hip", r".*_thigh")
KNEE_TARGETS = (r".*_knee_drive",)


@dataclass(frozen=True)
class Arm:
    name: str
    actuators: tuple            # mjlab ActuatorCfg, in the entity's order
    latency_steps: tuple        # each actuator config's own command latency, physics steps
    params: dict                # every parameter, echoed into the run config


def stock_profile() -> dict:
    return json.loads(paths.SERVO_PROFILES.read_text())["profiles"]["sim"]


def stock() -> Arm:
    p = stock_profile()
    kw = dict(stiffness=p["kp"], damping=p["kv"], effort_limit=p["umax"], armature=p["armature"],
              viscous_damping=p["damping"], delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS, delay_hold_prob=1.0)
    joints = BuiltinPositionActuatorCfg(target_names_expr=JOINT_TARGETS, **kw)
    knees = BuiltinPositionActuatorCfg(target_names_expr=KNEE_TARGETS, transmission_type=TransmissionType.TENDON, **kw)
    lat = int(round(p["latency_ms"] / 1000.0 / PHYSICS_DT))
    return Arm("stock", (joints, knees), (lat, lat),
               {"profile": "mpx-sim servo_profiles.json 'sim'", **{k: p[k] for k in
                ("kp", "kv", "umax", "damping", "armature", "latency_ms")},
                "source_sha256": paths.sha256(paths.SERVO_PROFILES)})


# The BAM arms: the knee and the abduction run the md01/id_3 set, the hip md01/id_2; class md01vp, m6, at its class
# defaults (kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V). Each set's fitted command delay is applied on the 1 ms
# physics grid, rounded up.
BAM_ROLES = (("abd", JOINT_TARGETS[:1], "md01/id_3", TransmissionType.JOINT),
             ("hip", JOINT_TARGETS[1:], "md01/id_2", TransmissionType.JOINT),
             ("knee", KNEE_TARGETS, "md01/id_3", TransmissionType.TENDON))


def command_delay_steps(params_path) -> int:
    """A set's fitted ``command_delay`` in physics steps: a command takes effect at the first 1 ms step at or after its
    time + delay, i.e. the delay rounded up. Published m6: id_3 1.08 ms -> 2, id_2 3.68 ms -> 4; T5 ub06: id_3 0.02 ms
    -> 1, id_2 4.98 ms -> 5."""
    import math
    ms = float(json.loads(Path(params_path).read_text())["command_delay"]) * 1000.0
    return int(math.ceil(ms / (PHYSICS_DT * 1000.0) - 1e-9))


def bam(name: str = "bam_published", params_json: dict | None = None, scales: tuple = (),
        vin_range: tuple | None = None, stiff_frictionloss: bool = False) -> Arm:
    """``params_json``: role -> a fitted JSON in place of the published m6;
    ``scales``: ((parameter, factor), ...) on every role (e.g. ``armature`` for a reduced-armature variant);
    ``stiff_frictionloss``: False matches mpx-sim's CPU bank (MuJoCo's default friction constraint)."""
    from .bam_servo import BamServoCfg
    params_json = params_json or {}
    cfgs, lat, params = [], [], {"roles": {}, "scales": [list(s) for s in scales], "vin_range": vin_range,
                                  "stiff_frictionloss": stiff_frictionloss}
    for role, targets, motor, trn in BAM_ROLES:
        src = dict(json_path=str(params_json[role])) if role in params_json else dict(motor_name=motor, model="m6")
        path = params_json.get(role) or (_bam_params() / motor / "m6.json")
        delay = command_delay_steps(path)
        cfgs.append(BamServoCfg(target_names_expr=targets, transmission_type=trn, vin_range=vin_range,
                                stiff_frictionloss=stiff_frictionloss, param_scales=tuple(scales),
                                delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS, delay_hold_prob=1.0, **src))
        lat.append(delay)
        params["roles"][role] = {"params": _rel(path), "sha256": paths.sha256(path), "command_delay_ms": delay}
    return Arm(name, tuple(cfgs), tuple(lat), params)


def _bam_params() -> Path:
    """The published parameter sets in the installed bam package."""
    import bam as bam_pkg
    return Path(bam_pkg.__file__).parent / "params"


def _rel(path) -> str:
    """A parameter file's path relative to this repository's root, or as "bam/params/..." inside the bam package (for
    run configs)."""
    path = Path(path).resolve()
    for root, tag in ((paths.REPO, ""), (_bam_params().parent, "bam/")):
        if path.is_relative_to(root.resolve()):
            return tag + str(path.relative_to(root.resolve()))
    return str(path)


def t5ub06() -> Arm:
    """T5 ub06: a refit of md01vp m6 on the bench data with the armature pinned at 0.6 of a bound measured on the bare
    output arm (the knee 2.2e-4, the hip 2.7e-4 kg m^2) and kd_ratio at 1.0; it predicts SKU2's joints better than the
    published sets at the robot's light loads (data/bam_t5ub06_id_*_m6.json)."""
    return bam(name="bam_t5ub06", params_json={"abd": paths.DATA / "bam_t5ub06_id_3_m6.json",
                                               "hip": paths.DATA / "bam_t5ub06_id_2_m6.json",
                                               "knee": paths.DATA / "bam_t5ub06_id_3_m6.json"})


ARMS = {"stock": stock, "bam_published": bam, "bam_t5ub06": t5ub06}


def trained_arm(path) -> str | None:
    """The arm a run or an exported policy was trained in: its ``config.json`` (in the directory, or next to a
    checkpoint)."""
    p = Path(path)
    for cfg in (p / "config.json", p.parent / "config.json"):
        if cfg.is_file():
            return json.loads(cfg.read_text())["task"]["arm"]
    return None


def get(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]()
