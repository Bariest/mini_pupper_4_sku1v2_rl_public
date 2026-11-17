"""SKU2 as an mjlab entity: mpx-sim's model read in place, with SKU2's weighed masses and corrected stand.

Source: mpx-sim ``model/sku_2/pupper.xml`` (75dd334; its hash is recorded in every run's config). Changes, made on the
MjSpec at load time (the file is never edited):

- the ghost robot (``g_base``), the ``stand`` keyframe, the twelve XML position actuators (the arm's actuators replace
  them) and the XML sensor are removed; with ``strip_visuals`` the mesh geoms (``contype = conaffinity = 0``) and the
  meshes too (masses are explicit ``<inertial>``s, so the dynamics do not change);
- an IMU at site ``imu``: gyro ``imu_ang_vel``, velocimeter ``imu_lin_vel``, accelerometer ``imu_lin_acc``;
- the weighed masses (data/sku2_masses.json: each module
  66.82 g, each leg's thigh and calf scaled together to 6.5 g, the base takes the rest of 450 g, inertias scale with
  mass, the base COM x set to the value solved for the measured 56.6 % rear share);
- the feet get contact priority 1, so that their friction (randomised) is the one MuJoCo uses against the ground
  (equal priorities take the larger of the two, and the ground's is 1.0).

The stand (all twelve joints) comes from ``data/sku2_contract.json``: mpx-sim's mapping with SKU2's stand geometry
corrected to the firmware's neutral pose (hip axes 70 mm above the feet).

MP3_ROBOT=sku1v2 (``paths``) builds SKU1's body instead: minipupperesp's ``sim/model/pupper.xml`` (the same joints,
knee tendons, feet, IMU site and ghost robot, so every change above applies as written), with
data/sku1v2_masses.json (390 g, 195 / 195 g on the front and rear feet, the base COM x solved for that 50 % rear
share by tools/robot_masses.py). The contract's stand puts SKU1's model at the same neutral pose.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from . import paths
from .contract import Contract

LEGS = ("FR", "FL", "RR", "RL")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
LEG_COLLISION_GEOMS = tuple(f"{leg}_{part}_collision" for leg in ("FR", "FL", "RR", "RL") for part in ("thigh", "calf"))
BODY_CONTACT_BODIES = ("base",) + tuple(f"{leg}_hip" for leg in LEGS)   # the trunk and the motor cans
# The base origin's height at the settled stand on the stock servos (71.9 mm); the start sits just above it.
INIT_BASE_Z = float(os.environ.get("MP3_INIT_Z", "") or 0.0725)   # MP3_INIT_Z: another start height (m)


def masses(path=None) -> dict:
    p = paths.MASSES_JSON if path is None else paths.DATA / path
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing (MP3_ROBOT={paths.ROBOT}): tools/robot_masses.py writes it")
    m = json.loads(p.read_text())
    want = m.get("robot")
    if want is not None and want != paths.ROBOT:
        raise ValueError(f"{p.name} is for {want}, but MP3_ROBOT={paths.ROBOT}")
    return m


def apply_masses(spec: mujoco.MjSpec, m: dict) -> dict:
    new = {}
    for leg in LEGS:
        new[f"{leg}_hip"] = m["module_kg"]
        t, c = spec.body(f"{leg}_thigh").mass, spec.body(f"{leg}_calf").mass
        k = m["leg_kg"] / (t + c)
        new[f"{leg}_thigh"], new[f"{leg}_calf"] = t * k, c * k
    others = sum(b.mass for b in spec.bodies if b.name and b.name not in new and b.name not in ("base", "world"))
    new["base"] = m["total_kg"] - sum(new.values()) - others
    for name, mass in new.items():
        b = spec.body(name)
        b.fullinertia = np.asarray(b.fullinertia) * (mass / b.mass)
        b.mass = mass
    ipos = np.array(spec.body("base").ipos)
    ipos[0] = m["base_com_x_m"]
    spec.body("base").ipos = ipos
    return new


def leg_base_collisions(spec: mujoco.MjSpec) -> None:
    """Let each shin and foot collide with the base boxes (the model has no self-collision: robot geoms are contype 1,
    conaffinity 2). Bit 4 on the base's conaffinity and the calves' contype pairs exactly those; the thighs start on
    the base box and stay excluded. Terrain contacts are unchanged."""
    for g in spec.body("base").geoms:
        if g.contype or g.conaffinity:
            g.conaffinity |= 4
    for leg in LEGS:
        for g in spec.body(f"{leg}_calf").geoms:
            if g.contype or g.conaffinity:
                g.contype |= 4


def get_spec(strip_visuals: bool = True, masses_json: str | None = None,
             self_collision: bool = False, foot_solref: tuple | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(paths.ROBOT_XML))     # MP3_ROBOT: sku2 (mpx-sim) or sku1v2 (minipupperesp)
    spec.delete(spec.body("g_base"))
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)
    for s in list(spec.sensors):
        spec.delete(s)
    if strip_visuals:
        for g in list(spec.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    for name, kind in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
                       ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
                       ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    for leg in LEGS:                         # name the legs' collision capsules (feet are named in the MJCF)
        for part in ("thigh", "calf"):
            for k, g in enumerate(g for g in spec.body(f"{leg}_{part}").geoms
                                  if (g.contype or g.conaffinity) and g.name not in FOOT_GEOMS):
                g.name = f"{leg}_{part}_collision{k or ''}"
    for g in FOOT_GEOMS:
        spec.geom(g).priority = 1
        if foot_solref is not None:          # the feet's priority makes their contact parameters the ones used
            spec.geom(g).solref = list(foot_solref)
    apply_masses(spec, masses(masses_json))
    if self_collision:
        leg_base_collisions(spec)
    return spec


def robot_cfg(actuators: tuple, strip_visuals: bool = True, masses_json: str | None = None,
              self_collision: bool = False, foot_solref: tuple | None = None) -> EntityCfg:
    c = Contract.load()
    joint_pos = {name: float(v) for name, v in zip(c.joint_order, c.stand_joint_rad)}
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_BASE_Z), joint_pos=joint_pos,
                                             joint_vel={".*": 0.0}),
        spec_fn=lambda: get_spec(strip_visuals, masses_json, self_collision, foot_solref),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


def sources(masses_json: str | None = None) -> dict:
    """Hashes of every input the model is built from (written into each run's config), and the robot."""
    def name(p):
        for root, tag in ((paths.MPX_SIM, "mpx-sim/"), (paths.MINIPUPPERESP, "minipupperesp/"), (paths.REPO, "")):
            try:
                return tag + p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                pass
        return str(p)
    m = paths.MASSES_JSON if masses_json is None else paths.DATA / masses_json
    return {"robot": paths.ROBOT,
            name(paths.ROBOT_XML): paths.sha256(paths.ROBOT_XML),
            name(paths.SERVO_PROFILES): paths.sha256(paths.SERVO_PROFILES),
            name(m): paths.sha256(m)}
