"""The SKU2 forward-walking task in mjlab (MP2's phase-2 recipe, ported).

One policy step is 20 ms (50 Hz) of twenty 1 ms physics steps. Commands: forward speed U(0.05, 0.20) m/s per episode
(the firmware trot's designed range), no sideways or turning command. Episodes: 10 s from the stand, deterministic
resets. Rewards: the Mini Pupper 2 walking policy's set (tracking, the
regularisers, straightness, swing progress with a recent touchdown, heavy stance slip). Randomisation, initially: the
feet's friction, the base COM, the per-episode chain delay, the observation noise (ramped as MP2).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import arms as arms_mod
from . import mdp
from .sku2 import BODY_CONTACT_BODIES, FOOT_GEOMS, robot_cfg
from .takes import SPIN_RANGE

DECIMATION = 20
EPISODE_S = 10.0
VX_RANGE = (0.05, 0.20)
CHAIN_MS = (8.0, 19.0)
# Observation noise, uniform half-widths (rad/s, unit vector, rad, rad/s). The ramp starts from a clean floor and ends
# at a measured profile: N1 (data/noise_n1.json, one value per channel, the variance excess of SKU2's logs over the
# simulation along a trot) or N2 (data/noise_n2.json, one value per element, the sensor noise plus the broadband
# excess in motion; see the README). NOISE_END is N1; training uses N2 by default (scripts/train.py --noise).
NOISE_START = {"ang_vel": 0.02, "gravity": 0.01, "pos": 0.002, "vel": 0.05}
NOISE_N1 = Path(__file__).resolve().parent / "data" / "noise_n1.json"
NOISE_END = {k: float(v) for k, v in json.loads(NOISE_N1.read_text())["profile_uniform_half_width"].items()}
NOISE_HOLD_ITERS, NOISE_RAMP_ITERS = 150, 300
NOISE_N2 = Path(__file__).resolve().parent / "data" / "noise_n2.json"
# Omni: |(vx, vy)| and |wz| ranges. Over them the omni teacher reaches 16.5 deg (abduction), 24.2 (hip), 19.6 (knee).
OMNI_SPEED_RANGE = (0.05, 0.12)
OMNI_WZ_MAX = 0.6
OMNI_ACTION_SCALE = (0.32, 0.45, 0.42) * 4
OMNI_RESAMPLE_S = (4.0, 6.0)
# The sprint (sprint_options): straight ahead as fast as told. MPX Arena's 1 m sprint asks for 0.10-0.80 m/s and
# scores the best; the arena's servos have only their own 5 ms command lag (no sensing-to-command chain), so the
# sprint trains with a short chain by default.
SPRINT_MIN = 0.10
SPRINT_MAX = 0.60
SPRINT_START_HI = 0.25
SPRINT_CHAIN_MS = (0.0, 4.0)


def noise_profile(name: str) -> dict:
    """The end amplitudes by name: ``n1`` (scalar per channel), ``n2`` (per element) and ``n2x2``
    (twice N2, the safety check)."""
    if name == "n1":
        return dict(NOISE_END)
    if name in ("n2", "n2x2"):
        prof = json.loads(NOISE_N2.read_text())["profile_uniform_half_width"]
        k = 2.0 if name == "n2x2" else 1.0
        return {ch: [k * float(v) for v in vals] for ch, vals in prof.items()}
    raise ValueError(f"unknown noise profile {name!r}")


@dataclass
class TaskOptions:
    arm: str = "stock"
    num_envs: int = 4096
    noise_start: dict = field(default_factory=lambda: dict(NOISE_START))
    noise_end: dict = field(default_factory=lambda: dict(NOISE_END))
    noise_ramp: bool = True                 # False: the end amplitudes from the start (evaluation under noise)
    noisy: bool = True                      # False: clean observations (evaluation, play)
    randomise: bool = True                  # friction, COM, chain delay drawn per env
    chain_ms: tuple = CHAIN_MS
    vx_range: tuple = VX_RANGE
    action_scale: tuple = (0.10, 0.40, 0.42) * 4    # the teacher reaches 21.5 deg (hip) and 23.9 deg (knee)
    episode_s: float = EPISODE_S
    visuals: bool = False                   # keep the meshes (rendering only)
    # omni (``omni_options``): joystick commands (vx, vy, wz), observed whole; direction-agnostic rewards
    omni: bool = False
    speed_range: tuple = OMNI_SPEED_RANGE
    wz_max: float = OMNI_WZ_MAX
    rel_turning: float = 0.5                # fraction of commands with a yaw rate
    rel_spin: float = 0.1                   # fraction turning in place
    fixed_command: tuple | None = None      # (vx, vy, wz) for every env (evaluation)
    resample_s: tuple = (1.0e6, 1.0e6)      # command resampling interval (the forward task: never)
    # omni with stops (``stops_options``): take-like schedules (takes.schedule_bank) instead of resampled commands
    stops: bool = False
    schedules: int = 16384
    schedule_seed: int = 0
    # the sprint (``sprint_options``): forward only, a speed curriculum, straightness weighted up
    sprint: bool = False
    sprint_max: float = SPRINT_MAX              # the curriculum's end: commands up to this (m/s)
    sprint_start_hi: float = SPRINT_START_HI    # ... starting from this
    sprint_hold_iters: int = 100
    sprint_ramp_iters: int = 700
    sprint_slow_frac: float = 0.15              # this share of episodes at the walk's speeds (0.05-0.12)
    track_rel: float = 0.20                     # tracking tolerance: max(0.10, track_rel * speed)
    heading_weight: float = -3.0
    cross_weight: float = -3.0
    base_mass_scale: tuple | None = None        # the base's mass drawn x this range (e.g. (0.85, 1.15))
    reset_yaw_deg: float = 0.0                  # a random heading at reset, +- (the arena: 2 deg)


def omni_options(**kw) -> TaskOptions:
    """The omni task: ``TaskOptions`` with omni commands and the omni action scale."""
    kw.setdefault("action_scale", OMNI_ACTION_SCALE)
    return TaskOptions(omni=True, **kw)


def sprint_options(**kw) -> TaskOptions:
    """The sprint: the omni observation and action (a walk policy trains on from where it is), commands straight
    ahead in [SPRINT_MIN, a top speed that a curriculum raises to ``sprint_max``], no turning, a short chain delay
    (the arena's), the heading and the line held harder, the base's mass varied."""
    kw.setdefault("chain_ms", SPRINT_CHAIN_MS)
    kw.setdefault("base_mass_scale", (0.85, 1.15))
    kw.setdefault("reset_yaw_deg", 3.0)
    kw.setdefault("rel_turning", 0.0)
    kw.setdefault("rel_spin", 0.0)
    kw.setdefault("speed_range", (SPRINT_MIN, kw.get("sprint_start_hi", SPRINT_START_HI)))
    return omni_options(sprint=True, **kw)


def stops_options(**kw) -> TaskOptions:
    """The omni task commanded by take-like schedules (walks, turns in place to SPIN_RANGE, stops), with a reward for
    standing still when stopped."""
    return omni_options(stops=True, **kw)


def install_noise(env, env_ids, start: dict, end: dict, ramp: bool):
    if not hasattr(env, "mp3_noise"):
        env.mp3_noise = mdp.NoiseTable(env, start, end)
        if not ramp:
            env.mp3_noise.amp = dict(env.mp3_noise.end)


def _obs_terms(noisy: bool, omni: bool = False) -> dict:
    return {
        "ang_vel": ObservationTermCfg(func=mdp.obs_ang_vel, params={"noisy": noisy}),
        "gravity": ObservationTermCfg(func=mdp.obs_gravity, params={"noisy": noisy}),
        "command": ObservationTermCfg(func=mdp.obs_command, params={"omni": omni}),
        "pos": ObservationTermCfg(func=mdp.obs_pos, params={"noisy": noisy}),
        "vel": ObservationTermCfg(func=mdp.obs_vel, params={"noisy": noisy}),
        "prev_action": ObservationTermCfg(func=mdp.obs_prev_action),
    }


def make_env_cfg(opt: TaskOptions | None = None) -> ManagerBasedRlEnvCfg:
    opt = opt or TaskOptions()
    arm = arms_mod.get(opt.arm)
    critic = _obs_terms(False, opt.omni)
    critic.update({
        "lin_vel": ObservationTermCfg(func=mdp.obs_true_lin_vel),
        "height": ObservationTermCfg(func=mdp.obs_base_height),
        "contact": ObservationTermCfg(func=mdp.obs_foot_contact),
    })
    if opt.omni:
        critic["track_errors"] = ObservationTermCfg(func=mdp.obs_track_errors)
    observations = {
        "actor": ObservationGroupCfg(terms=_obs_terms(opt.noisy, opt.omni), concatenate_terms=True,
                                     enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
        "anchor": ObservationGroupCfg(terms=_obs_terms(False, opt.omni), concatenate_terms=True,
                                      enable_corruption=False),
    }
    sensors = (
        ContactSensorCfg(name="feet_contact",
                         primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),
        ContactSensorCfg(name="body_contact",
                         primary=ContactMatch(mode="body", pattern=BODY_CONTACT_BODIES, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found",), reduce="none", num_slots=1),
    )
    events = {
        "noise": EventTermCfg(func=install_noise, mode="startup",
                              params={"start": opt.noise_start, "end": opt.noise_end, "ramp": opt.noise_ramp}),
        "reset_base": EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {}, "velocity_range": {}}),
        "reset_joints": EventTermCfg(func=vel_mdp.reset_joints_by_offset, mode="reset",
                                     params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                             "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "command_delay": EventTermCfg(func=mdp.reset_command_delay, mode="reset",
                                      params={"latency_steps": arm.latency_steps,
                                              "chain_ms": opt.chain_ms if opt.randomise else
                                              (sum(opt.chain_ms) / 2,) * 2}),
    }
    if arm.name.startswith("bam"):
        from .bam_servo import bam_servo_init
        events["bam_init"] = EventTermCfg(func=bam_servo_init, mode="startup")
    if opt.randomise:
        events["foot_friction"] = EventTermCfg(
            func=envs_mdp.dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                    "axes": [0], "ranges": (0.4, 1.2), "shared_random": True})
        events["base_com"] = EventTermCfg(
            func=envs_mdp.dr.body_com_offset, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "add",
                    "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.003, 0.003)}})
        if opt.base_mass_scale is not None:
            events["base_mass"] = EventTermCfg(
                func=envs_mdp.dr.body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "scale",
                        "ranges": tuple(opt.base_mass_scale)})
    if opt.reset_yaw_deg > 0.0:
        a = math.radians(opt.reset_yaw_deg)
        events["reset_base"] = EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                            params={"pose_range": {"yaw": (-a, a)}, "velocity_range": {}})
    rewards = {
        "track_lin_vel": RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.10}),
        "track_ang_vel": RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.50, params={"std": 0.10}),
        "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.10),
        "torques": RewardTermCfg(func=mdp.torques_l2, weight=-1.0e-4),
        "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-7),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
        "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
        "yaw": RewardTermCfg(func=mdp.yaw_l2, weight=-0.40),
        "lateral_position": RewardTermCfg(func=mdp.lateral_position_l2, weight=-1.50),
        "lateral_velocity": RewardTermCfg(func=mdp.lateral_velocity_l2, weight=-0.40),
        "swing_progress": RewardTermCfg(func=mdp.SwingProgressTouchdown, weight=0.80, params={}),
        "stance_slip": RewardTermCfg(func=mdp.stance_foot_slip_l2, weight=-2.0, params={"threshold": 1.0}),
    }
    if opt.omni:
        # the forward task's terms that assume travel along +x, made direction-agnostic; the yaw terms stronger, as
        # commanded turns need them (at the forward weights commanded turns drift and turns in place fail)
        rewards["track_ang_vel"] = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"std": 0.25})
        rewards["yaw"] = RewardTermCfg(func=mdp.HeadingTrack, weight=-2.0, params={})
        rewards["lateral_position"] = RewardTermCfg(func=mdp.CrossTrack, weight=-1.50, params={})
        del rewards["lateral_velocity"]                     # the sideways speed is commanded now
        rewards["swing_progress"] = RewardTermCfg(func=mdp.SwingProgressOmni, weight=0.80, params={})
    if opt.stops:
        rewards["stand_still"] = RewardTermCfg(func=mdp.stand_still_l2, weight=-2.0, params={})
    if opt.sprint:
        # fast: tracked to a tolerance that grows with the speed; straight: the heading and the line held harder
        rewards["track_lin_vel"] = RewardTermCfg(func=mdp.track_lin_vel_rel_exp, weight=2.0,
                                                 params={"std": 0.10, "rel": opt.track_rel})
        rewards["yaw"] = RewardTermCfg(func=mdp.HeadingTrack, weight=opt.heading_weight, params={})
        rewards["lateral_position"] = RewardTermCfg(func=mdp.CrossTrack, weight=opt.cross_weight, params={})
        rewards["action_rate"] = RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01)
    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "tilt": TerminationTermCfg(func=mdp.tilt_beyond, params={"limit_rad": 0.7}),
        "body_contact": TerminationTermCfg(func=mdp.body_contact),
    }
    curriculum = {}
    if opt.noise_ramp and opt.noisy:
        curriculum["noise_ramp"] = CurriculumTermCfg(
            func=mdp.ramp_noise, params={"hold_iters": NOISE_HOLD_ITERS, "ramp_iters": NOISE_RAMP_ITERS})
    if opt.sprint and opt.fixed_command is None:
        curriculum["sprint_speed"] = CurriculumTermCfg(
            func=mdp.ramp_sprint_speed, params={"start_hi": opt.sprint_start_hi, "end_hi": opt.sprint_max,
                                                "hold_iters": opt.sprint_hold_iters,
                                                "ramp_iters": opt.sprint_ramp_iters})
    if opt.omni:
        hi = max(float(opt.speed_range[1]), opt.sprint_max if opt.sprint else 0.0)
        wz_hi = max(opt.wz_max, SPIN_RANGE[1]) if opt.stops else opt.wz_max
        commands = {"twist": mdp.OmniVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(1.0e6, 1.0e6) if opt.stops else tuple(opt.resample_s),
            heading_command=False, rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-hi, hi), lin_vel_y=(-hi, hi),
                                                     ang_vel_z=(-wz_hi, wz_hi)),
            speed_range=tuple(opt.speed_range), wz_max=opt.wz_max, rel_turning=opt.rel_turning,
            rel_spin=opt.rel_spin, fixed=None if opt.fixed_command is None else tuple(opt.fixed_command),
            schedules=opt.schedules if opt.stops else 0, schedule_steps=int(round(opt.episode_s / 0.020)),
            schedule_seed=opt.schedule_seed, forward_only=opt.sprint,
            slow_frac=opt.sprint_slow_frac if opt.sprint else 0.0)}
    else:
        commands = {"twist": UniformVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
            rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=tuple(opt.vx_range), lin_vel_y=(0.0, 0.0),
                                                     ang_vel_z=(0.0, 0.0)))}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=opt.num_envs, env_spacing=1.0,
                       entities={"robot": robot_cfg(arm.actuators, strip_visuals=not opt.visuals)}, sensors=sensors),
        observations=observations,
        actions={"servo": mdp.ServoActionCfg(scale=tuple(opt.action_scale))},
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base",
                            distance=0.6, elevation=-10.0, azimuth=90.0),
        sim=SimulationCfg(njmax=300, nconmax=None,
                          mujoco=MujocoCfg(timestep=arms_mod.PHYSICS_DT, iterations=10, ls_iterations=20)),
        decimation=DECIMATION,
        episode_length_s=opt.episode_s,
    )


def describe(opt: TaskOptions) -> dict:
    """Everything a run's saved config must name (the arm's parameters, the sources)."""
    from .sku2 import sources
    arm = arms_mod.get(opt.arm)
    return {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.__dict__.items()},
            "arm": arm.name, "arm_params": arm.params, "latency_steps": list(arm.latency_steps),
            "decimation": DECIMATION, "physics_dt": arms_mod.PHYSICS_DT, "sources": sources(getattr(opt, "masses_json", None))}
