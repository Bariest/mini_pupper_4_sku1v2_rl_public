"""SKU2's MDP terms for mjlab: the servo action, the robot-side observations, delays, rewards, terminations.

The action and the observations implement ``contract.py`` in torch (the same arithmetic; ``tests/test_contract.py``
compares them). Observation noise is applied here, from a per-environment amplitude table (``NoiseTable``) that the
noise ramp updates: mjlab caches a noise config's bounds as tensors on first use (``utils/noise/noise_cfg.py:27-36``),
so mutating ``n_min``/``n_max`` later, as MP2's ramp did under Isaac Lab, would do nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from .contract import (ABDUCTION, BAND_DEG, DD0, STATE_DT, STEP_LIMIT_DEG, Contract)
from .sku2 import FOOT_SITES, LEGS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

F32 = torch.float32
SAMPLE_EVERY = 5            # physics steps per STATE frame (5 ms at 1 ms physics)


# ------------------------------------------------------------------------------------------------ the contract, torch
class TorchContract:
    def __init__(self, device):
        c = Contract.load()
        self.c = c
        self.signs = torch.tensor(c.signs, dtype=F32, device=device)
        self.offsets = torch.tensor(c.offsets_deg, dtype=F32, device=device)
        self.stand = torch.tensor(c.stand_rel_rad, dtype=F32, device=device)

    def ticks_from_servo_deg(self, sdeg):
        ang = sdeg.to(F32) + self.offsets
        sig = 511 + torch.trunc(ang * 10240.0 / 3100.0)
        return sig.clamp(0, 1023)

    @staticmethod
    def dd_from_ticks(ticks):
        return torch.trunc(3100.0 - ticks.to(F32) * 3100.0 / 1024.0)

    def servo_deg_from_dd(self, dd):
        return (DD0 - dd) / 10.0 - self.offsets

    def quantise_rel(self, rel):
        sdeg = torch.rad2deg(rel) * self.signs
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(sdeg))
        return dd, torch.deg2rad(self.servo_deg_from_dd(dd)) * self.signs

    def measure_rel(self, rel):
        """A true position as the AT32 reports it: an integer deci-degree on the servo's track."""
        sdeg = torch.rad2deg(rel) * self.signs
        dd = torch.round(DD0 - 10.0 * (sdeg + self.offsets))
        return torch.deg2rad(self.servo_deg_from_dd(dd)) * self.signs


# ------------------------------------------------------------------------------------------------------ the action
@dataclass(kw_only=True)
class ServoActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    scale: tuple = (0.10, 0.40, 0.42) * 4          # rad per unit action, servo order (abduction, hip, knee) x 4
    step_limit_deg: float = STEP_LIMIT_DEG
    quantise: bool = True

    def build(self, env):
        return ServoAction(self, env)


class ServoAction(ActionTerm):
    """Policy action (12, servo order) -> targets on the abduction and hip joints and the knee tendons.

    Also samples the positions every 5 physics steps, as the robot's 200 Hz STATE stream, quantised to the AT32's
    deci-degree, for the position and velocity observations. Positions are computed from the joint angles (the knee:
    the tendon's coefficients), which are valid right after a reset write, before any forward pass."""

    cfg: ServoActionCfg

    def __init__(self, cfg: ServoActionCfg, env):
        super().__init__(cfg, env)
        dev = self.device
        self.tc = TorchContract(dev)
        c = self.tc.c
        names = []
        for leg in LEGS:
            names += [f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"]
        jid, jn = self._entity.find_joints(names, preserve_order=True)
        assert list(jn) == names, jn
        self._jid_all = torch.tensor(jid, device=dev, dtype=torch.long)
        target_j = [n for n in names if not n.endswith("_calf")]
        tj, _ = self._entity.find_joints(target_j, preserve_order=True)
        self._tj = torch.tensor(tj, device=dev, dtype=torch.long)
        tt, tn = self._entity.find_tendons([f"{leg}_knee_drive" for leg in LEGS], preserve_order=True)
        self._tt = torch.tensor(tt, device=dev, dtype=torch.long)
        self._jidx = torch.tensor([i for i in range(12) if i % 3 != 2], device=dev)   # servo index of each joint target
        self._kidx = torch.tensor([2, 5, 8, 11], device=dev)
        # knee tendon = coef_thigh * q_thigh + coef_calf * q_calf
        co = c.meta["tendon_coefs"]
        self._kc = torch.tensor([[co[leg][f"{leg}_thigh"], co[leg][f"{leg}_calf"]] for leg in LEGS], dtype=F32,
                                device=dev)
        self.scale = torch.tensor(cfg.scale, dtype=F32, device=dev).expand(12).clone()
        self.band = torch.deg2rad(torch.tensor(BAND_DEG, dtype=F32, device=dev))
        self.limit = math.radians(cfg.step_limit_deg)
        n = self.num_envs
        self._raw = torch.zeros(n, 12, device=dev)
        self.prev_rel = torch.zeros(n, 12, device=dev)
        self.applied_action = torch.zeros(n, 12, device=dev)
        self.target_model = self.tc.stand.expand(n, 12).clone()
        self.hist = torch.zeros(n, 6, 12, device=dev)
        self._substep = 0
        self._decimation = int(env.cfg.decimation)
        self._synced_at = -1
        self.reset(None)

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def true_rel(self) -> torch.Tensor:
        """The true positions (servo order, rel to the stand, rad), from the joint angles."""
        q = self._entity.data.joint_pos[:, self._jid_all].view(-1, 4, 3)
        knee = (q[:, :, 1:3] * self._kc).sum(-1)
        model = torch.stack([q[:, :, 0], q[:, :, 1], knee], dim=-1).view(-1, 12)
        return model - self.tc.stand

    def joint_state_from_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """Joint angles (hip, thigh, calf per leg) that put the servo coordinates at ``rel`` (the knee via its
        tendon); for tools that pose the robot directly."""
        model = (self.tc.stand + rel).view(-1, 4, 3)
        q = torch.empty_like(model)
        q[..., 0] = model[..., 0]
        q[..., 1] = model[..., 1]
        q[..., 2] = (model[..., 2] - self._kc[:, 0] * model[..., 1]) / self._kc[:, 1]
        return q.view(-1, 12)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        rel = torch.maximum(torch.minimum(self.scale * a, self.band), -self.band)
        rel = torch.maximum(torch.minimum(rel, self.prev_rel + self.limit), self.prev_rel - self.limit)
        if self.cfg.quantise:
            _, rel = self.tc.quantise_rel(rel)
        self.prev_rel = rel
        self.applied_action = rel / self.scale
        self.target_model = self.tc.stand + rel

    def apply_actions(self) -> None:
        # Within a policy step, apply_actions sees the state at 0, 1, ..., decimation-1 ms. The 200 Hz samples are the
        # states at 5, 10 and 15 ms here, and at the observation's own instant (20 ms) in ``sync``, so the newest
        # position is as fresh as the IMU (the chain delay carries the robot's sensing age).
        i = self._substep % self._decimation
        if i and i % SAMPLE_EVERY == 0:
            self._push(self.tc.measure_rel(self.true_rel()))
        self._substep += 1
        self._entity.set_joint_position_target(self.target_model[:, self._jidx], joint_ids=self._tj)
        self._entity.set_tendon_len_target(self.target_model[:, self._kidx], tendon_ids=self._tt)

    def _push(self, x, env_ids=None):
        if env_ids is None:
            self.hist = torch.cat([self.hist[:, 1:], x[:, None, :]], dim=1)
        else:
            self.hist[env_ids] = x[env_ids, None, :].expand(-1, 6, -1)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self.prev_rel[ids] = 0.0
        self.applied_action[ids] = 0.0
        self.target_model[ids] = self.tc.stand
        all_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._push(self.tc.measure_rel(self.true_rel()), all_ids)

    def sync(self) -> None:
        """Push the state at the observation's instant, once per policy step."""
        step = int(self._env.common_step_counter)
        if step != self._synced_at:
            self._synced_at = step
            self._push(self.tc.measure_rel(self.true_rel()))

    # the robot's own estimators -----------------------------------------------------------------------------------
    @property
    def pos_rel(self) -> torch.Tensor:
        return self.hist[:, -1]

    @property
    def velocity(self) -> torch.Tensor:
        return velocity_from_hist(self.hist)


def velocity_from_hist(h: torch.Tensor) -> torch.Tensor:
    """contract.VelocityFilter in torch: a median of 3, then the 4-sample least-squares slope (h: (n, 6, 12))."""
    med = torch.stack([h[:, k:k + 3].median(dim=1).values for k in range(4)], dim=1)       # (n, 4, 12)
    w = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=h.device, dtype=h.dtype) / (5.0 * STATE_DT)
    return (w[None, :, None] * med).sum(1)


def servo_action(env) -> ServoAction:
    return env.action_manager.get_term("servo")


# ------------------------------------------------------------------------------------------------- observation noise
NOISY = ("ang_vel", "gravity", "pos", "vel")


class NoiseTable:
    """Uniform noise half-widths per observation channel, a scalar or one value per element (N2); the ramp writes
    ``amp``. One per env object."""

    def __init__(self, env, start: dict, end: dict):
        dev = env.device
        self.start = {k: torch.as_tensor(start[k], dtype=torch.float32, device=dev) for k in NOISY}
        self.end = {k: torch.as_tensor(end[k], dtype=torch.float32, device=dev) for k in NOISY}
        self.amp = dict(self.start)

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        a = self.amp[name]
        if not bool((a > 0.0).any()):
            return x
        return x + (torch.rand_like(x) * 2.0 - 1.0) * a


def _noise(env) -> NoiseTable | None:
    return getattr(env, "mp3_noise", None)


def _maybe_noise(env, name, x, noisy):
    nt = _noise(env)
    return nt.apply(name, x) if (noisy and nt is not None) else x


# ------------------------------------------------------------------------------------------------- observations
def obs_ang_vel(env, noisy: bool = True):
    x = _sensor(env, "robot/imu_ang_vel")
    return _maybe_noise(env, "ang_vel", x, noisy)


def obs_gravity(env, noisy: bool = True):
    acc = _sensor(env, "robot/imu_lin_acc")
    g = -acc / acc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return _maybe_noise(env, "gravity", g, noisy)


def obs_command(env, command_name: str = "twist", omni: bool = False):
    cmd = env.command_manager.get_command(command_name)
    if omni:
        return cmd.clone()
    out = torch.zeros_like(cmd)
    out[:, 0] = cmd[:, 0]
    return out


def obs_pos(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "pos", term.pos_rel, noisy)


def obs_vel(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "vel", term.velocity, noisy)


def obs_prev_action(env):
    return servo_action(env).applied_action


def obs_true_lin_vel(env):
    return env.scene["robot"].data.root_link_lin_vel_b


def obs_base_height(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def obs_track_errors(env):
    """The critic's view of the omni tracking state (``HeadingTrack``, ``CrossTrack``, as of the last reward): the
    heading error (rad) and the cross-track error in the body frame (m, x 10)."""
    n = env.num_envs
    h = getattr(env, "mp3_heading_error", None)
    c = getattr(env, "mp3_cross_track", None)
    h = torch.zeros(n, device=env.device) if h is None else h
    c = torch.zeros(n, 2, device=env.device) if c is None else c
    return torch.cat([h[:, None], 10.0 * c], dim=-1)


def obs_foot_contact(env, sensor_name: str = "feet_contact"):
    return (env.scene[sensor_name].data.found > 0).float()


def _sensor(env, name):
    from mjlab.envs.mdp import builtin_sensor
    return builtin_sensor(env, sensor_name=name)


# ------------------------------------------------------------------------------------------------------ events
def reset_command_delay(env, env_ids, latency_steps: tuple, chain_ms: tuple = (8.0, 19.0)):
    """One sensing-to-command chain delay per environment and episode, added to each actuator's own latency."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lo, hi = int(round(chain_ms[0])), int(round(chain_ms[1]))
    chain = torch.randint(lo, hi + 1, (len(env_ids),), device=env.device)
    acts = robot.actuators
    assert len(acts) == len(latency_steps), (len(acts), latency_steps)
    for act, lat in zip(acts, latency_steps):
        act.set_lags(chain + int(lat), env_ids)
    if not hasattr(env, "mp3_chain_ms"):
        env.mp3_chain_ms = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.mp3_chain_ms[env_ids] = chain


# ------------------------------------------------------------------------------------------------------ rewards
def _yaw(quat_wxyz):
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def track_lin_vel_xy_exp(env, std: float, command_name: str = "twist"):
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-((v - cmd) ** 2).sum(-1) / std ** 2)


def track_lin_vel_rel_exp(env, std: float, rel: float, command_name: str = "twist"):
    """``track_lin_vel_xy_exp`` with a tolerance that grows with the commanded speed (the sprint): the kernel's width
    is ``max(std, rel * |command|)``, so being 5 cm/s short at 0.5 m/s earns as being 1 cm/s short at 0.1 m/s."""
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    width = torch.clamp(rel * cmd.norm(dim=-1), min=std)
    return torch.exp(-((v - cmd) ** 2).sum(-1) / width ** 2)


def track_ang_vel_z_exp(env, std: float, command_name: str = "twist"):
    w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-((w - cmd) ** 2) / std ** 2)


def lin_vel_z_l2(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(env):
    return (env.scene["robot"].data.root_link_ang_vel_b[:, :2] ** 2).sum(-1)


def torques_l2(env):
    return (env.scene["robot"].data.actuator_force ** 2).sum(-1)


def joint_acc_l2(env):
    return (env.scene["robot"].data.joint_acc ** 2).sum(-1)


def flat_orientation_l2(env):
    return (env.scene["robot"].data.projected_gravity_b[:, :2] ** 2).sum(-1)


def yaw_l2(env):
    return _yaw(env.scene["robot"].data.root_link_quat_w) ** 2


def lateral_position_l2(env):
    y = env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return y ** 2


def lateral_velocity_l2(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 1] ** 2


def _foot_site_ids(env):
    if not hasattr(env, "_mp3_foot_sites"):
        ids, _ = env.scene["robot"].find_sites(list(FOOT_SITES), preserve_order=True)
        env._mp3_foot_sites = torch.tensor(ids, device=env.device, dtype=torch.long)
    return env._mp3_foot_sites


def stance_foot_slip_l2(env, sensor_name: str = "feet_contact", threshold: float = 1.0):
    s = env.scene[sensor_name].data
    contact = (s.force.norm(dim=-1) > threshold).float()             # (n, 4)
    v = env.scene["robot"].data.site_lin_vel_w[:, _foot_site_ids(env), :2]
    return (contact * (v ** 2).sum(-1)).sum(-1)


class SwingProgressTouchdown(ManagerTermBase):
    """MP2's swing-progress term (mp2_x_walk_policy mdp.py): a lifted foot moving forward relative to the body earns,
    only if it was down within ``touchdown_window_s``; "down" is geometric (within ``planted_tolerance`` of the lowest
    foot)."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._since_down = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._since_down.zero_()
        else:
            self._since_down[env_ids] = 0.0

    def __call__(self, env, command_name="twist", lift_threshold=0.008, target_lift=0.020, reference_speed=0.15,
                 max_rewarded_feet=2.0, command_deadband=0.02, touchdown_window_s=0.40, planted_tolerance=0.005):
        d = env.scene["robot"].data
        ids = _foot_site_ids(env)
        pos = d.site_pos_w[:, ids, :]
        lift = pos[:, :, 2] - pos[:, :, 2].amin(dim=1, keepdim=True)
        down = lift < planted_tolerance
        self._since_down = torch.where(down, torch.zeros_like(self._since_down), self._since_down + env.step_dt)
        recently = (self._since_down <= touchdown_window_s).float()
        credit = ((lift - lift_threshold) / max(target_lift - lift_threshold, 1e-6)).clamp(0.0, 1.0)
        rel = d.site_lin_vel_w[:, ids, :2] - d.root_link_lin_vel_w[:, None, :2]
        yaw = _yaw(d.root_link_quat_w)
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)[:, None, :]
        progress = ((rel * fwd).sum(-1) / reference_speed).clamp(0.0, 1.0)
        r = (credit * progress * recently).sum(-1).clamp(max=max_rewarded_feet)
        cmd = env.command_manager.get_command(command_name)
        return r * (cmd[:, :2].norm(dim=-1) > command_deadband).float()


# ------------------------------------------------------------------------------------------- omni: commands, rewards
def _rot2(yaw, xy):
    """Rotate (n, ..., 2) vectors by yaw (n,)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < xy.dim() - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    return torch.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], dim=-1)


def _wrap(a):
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


class HeadingTrack(ManagerTermBase):
    """The heading error against the commanded yaw rate integrated from the episode's start (or from a fresh
    command): ``wrap(yaw - target)^2``. With wz = 0 it holds the heading (the forward task's ``yaw_l2``), and it
    penalises a drift that ``track_ang_vel`` alone forgives once it has happened."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._target = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True

    def __call__(self, env, command_name="twist"):
        yaw = _yaw(env.scene["robot"].data.root_link_quat_w)
        self._target = torch.where(self._fresh, yaw, self._target)
        self._fresh[:] = False
        self._target = _wrap(self._target + env.command_manager.get_command(command_name)[:, 2] * env.step_dt)
        err = _wrap(yaw - self._target)
        env.mp3_heading_error = err
        return err ** 2


class CrossTrack(ManagerTermBase):
    """The omni form of ``lateral_position_l2``: the body's distance from the commanded path, across the commanded
    direction of travel (the body-frame command turned by the body's yaw). Along-track lag is forgiven (re-anchored
    every step), so only the sideways drift accumulates; with no travel commanded the whole displacement counts."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._target = torch.zeros(env.num_envs, 2, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True

    def __call__(self, env, command_name="twist", min_speed=0.02):
        d = env.scene["robot"].data
        p = d.root_link_pos_w[:, :2]
        self._target = torch.where(self._fresh[:, None], p, self._target)
        self._fresh[:] = False
        v = _rot2(_yaw(d.root_link_quat_w), env.command_manager.get_command(command_name)[:, :2])
        self._target = self._target + v * env.step_dt
        e = p - self._target
        speed = v.norm(dim=-1, keepdim=True)
        u = v / speed.clamp_min(1e-6)
        moving = speed > min_speed
        e_perp = torch.where(moving, e - (e * u).sum(-1, keepdim=True) * u, e)
        self._target = torch.where(moving, p - e_perp, self._target)
        env.mp3_cross_track = _rot2(-_yaw(d.root_link_quat_w), e_perp)
        return (e_perp ** 2).sum(-1)


def stand_still_l2(env, command_name: str = "twist", deadband: float = 1e-3):
    """With no command (a stop), the joints' squared distance from the stand (rel 0, the firmware's stance)."""
    cmd = env.command_manager.get_command(command_name)
    idle = (cmd.abs().amax(dim=-1) < deadband).float()
    return idle * (servo_action(env).true_rel() ** 2).sum(-1)


class SwingProgressOmni(SwingProgressTouchdown):
    """``SwingProgressTouchdown`` with each foot's own commanded direction: a swing foot earns for moving, relative to
    the body, along the velocity the command asks of the body at that foot (``v + wz x r``, turned into the world by
    the body's yaw), so strafing and turning in place are credited as walking forward is."""

    def __call__(self, env, command_name="twist", lift_threshold=0.008, target_lift=0.020, reference_speed=0.15,
                 max_rewarded_feet=2.0, command_deadband=0.01, touchdown_window_s=0.40, planted_tolerance=0.005):
        d = env.scene["robot"].data
        ids = _foot_site_ids(env)
        pos = d.site_pos_w[:, ids, :]
        lift = pos[:, :, 2] - pos[:, :, 2].amin(dim=1, keepdim=True)
        down = lift < planted_tolerance
        self._since_down = torch.where(down, torch.zeros_like(self._since_down), self._since_down + env.step_dt)
        recently = (self._since_down <= touchdown_window_s).float()
        credit = ((lift - lift_threshold) / max(target_lift - lift_threshold, 1e-6)).clamp(0.0, 1.0)
        yaw = _yaw(d.root_link_quat_w)
        r_b = _rot2(-yaw, pos[:, :, :2] - d.root_link_pos_w[:, None, :2])               # (n, 4, 2), body frame
        cmd = env.command_manager.get_command(command_name)
        wz = cmd[:, 2:3]
        v_b = torch.stack([cmd[:, None, 0] - wz * r_b[..., 1], cmd[:, None, 1] + wz * r_b[..., 0]], dim=-1)
        v_w = _rot2(yaw, v_b)
        speed = v_w.norm(dim=-1)
        u = v_w / speed.clamp_min(1e-6).unsqueeze(-1)
        rel = d.site_lin_vel_w[:, ids, :2] - d.root_link_lin_vel_w[:, None, :2]
        progress = ((rel * u).sum(-1) / reference_speed).clamp(0.0, 1.0) * (speed > command_deadband).float()
        return (credit * progress * recently).sum(-1).clamp(max=max_rewarded_feet)


class OmniVelocityCommand(UniformVelocityCommand):
    """Joystick-like commands: the travel speed uniform in ``speed_range`` at a uniform angle; the yaw rate uniform in
    +-``wz_max`` for a ``rel_turning`` fraction (0 otherwise); a ``rel_spin`` fraction turn in place (no travel,
    |wz| in [``spin_min``, ``wz_max``]). ``fixed`` overrides all of it (evaluation): one (vx, vy, wz), or a list of
    them dealt out round-robin (env i gets ``fixed[i % len(fixed)]``).

    ``schedules`` > 0: the command instead follows a take-like keyframe schedule (``takes.schedule_bank``: walks, turns
    in place, stops, ramped in), indexed by the episode's step; each episode draws one of ``schedules`` schedules.
    ``set_bank`` replaces the bank and deals schedule i to env i (the clone's collection, which labels with the same
    schedules played through the firmware)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bank = None
        self._dealt = False
        self.schedule_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.fixed is None and cfg.schedules > 0:
            from .takes import schedule_bank
            cmd, _ = schedule_bank(cfg.schedules, cfg.schedule_steps, seed=cfg.schedule_seed)
            self.bank = torch.tensor(cmd, device=self.device)

    def set_bank(self, cmd):
        self.bank = torch.as_tensor(cmd, dtype=torch.float32, device=self.device)
        self._dealt = True
        self.schedule_id[:] = torch.arange(self.num_envs, device=self.device) % len(self.bank)
        self._update_command()

    def _update_command(self):
        if self.bank is not None:
            k = self._env.episode_length_buf.clamp(max=self.bank.shape[1] - 1)
            self.vel_command_b[:] = self.bank[self.schedule_id, k]
        else:
            super()._update_command()

    def _resample_command(self, env_ids):
        cfg = self.cfg
        n = len(env_ids)
        dev = self.device
        if self.bank is not None:
            if not self._dealt:
                self.schedule_id[env_ids] = torch.randint(len(self.bank), (n,), device=dev)
            self._update_command()
            return
        if cfg.fixed is not None:
            table = torch.tensor(cfg.fixed, dtype=torch.float32, device=dev).view(-1, 3)
            self.vel_command_b[env_ids] = table[env_ids % len(table)]
            return
        if cfg.forward_only:
            # the sprint: straight ahead, no turning; a ``slow_frac`` share at the walk's own speeds (``slow_range``)
            sp = torch.empty(n, device=dev).uniform_(*cfg.speed_range)
            slow = torch.empty(n, device=dev).uniform_(*cfg.slow_range)
            sp = torch.where(torch.rand(n, device=dev) < cfg.slow_frac, slow, sp)
            self.vel_command_b[env_ids, 0] = sp
            self.vel_command_b[env_ids, 1] = 0.0
            self.vel_command_b[env_ids, 2] = 0.0
            return
        sp = torch.empty(n, device=dev).uniform_(*cfg.speed_range)
        th = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        wz = torch.empty(n, device=dev).uniform_(-cfg.wz_max, cfg.wz_max)
        wz = torch.where(torch.rand(n, device=dev) < cfg.rel_turning, wz, torch.zeros_like(wz))
        spin = torch.rand(n, device=dev) < cfg.rel_spin
        sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        wz = torch.where(spin, sign * torch.empty(n, device=dev).uniform_(cfg.spin_min, cfg.wz_max), wz)
        sp = torch.where(spin, torch.zeros_like(sp), sp)
        self.vel_command_b[env_ids, 0] = sp * torch.cos(th)
        self.vel_command_b[env_ids, 1] = sp * torch.sin(th)
        self.vel_command_b[env_ids, 2] = wz


@dataclass(kw_only=True)
class OmniVelocityCommandCfg(UniformVelocityCommandCfg):
    speed_range: tuple = (0.05, 0.12)
    wz_max: float = 0.6
    rel_turning: float = 0.5
    rel_spin: float = 0.1
    spin_min: float = 0.2
    fixed: tuple | None = None
    schedules: int = 0
    schedule_steps: int = 500
    schedule_seed: int = 0
    forward_only: bool = False          # the sprint: vx in speed_range (or slow_range), vy = wz = 0
    slow_frac: float = 0.0
    slow_range: tuple = (0.05, 0.12)

    def build(self, env):
        return OmniVelocityCommand(self, env)


# -------------------------------------------------------------------------------------------------- terminations
def body_contact(env, sensor_name: str = "body_contact"):
    return (env.scene[sensor_name].data.found > 0).any(dim=-1)


def tilt_beyond(env, limit_rad: float):
    g = env.scene["robot"].data.projected_gravity_b
    return torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) > limit_rad


# --------------------------------------------------------------------------------------------------- curriculum
def ramp_sprint_speed(env, env_ids, start_hi: float, end_hi: float, hold_iters: int, ramp_iters: int,
                      command_name: str = "twist", steps_per_iter: int = 24):
    """The sprint's speed curriculum: the top commanded speed held at ``start_hi``, then raised linearly to
    ``end_hi`` (the next episodes' commands; the command term reads its ``speed_range`` at every draw)."""
    it = env.common_step_counter // steps_per_iter
    frac = 0.0 if it <= hold_iters else min(1.0, (it - hold_iters) / max(ramp_iters, 1))
    hi = start_hi + frac * (end_hi - start_hi)
    term = env.command_manager.get_term(command_name)
    lo = float(term.cfg.speed_range[0])
    term.cfg.speed_range = (lo, max(lo, hi))
    return hi


def ramp_noise(env, env_ids, hold_iters: int, ramp_iters: int, steps_per_iter: int = 24):
    """MP2's noise ramp: hold the start amplitudes, then move linearly to the end amplitudes."""
    nt = _noise(env)
    if nt is None:
        return 0.0
    it = env.common_step_counter // steps_per_iter
    frac = 0.0 if it <= hold_iters else min(1.0, (it - hold_iters) / max(ramp_iters, 1))
    for k in NOISY:
        nt.amp[k] = nt.start[k] + frac * (nt.end[k] - nt.start[k])
    return frac
