"""The SKU2 environment: the parameter echo (what the compiled model and the live buffers really run), the stand."""
import math

import mujoco
import numpy as np
import pytest
import torch

from mp3_walk_policy import arms
from mp3_walk_policy.env_cfg import TaskOptions, make_env_cfg

N = 64


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv
    e = ManagerBasedRlEnv(cfg=make_env_cfg(TaskOptions(num_envs=N)), device="cuda:0")
    e.reset()
    yield e
    e.close()


def _id(m, kind, name):
    return mujoco.mj_name2id(m, kind, name)


def test_masses(env):
    m = env.sim.mj_model
    assert abs(m.body_mass.sum() - 0.450) < 1e-9
    for leg in ("FR", "FL", "RR", "RL"):
        assert abs(m.body_mass[_id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{leg}_hip")] - 0.06682) < 1e-12


def test_stock_actuator_echo(env):
    m = env.sim.mj_model
    p = arms.stock_profile()
    names = [m.actuator(i).name for i in range(m.nu)]
    assert len(names) == 12
    for i, n in enumerate(names):
        assert m.actuator_gainprm[i, 0] == pytest.approx(p["kp"])
        assert m.actuator_biasprm[i, 1] == pytest.approx(-p["kp"])
        assert m.actuator_biasprm[i, 2] == pytest.approx(-p["kv"])
        assert m.actuator_forcelimited[i] and tuple(m.actuator_forcerange[i]) == pytest.approx((-p["umax"], p["umax"]))
        knee = n.endswith("_knee_drive")
        assert m.actuator_trntype[i] == (mujoco.mjtTrn.mjTRN_TENDON if knee else mujoco.mjtTrn.mjTRN_JOINT)
    for leg in ("FR", "FL", "RR", "RL"):
        for j in ("hip", "thigh"):
            dof = m.jnt_dofadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_{j}")]
            assert m.dof_armature[dof] == pytest.approx(p["armature"])
            assert m.dof_damping[dof] == pytest.approx(p["damping"])
            assert m.dof_frictionloss[dof] == pytest.approx(0.003)
        calf = m.jnt_dofadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_calf")]
        assert (m.dof_armature[calf], m.dof_damping[calf], m.dof_frictionloss[calf]) == pytest.approx((2e-5, 5e-4, 1e-3))
        t = _id(m, mujoco.mjtObj.mjOBJ_TENDON, f"robot/{leg}_knee_drive")
        assert m.tendon_armature[t] == pytest.approx(p["armature"])
        assert m.tendon_damping[t] == pytest.approx(p["damping"])
        assert m.tendon_frictionloss[t] == pytest.approx(0.0)
        assert m.geom_priority[_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{leg}_foot")] == 1
    assert m.opt.timestep == pytest.approx(0.001)


def test_delays_held_and_shared(env):
    """One chain delay per env, the same in both actuator groups, latency + chain within its range, and held."""
    acts = env.scene["robot"].actuators
    lags = [a._delay_buffer.current_lags.clone() for a in acts]
    lat = arms.stock().latency_steps
    for l, la in zip(lags, lat):
        assert int(l.min()) >= la + 8 and int(l.max()) <= la + 19
    assert torch.equal(lags[0] - lat[0], lags[1] - lat[1])
    assert len(torch.unique(lags[0])) > 3                      # drawn per environment
    a = torch.zeros(N, 12, device="cuda:0")
    for _ in range(5):
        env.step(a)
    for l, act in zip(lags, acts):
        assert torch.equal(l, act._delay_buffer.current_lags)  # never resampled within the episode


def test_stand(env):
    env.reset()
    a = torch.zeros(N, 12, device="cuda:0")
    for _ in range(100):                                       # 2 s
        obs, *_ = env.step(a)
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2]
    g = robot.data.projected_gravity_b
    assert torch.all(torch.abs(g[:, :2]).norm(dim=-1) < 0.01)
    assert 0.068 < float(z.mean()) < 0.076
    f = env.scene["feet_contact"].data.force[..., 2].abs().reshape(N, 4)   # FR, FL, RR, RL
    rear = (f[:, 2:].sum(-1) / f.sum(-1)).mean()
    assert abs(float(rear) - 0.566) < 0.015
    actor = obs["actor"]
    assert actor.shape == (N, 45)
    clean = env.observation_manager.compute()["anchor"]
    assert torch.all(clean[:, 3:6].sub(torch.tensor([0, 0, -1.0], device=clean.device)).abs() < 0.02)
    assert float(clean[:, 9:21].abs().max()) < math.radians(3.0)          # standing near the stand pose


def test_command_reaches_ctrl_after_latency_plus_chain(env):
    """A step in the action reaches the position actuator's ctrl exactly lag physics steps later (lag = the arm's
    latency + the chain delay, held per episode)."""
    import mujoco as mj
    from mp3_walk_policy.mdp import servo_action
    env.reset()
    a = torch.zeros(N, 12, device="cuda:0")
    for _ in range(3):                                      # fill the delay buffers with the stand target
        env.step(a)
    lag = 7
    for act in env.scene["robot"].actuators:
        act.set_lags(torch.full((N,), lag, dtype=torch.long, device="cuda:0"))
    m = env.sim.mj_model
    idx = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, "robot/FR_thigh")
    before = float(env.sim.data.ctrl[0, idx])
    a[:, 1] = 0.5
    env.action_manager.process_action(a)
    seen = []
    for i in range(env.cfg.decimation):
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        seen.append(float(env.sim.data.ctrl[0, idx]))
        env.sim.step()
    target = float(servo_action(env).target_model[0, 1])
    first = next(i for i, v in enumerate(seen) if abs(v - before) > 1e-6)
    assert first == lag and abs(seen[-1] - target) < 1e-6


def test_position_observation_is_fresh(env):
    """After a step, the newest position sample is the state at the observation's instant (as fresh as the IMU)."""
    from mp3_walk_policy.mdp import servo_action
    a = torch.zeros(N, 12, device="cuda:0")
    env.step(a)
    term = servo_action(env)
    env.observation_manager.compute()
    assert torch.allclose(term.hist[:, -1], term.tc.measure_rel(term.true_rel()), atol=1e-6)
