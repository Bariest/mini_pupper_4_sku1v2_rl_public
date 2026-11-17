"""The BAM arm's parameter echo: what the compiled model and the actuators really run."""
import mujoco
import pytest
import torch

from bam.model import load_model

from mp3_walk_policy import arms
from mp3_walk_policy.env_cfg import TaskOptions, make_env_cfg

N = 8


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv
    e = ManagerBasedRlEnv(cfg=make_env_cfg(TaskOptions(arm="bam_published", num_envs=N)), device="cuda:0")
    e.reset()
    yield e
    e.close()


def test_bam_actuators_echo(env):
    m = env.sim.mj_model
    arm = arms.get("bam_published")
    acts = env.scene["robot"].actuators
    assert [type(a).__name__ for a in acts] == ["BamServo"] * 3
    J = {role: float(load_model(motor_name=motor, model="m6").actuator.get_extra_inertia())
         for role, _, motor, _ in arms.BAM_ROLES}
    for leg in ("FR", "FL", "RR", "RL"):
        for joint, role in (("hip", "abd"), ("thigh", "hip")):
            dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_{joint}")]
            assert m.dof_armature[dof] == pytest.approx(J[role])
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"robot/{leg}_knee_drive")
        assert m.tendon_armature[t] == pytest.approx(J["knee"])
        assert m.tendon_damping[t] == 0.0 and m.tendon_frictionloss[t] == 0.0
    for i in range(m.nu):                                   # torque actuators without limits
        assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
        assert not m.actuator_forcelimited[i] and not m.actuator_ctrllimited[i]
    for a in acts:                                          # the identified configuration
        act = a._bam_model.actuator                         # kp and vin become per-env tensors while stepping
        assert (a._base_kp, act.kd, act.max_pwm) == (80.0, 400.0, 0.99)
        assert torch.all(a.vin_tensor == 12.0)
        assert not a.cfg.stiff_frictionloss
    lat = [int(a._delay_buffer.current_lags[0]) for a in acts]
    chain = lat[0] - arm.latency_steps[0]
    assert [l - chain for l in lat] == list(arm.latency_steps) == [2, 4, 2]
    assert 8 <= chain <= 19


def test_bam_writes_tendon_friction_per_env(env):
    a = torch.zeros(N, 12, device="cuda:0")
    for _ in range(10):
        env.step(a)
    model = env.sim.model
    fl = model.tendon_frictionloss
    assert fl.stride(0) != 0                                # expanded per world
    knee = env.scene["robot"].actuators[2]
    vals = fl[:, knee._tendons]
    assert torch.all(vals > 0)                              # the BAM budget, not the XML's 0


def test_bam_delays_from_each_set():
    """Each role's command delay comes from its own set's fitted command_delay, rounded up to the 1 ms step
    (the published sets and the refit T5 ub06 differ)."""
    from mp3_walk_policy import arms
    assert arms.get("bam_published").latency_steps == (2, 4, 2)
    assert arms.get("bam_t5ub06").latency_steps == (1, 5, 1)
