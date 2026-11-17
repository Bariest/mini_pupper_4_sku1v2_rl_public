"""The omni task: the teacher table against the firmware gait, its interpolation, the joystick commands and their
observation, the contract's command forms."""
import json
import math

import numpy as np
import pytest
import torch

from mp3_walk_policy import teacher
from mp3_walk_policy.contract import GAIT_CYCLE_STEPS, OBS_SIZE, assemble_observation
from mp3_walk_policy.env_cfg import OMNI_SPEED_RANGE, OMNI_WZ_MAX, make_env_cfg, omni_options

N = 256


@pytest.mark.xfail(reason="the committed table was built with MSVC; a gcc build differs by one tick in 62 of 437400 "
                          "entries (max 5.28 mrad), same sources", strict=False)
def test_omni_table_is_the_firmware_gait():
    """The committed table is a fresh build from the current sources, tick for tick."""
    d = np.load(teacher.OMNI_TABLE)
    t = teacher.build_omni_table()
    assert np.array_equal(d["rel"], t["rel"])
    assert json.loads(str(d["sources"])) == t["sources"]
    assert t["rel"].shape == (len(teacher.OMNI_VXY_GRID_MM_S),) * 2 + (len(teacher.OMNI_WZ_GRID_RAD_S),
                                                                        GAIT_CYCLE_STEPS, 12)


def test_omni_table_forward_row_is_the_forward_table():
    """At vy = wz = 0 the omni table is the forward teacher at the speeds both tabulate."""
    o, f = np.load(teacher.OMNI_TABLE), np.load(teacher.TABLE)
    jy = list(o["vxy_mm_s"]).index(0.0)
    jw = int(np.argmin(np.abs(o["wz_rad_s"])))
    assert abs(o["wz_rad_s"][jw]) < 1e-9
    for k, vx in enumerate(f["vx_mm_s"]):
        if vx in o["vxy_mm_s"]:
            i = list(o["vxy_mm_s"]).index(vx)
            assert np.allclose(o["rel"][i, jy, jw], f["rel"][k], atol=1e-6), vx


def test_omni_interpolation_hits_the_grid():
    T = teacher.TorchOmniTeacher("cpu")
    d = np.load(teacher.OMNI_TABLE)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i, j, k = (rng.integers(len(d["vxy_mm_s"])), rng.integers(len(d["vxy_mm_s"])),
                   rng.integers(len(d["wz_rad_s"])))
        cmd = torch.tensor([[d["vxy_mm_s"][i] / 1000, d["vxy_mm_s"][j] / 1000, d["wz_rad_s"][k]]] * GAIT_CYCLE_STEPS,
                           dtype=torch.float32)
        got = T.target_rel(cmd, torch.arange(GAIT_CYCLE_STEPS)).numpy()
        assert np.allclose(got, d["rel"][i, j, k], atol=1e-5)


def test_assemble_observation_command_forms():
    z = np.zeros(12)
    fwd = assemble_observation(np.zeros(3), (0, 0, -1), 0.1, z, z, z)
    omni = assemble_observation(np.zeros(3), (0, 0, -1), (0.05, -0.08, 0.3), z, z, z)
    assert fwd.shape == omni.shape == (OBS_SIZE,)
    assert fwd[6:9].tolist() == pytest.approx([0.1, 0.0, 0.0])
    assert omni[6:9].tolist() == pytest.approx([0.05, -0.08, 0.3])


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv
    e = ManagerBasedRlEnv(cfg=make_env_cfg(omni_options(num_envs=N, noisy=False, randomise=False,
                                                        noise_ramp=False)), device="cuda:0")
    e.reset()
    yield e
    e.close()


def test_omni_commands_and_observation(env):
    cmd = env.command_manager.get_command("twist")
    sp = cmd[:, :2].norm(dim=-1)
    spin = sp < 1e-6
    assert bool(((sp >= OMNI_SPEED_RANGE[0] - 1e-6) & (sp <= OMNI_SPEED_RANGE[1] + 1e-6) | spin).all())
    assert bool((cmd[:, 2].abs() <= OMNI_WZ_MAX + 1e-6).all())
    assert bool((cmd[spin, 2].abs() >= 0.2 - 1e-6).all()) and 0 < int(spin.sum()) < N // 4
    ang = torch.atan2(cmd[~spin, 1], cmd[~spin, 0])
    assert len(torch.unique(torch.floor((ang + math.pi) / (math.pi / 4)))) == 8      # every octant
    obs = env.observation_manager.compute()
    for g in ("actor", "anchor", "critic"):
        assert torch.equal(obs[g][:, 6:9], cmd)


def test_fixed_commands_round_robin():
    from mjlab.envs import ManagerBasedRlEnv
    C = ((0.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 0.4))
    e = ManagerBasedRlEnv(cfg=make_env_cfg(omni_options(num_envs=6, noisy=False, randomise=False, noise_ramp=False,
                                                        fixed_command=C)), device="cuda:0")
    e.reset()
    got = e.command_manager.get_command("twist").cpu().numpy()
    e.close()
    assert np.allclose(got, np.array(C * 2), atol=1e-7)


def test_omni_options_scale():
    o = omni_options()
    assert o.omni and o.action_scale[:3] == (0.32, 0.45, 0.42)
