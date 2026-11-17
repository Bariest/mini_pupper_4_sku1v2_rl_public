"""The mirror map (mp3_walk_policy.symmetry): mirroring twice is the identity; two envs started at the stand, the
second given the mirror image of the first's command and actions, stay mirror images of each other in every actor and
critic observation (history, contacts and foot heights included); the body scan's permutation mirrors the ray grid."""
import pytest
import torch

from mp3_walk_policy import symmetry


def _env(history=2):
    from mjlab.envs import ManagerBasedRlEnv
    from mp3_walk_policy.terrain_task import TerrainOptions, make_terrain_env_cfg
    opt = TerrainOptions(arm="bam_t5ub06", num_envs=2, terrain="flat", noisy=False, randomise=False, history=history,
                         fixed_command=(0.10, 0.0, 0.0), push_mps=0.0, level=0)
    return ManagerBasedRlEnv(cfg=make_terrain_env_cfg(opt), device="cuda:0")


def test_mirror_twice_is_identity_and_scan_mirrors_the_grid():
    env = _env()
    m = symmetry.maps(env)
    for g in ("actor", "critic", "actions"):
        n = m[g][0].numel()
        x = torch.randn(5, n, device="cuda:0")
        assert torch.allclose(symmetry.mirror(symmetry.mirror(x, m[g]), m[g]), x)
    off, _ = env.scene["body_scan"].cfg.pattern.generate_rays(None, "cpu")
    p = symmetry._scan_perm(env)
    assert torch.allclose(off[p, 0], off[:, 0]) and torch.allclose(off[p, 1], -off[:, 1])
    env.close()


def test_mirrored_env_stays_the_mirror_image():
    """Small random actions for 4 steps from the stand (larger ones tip the robot and the two runs diverge); per term
    the mirrored observation agrees within 15 % of its size (the two runs drift apart through contact events and
    position quantisation; the model is near-symmetric, base COM y -0.16 mm), where a sign or swap error gives
    100-200 %."""
    torch.manual_seed(0)
    env = _env()
    obs, _ = env.reset()
    m = symmetry.maps(env)
    om = env.observation_manager
    cmd = env.command_manager.get_term("twist")
    c0 = torch.tensor([0.10, 0.03, 0.25], device="cuda:0")
    c1 = torch.tensor([0.10, -0.03, -0.25], device="cuda:0")
    cmd.vel_command_b[0], cmd.vel_command_b[1] = c0, c1
    for k in range(4):
        a0 = 0.2 * torch.randn(1, 12, device="cuda:0")
        obs, *_ = env.step(torch.cat([a0, symmetry.mirror(a0, m["actions"])], 0))
        cmd.vel_command_b[0], cmd.vel_command_b[1] = c0, c1
    for g in ("actor", "critic"):
        e = obs[g][1] - symmetry.mirror(obs[g][0:1], m[g])[0]
        off = 0
        for name, dims in zip(om.active_terms[g], om.group_obs_term_dim[g]):
            s = 1
            for v in dims:
                s *= int(v)
            x = obs[g][0, off:off + s]       # a floor of 3 mm per float: on flat ground the scan is ~1 mm everywhere
            assert float(e[off:off + s].norm()) < 0.15 * float(x.norm()) + 0.003 * s ** 0.5, (g, name)
            off += s
    env.close()


def test_heading_error_mirrors_to_its_negative():
    """The heading loop's error is a yaw angle: every history frame changes sign, in place (CPU, no simulation)."""
    perm, sign = symmetry._term_map(None, "heading_error", 10)
    assert perm == list(range(10)) and sign == [-1.0] * 10
