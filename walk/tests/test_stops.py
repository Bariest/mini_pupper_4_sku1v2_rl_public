"""Stops: the takes' schedules through the firmware's leg phases (against the recordings when present), the random
take-like schedules, and the schedule command in the environment."""
import glob
import json

import numpy as np
import pytest
import torch

from mp3_walk_policy import paths, takes
from mp3_walk_policy.contract import Contract
from mp3_walk_policy.env_cfg import make_env_cfg, stops_options
from mp3_walk_policy.gates import TAKES_JSON, take_schedules

TAKE_CSVS = sorted(glob.glob(str(paths.MPX_SIM / "mpxrec_000cb09b-omnis" / "*.csv")))


@pytest.mark.skipif(not TAKE_CSVS, reason="the recordings are not checked out")
def test_takes_json_is_the_recordings():
    d = json.loads(TAKES_JSON.read_text())["takes"]
    assert len(d) == len(TAKE_CSVS)
    for p in TAKE_CSVS:
        v = next(v for v in d.values() if v["file"] == p.replace("\\", "/").split("/")[-1])
        assert v["sha256"] == paths.sha256(p)
        assert v["keyframes"] == takes.read_take(p)[0]["keyframes"]


@pytest.mark.skipif(not TAKE_CSVS, reason="the recordings are not checked out")
def test_firmware_legs_replay_the_recordings():
    """Every recorded tick within one of the host gait's through the leg phases (the rest is ms timestamp jitter)."""
    from mp3_walk_policy.firmware_gait import Gait
    for p in TAKE_CSVS:
        head, take = takes.read_take(p)
        best = 0.0
        for t0 in range(195, 215):
            g = Gait(head["trim_deg"])
            legs = takes.FirmwareLegs(g, head["height_mm"])
            out = []
            for i, t in enumerate(take["t"]):
                if t < t0:
                    continue
                m = takes.schedule(head["keyframes"], t - t0, head["height_mm"])
                if m is None:
                    break
                out.append((i, g.ticks(legs.tick(t, m))))
            idx = np.array([i for i, _ in out])
            err = np.abs(np.array([x for _, x in out]) - take["cmd"][idx]).max(axis=1)
            best = max(best, float((err <= 1).mean()))
        assert best >= 0.98, (p, best)


def test_stops_reach_the_stance_and_restart_from_reset():
    c = Contract.load()
    kfs = [takes.keyframe(vx=60, hold_ms=1000), takes.keyframe(stand=True, hold_ms=500),
           takes.keyframe(vy=60, hold_ms=1000)]
    cmd, rel = takes.firmware_rel(kfs, 150)
    stop = (cmd == 0).all(axis=1)
    hold = np.arange(150)[stop][int(takes.RAMP_OUT_MS / 20) + 1:]                   # after the feet came home
    assert len(hold) > 0 and np.abs(rel[hold] - c.rel_from_ticks(c.stand_ticks)).max() < 1e-9
    # the walk after the stop is the gait from its reset (a fresh gait at the stance), its command ramping from zero
    from mp3_walk_policy.firmware_gait import Gait
    restart = takes.duration_ms(kfs[:2])
    j0 = -(-restart // takes.TICK_MS)                                               # the first walking tick
    pick = (4 * np.arange(150)) // 3
    ks = np.nonzero(pick >= j0)[0][:12]
    g = Gait(c.offsets_deg)
    legs = takes.FirmwareLegs(g)
    fresh = {}
    for j in range(j0, pick[ks[-1]] + 1):
        t = j * takes.TICK_MS
        fresh[j] = g.ticks(legs.tick(t, takes.schedule(kfs, t)))
    assert (cmd[ks[0] - 1] == 0).all() and 0 < cmd[ks[0], 1] < cmd[ks[-1], 1]
    for k in ks:
        assert np.array_equal(c.rel_from_ticks(fresh[pick[k]][None])[0], rel[k]), k


def test_schedule_commands_are_firmware_rels():
    rng = np.random.default_rng(3)
    for _ in range(5):
        kfs = takes.random_schedule(rng)
        a, _ = takes.firmware_rel(kfs, 500)
        assert np.allclose(takes.schedule_commands(kfs, 500), a, atol=1e-12)


def test_random_schedules():
    rng = np.random.default_rng(0)
    for _ in range(200):
        kfs = takes.random_schedule(rng)
        assert takes.duration_ms(kfs) >= 10000
        st = [k["stand"] for k in kfs]
        assert not any(a and b for a, b in zip(st, st[1:]))
        for k in kfs:
            c = takes.command_of(k)
            sp = np.hypot(c[0], c[1])
            if k["stand"]:
                assert (c == 0).all()
            elif sp < 1e-9:
                assert takes.SPIN_RANGE[0] - 1e-9 <= abs(c[2]) <= takes.SPIN_RANGE[1] + 1e-9
            else:
                assert takes.SPEED_RANGE[0] - 1e-9 <= sp <= takes.SPEED_RANGE[1] + 1e-9
                assert abs(c[2]) <= takes.WALK_WZ_MAX + 1e-9
    assert set(take_schedules()) == {f"000cb09b-{i}" for i in range(1, 5)}


def test_schedule_command_in_env():
    """The command is the bank's row for the env's schedule at the episode's step, observed whole; the stand-still
    penalty is zero at the stand and counts only with no command."""
    from mjlab.envs import ManagerBasedRlEnv
    from mp3_walk_policy import mdp
    e = ManagerBasedRlEnv(cfg=make_env_cfg(stops_options(num_envs=8, noisy=False, randomise=False, noise_ramp=False,
                                                         schedules=8)), device="cuda:0")
    term = e.command_manager.get_term("twist")
    bank = np.zeros((8, 500, 3), dtype=np.float32)
    bank[:, :, 0] = np.linspace(0, 0.1, 500)
    bank[::2, :50] = 0.0                                     # even envs stopped for the first 50 steps
    term.set_bank(bank)
    obs, _ = e.reset()
    assert torch.equal(obs["actor"][:, 6:9], torch.tensor(bank[:, 0], device=e.device))
    assert float(mdp.stand_still_l2(e).abs().max()) < 1e-6
    a = torch.zeros(8, 12, device=e.device)
    a[:, 1] = 0.5
    for _ in range(10):
        obs, *_ = e.step(a)
    k = int(e.episode_length_buf[0])
    assert torch.allclose(obs["actor"][:, 6:9], torch.tensor(bank[:, k], device=e.device))
    pen = mdp.stand_still_l2(e).cpu().numpy()
    e.close()
    assert (pen[::2] > 0.01).all() and (pen[1::2] == 0).all()
