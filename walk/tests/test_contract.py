"""The contract: firmware units, coordinates, the action decoder and the velocity filter, numpy against torch."""
import json

import numpy as np
import pytest
import torch

from mp3_walk_policy import paths
from mp3_walk_policy.contract import (DD0, ActionDecoder, Contract, VelocityFilter, STEP_LIMIT_DEG)
from mp3_walk_policy.mdp import TorchContract, velocity_from_hist

C = Contract.load()


def test_stand_ticks_and_dd_match_the_stand():
    """servo_write(0 + offset) gives the contract's stand ticks and dd (its own formula, main.c:2526)."""
    ticks = C.ticks_from_servo_deg(np.zeros(12))
    assert ticks.tolist() == C.stand_ticks.tolist()
    assert C.dd_from_ticks(ticks).tolist() == C.stand_dd.tolist()


def test_contract_sources_unchanged():
    for rel, digest in C.meta["sources"].items():
        assert rel.startswith("mpx-sim/"), rel
        assert paths.sha256(paths.MPX_SIM / rel.removeprefix("mpx-sim/")) == digest, rel


def test_dd_inverse_is_exact_on_the_tick_grid():
    """A dd produced from a tick reads back within half a tick (0.15 deg) of the angle that produced it."""
    rng = np.random.default_rng(1)
    s = rng.uniform(-40, 40, (200, 12))
    dd = C.dd_from_ticks(C.ticks_from_servo_deg(s))
    back = C.servo_deg_from_dd(dd)
    assert np.abs(back - s).max() < 3100 / 10240 + 0.11   # one tick (truncation) plus the dd truncation
    assert abs(DD0 - 1553.02734375) < 1e-12


def test_torch_matches_numpy_quantisation():
    tc = TorchContract("cpu")
    rng = np.random.default_rng(2)
    rel = rng.uniform(-0.6, 0.6, (500, 12))
    dd_np, q_np = zip(*(C.quantise_rel(r) for r in rel))
    dd_t, q_t = tc.quantise_rel(torch.tensor(rel, dtype=torch.float32))
    assert np.array_equal(np.array(dd_np), dd_t.numpy().astype(np.int64))
    assert np.abs(np.array(q_np) - q_t.numpy()).max() < 1e-5


def test_decoder_step_limit_band_and_quantisation():
    d = ActionDecoder(C, scale=[1.0] * 12)
    out = d(np.full(12, 1.0))                       # 57 deg asked; 20 deg allowed in one step
    deg = np.degrees(out["rel"] * C.signs) * C.signs
    assert np.all(np.degrees(np.abs(out["rel"])) <= STEP_LIMIT_DEG + 0.31)
    out2 = d(np.full(12, 1.0))
    assert np.all(np.degrees(np.abs(out2["rel"] - out["rel"])) <= STEP_LIMIT_DEG + 0.31)
    for _ in range(5):
        out = d(np.full(12, 1.0))
    ab = [0, 3, 6, 9]
    assert np.all(np.degrees(np.abs(out["rel"][ab])) <= 20.0 + 0.31)      # the abduction's band
    assert np.allclose(out["action"], out["rel"])                          # scale 1


def test_velocity_filter_ramp_and_spike():
    vf = VelocityFilter()
    t = np.arange(40) * 0.005
    rng = np.random.default_rng(3)
    x = 0.8 * t[:, None] * np.ones(12) + rng.normal(0, 1e-4, (40, 12))
    x[20, 3] += 0.05                                # a one-sample chatter spike
    out = []
    for k in range(40):
        vf.push(x[k])
        out.append(vf.velocity())
    out = np.array(out)
    assert np.abs(out[10:, 0] - 0.8).max() < 0.05          # a ramp reads its slope
    # The median replaces the spike by a neighbouring sample: on a ramp the error is one sample's increment (here
    # 0.8 rad/s x 5 ms = 4 mrad, weighted up to 1.5 / 25 ms), against 0.05 rad x 60 = 3 rad/s without the median.
    assert np.abs(out[18:26, 3] - 0.8).max() < 0.35


def test_tick_centre_requantises_to_itself():
    """Every tick, for every servo's offset: the target at the tick's centre goes back to exactly that tick, in numpy
    and in torch (the dd's AT32 angle, converted back, would move ticks below 511 by one)."""
    ticks = np.tile(np.arange(1024)[:, None], (1, 12))
    assert np.array_equal(C.ticks_from_servo_deg(C.servo_deg_of_tick(ticks)), ticks)
    tc = TorchContract("cpu")
    dd_t, _ = tc.quantise_rel(torch.tensor(C.rel_from_ticks(ticks), dtype=torch.float32))
    assert np.array_equal(dd_t.numpy().astype(np.int64), C.dd_from_ticks(ticks))


def test_teacher_decodes_to_firmware_ticks():
    """The teacher's every target, decoded as the simulator does, is the dd the firmware sends. The table was built on
    the calibration of 2026-10-04 (its ticks include those offsets), listed in the contract as superseded."""
    import dataclasses
    from mp3_walk_policy.teacher import TABLE
    d = np.load(TABLE)
    ticks = d["ticks"][:, d["pick"]]                       # (n_vx, 18, 12)
    built = next(x for x in C.meta["calibrations_superseded"] if x["until"] == "2026-10-12")
    c = dataclasses.replace(C, offsets_deg=np.array(built["offsets_deg"]))
    dd, _ = c.quantise_rel(d["rel"].reshape(-1, 12))
    assert np.array_equal(dd, C.dd_from_ticks(ticks.reshape(-1, 12)))


def test_sim_velocity_code_matches_numpy_filter():
    vf = VelocityFilter()
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 0.01, (30, 12)), axis=0)
    hist = None
    for k in range(30):
        vf.push(x[k])
        row = torch.tensor(x[k], dtype=torch.float32)[None, None]
        hist = row.expand(1, 6, 12).clone() if hist is None else torch.cat([hist[:, 1:], row], 1)
        assert np.abs(vf.velocity() - velocity_from_hist(hist)[0].numpy()).max() < 1e-4
