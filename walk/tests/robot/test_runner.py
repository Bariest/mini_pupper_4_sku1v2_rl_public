"""The Pi runner against mp3-link's fake ESP32 (a pty). Numpy and pyserial only, as on the Pi:

    python -m pytest -q tests/robot
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("serial")
REPO = Path(__file__).resolve().parents[2]
PI = Path(os.environ.get("MP3_LINK_DIR", REPO.parent / "mp3-link")) / "pi"
sys.path[:0] = [str(REPO), str(PI), str(PI / "tests")]

from fake_esp32 import FakeEsp32                                     # noqa: E402
from mp3link import protocol as p                                    # noqa: E402
from mp3link.link import Link                                        # noqa: E402

from mp3_walk_policy import runner as rn                             # noqa: E402
from mp3_walk_policy.contract import CONTRACT_JSON, Contract        # noqa: E402

C = Contract.load()


class FakeSku2(FakeEsp32):
    """The fake ESP32 with SKU2's IMU sign (-1 g on the sensor's z upright), its offsets and a settable IMU."""

    def __init__(self):
        self.acc = [0, 0, -1000]
        self.gyr = [0, 0, 0]
        self.imu_ok = True
        self.temp_dc = 315
        super().__init__()
        self.offsets = {i + 1: float(v) for i, v in enumerate(C.offsets_deg)}
        self.pos = [float(v) for v in C.stand_dd]

    def configure(self):
        ids = {name: i for i, name in enumerate(p.PARAMS)}
        for s in range(1, 13):
            for name, val in rn.IDENTIFIED.items():
                self.params[s][ids[name]] = val

    def _state(self) -> bytes:
        flags = self.latched | (p.ST_ARMED if self.armed else 0) | (p.ST_IMU_OK if self.imu_ok else 0)
        body = p.STATE.pack(p.MAGIC_STATE, 1, self.seq, self.tick, int(time.monotonic() * 1e6) & 0xFFFFFFFF,
                            flags, self.fault_servo, 0x0F,
                            *[int(x) for x in self.pos], *[0] * 12, *[self.temp_dc] * 12,
                            *[int(v) for v in self.acc], *[int(v) for v in self.gyr], 0)
        return body + struct.pack("<H", p.crc16(body))


def make_policy(tmp_path, bias=0.0):
    d = tmp_path / "pol"
    d.mkdir()
    meta = {"contract_sha256": rn.sha256(CONTRACT_JSON)}
    obs = np.zeros((2, 45), dtype=np.float32)
    act = np.full((2, 12), bias, dtype=np.float32)
    np.savez(d / "policy.npz", n_layers=1, scale=np.array([0.10, 0.40, 0.42] * 4, dtype=np.float32),
             meta=json.dumps(meta), fixture_obs=obs, fixture_act=act,
             W0=np.zeros((12, 45), dtype=np.float32), b0=np.full(12, bias, dtype=np.float32))
    return d


@pytest.fixture
def fake():
    f = FakeSku2()
    yield f
    f.close()


def run(fake, tmp_path, *extra, bias=0.0, seconds=0.5):
    pol = make_policy(tmp_path, bias)
    log = tmp_path / "take"
    argv = ["--port", fake.port, "run", "--policy", str(pol), "--seconds", str(seconds), "--mount", "flip_x",
            "--hold-before", "0.2", "--hold-after", "0.2", "--log", str(log), *extra]
    rc = rn.main(argv)
    return rc, log


def test_refuses_without_config(fake, tmp_path):
    with pytest.raises(SystemExit, match="config"):
        run(fake, tmp_path)
    assert not fake.armed_seen


def test_refuses_without_imu(fake, tmp_path):
    fake.configure()
    fake.imu_ok = False
    with pytest.raises(SystemExit, match="IMU"):
        run(fake, tmp_path)


def test_dry_run_never_arms(fake, tmp_path):
    rc, log = run(fake, tmp_path, "--dry")
    assert rc == 0 and not fake.armed_seen
    pol = np.load(log.with_suffix(".policy.npz"))
    assert 20 <= len(pol["tick"]) <= 26
    assert np.all(np.diff(pol["tick"]) == 4)                      # one step per 4 STATE ticks


def test_run_holds_stand_and_logs(fake, tmp_path):
    fake.configure()
    rc, log = run(fake, tmp_path, seconds=0.4)
    assert rc == 0 and fake.armed_seen
    assert fake.caps_armed == {rn.CAP_MA}
    pol = np.load(log.with_suffix(".policy.npz"))
    assert np.all(pol["dd"] == np.asarray(C.stand_dd))             # zero action = the stand, exactly
    meta = json.loads(str(pol["meta"]))
    assert meta["stop_reason"] == "completed" and not meta["warnings"]
    st = json.loads(log.with_suffix(".stand.json").read_text())
    assert st["stand_dd"] == list(C.stand_dd) and len(st["offsets_deg"]) == 12
    raw = np.load(log.with_suffix(".npz"))
    assert len(raw["sent_seq"]) > 0 and len(raw["pos_dd"]) > 0
    assert not fake.armed                                          # disarmed at the end


def test_step_limit_and_band(fake, tmp_path):
    fake.configure()
    rc, log = run(fake, tmp_path, bias=1.0, seconds=0.2)
    pol = np.load(log.with_suffix(".policy.npz"))
    first = np.degrees(np.abs(pol["rel"][0]))
    assert np.all(first <= 20.0 + 0.31)                            # 20 deg, then one firmware tick (as in simulation)
    assert np.all(np.degrees(np.abs(pol["rel"][-1])) <= np.array([20.0 if i % 3 == 0 else 60.0 for i in range(12)]) + 0.31)


def test_offsets_warn_only(fake, tmp_path):
    fake.configure()
    fake.offsets[5] += 2.0
    rc, log = run(fake, tmp_path, seconds=0.2)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert rc == 0 and meta["warnings"] and "differ" in meta["warnings"][0]


def test_clear_rearms_after_a_guard_trip(fake, tmp_path):
    fake.configure()
    fake.latched, fake.fault_servo = p.ST_FAULT_CUR, 9                 # a guard trip from an earlier take
    with pytest.raises(Exception, match="fault latched"):
        run(fake, tmp_path)
    assert not fake.armed_seen
    (tmp_path / "c").mkdir()
    rc, log = run(fake, tmp_path / "c", "--clear", seconds=0.4)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert rc == 0 and fake.armed_seen and meta["stop_reason"] == "completed"
    assert not fake.latched & p.ST_FAULTS


def test_tilt_disarms_at_once(fake, tmp_path):
    fake.configure()
    import threading
    threading.Timer(4.5, lambda: setattr(fake, "acc", [0, 700, -700])).start()     # 45 deg
    rc, log = run(fake, tmp_path, seconds=3.0)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert meta["stop_reason"].startswith("disarm: tilt")
    assert not fake.armed


def test_imu_dropout_stops(fake, tmp_path):
    fake.configure()
    import threading
    threading.Timer(4.5, lambda: (setattr(fake, "acc", [0, 0, 0]), setattr(fake, "imu_ok", False))).start()
    rc, log = run(fake, tmp_path, seconds=3.0)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert meta["stop_reason"] == "stop: IMU frame invalid"
    assert not fake.armed


def test_hot_servo_stops(fake, tmp_path):
    fake.configure()
    import threading
    threading.Timer(4.5, lambda: setattr(fake, "temp_dc", 560)).start()
    rc, log = run(fake, tmp_path, seconds=3.0)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert meta["stop_reason"].startswith("stop: servo at 56.0")


def test_stop_file_ends_gracefully(fake, tmp_path):
    """The stop file (tools/pi_take.sh --soft-stop) ends a take as Enter does: ramp to the stand, hold, disarm, log."""
    fake.configure()
    import threading
    threading.Timer(4.5, lambda: Path(f"{tmp_path / 'take'}.stop").touch()).start()
    rc, log = run(fake, tmp_path, seconds=5.0)
    pol = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(pol["meta"]))
    assert rc == 0 and meta["stop_reason"] == "stop: stop file"
    assert 0 < len(pol["tick"]) < 200                              # stopped before the 5 s (250 steps) ran out
    raw = np.load(log.with_suffix(".npz"))
    assert np.all(raw["sent_pos_dd"][-1] == np.asarray(C.stand_dd))  # ramped back to the stand
    assert not fake.armed


def test_tilt_and_gravity_frames():
    up = [0, 0, -990]
    assert rn.tilt_deg(up) < 1e-6
    for name, R in rn.IMU_MOUNTS.items():
        ob = rn.ObservationBuilder(C, R)
        st = type("S", (), {"acc_mg": np.array(up), "gyr_ddps": np.zeros(3)})()
        assert np.allclose(ob.gravity_body(st), [0, 0, -1]), name
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_safety_rules_on_sku2_logs():
    """SKU2's real captures (accelerometer and flags, tests/data): no IMU or tilt trip on the trots and stands."""
    d = np.load(REPO / "tests" / "data" / "sku2_imu_captures.npz")
    peak = {}
    for name in sorted({k.split("__")[0] for k in d.files}):
        acc, flags = d[name + "__acc_mg"], d[name + "__flags"]
        tf = rn.TiltFilter()
        ok = [bool(fl & p.ST_IMU_OK) for fl in flags]
        if sum(ok) < 0.9 * len(ok):
            continue
        peak[name] = max(tf.push(a) for a, o in zip(acc, ok) if o)
        st = [type("S", (), {"flags": fl, "acc_mg": a})() for fl, a in zip(flags, acc)]
        assert all(rn.imu_valid(s, p.ST_IMU_OK) for s, o in zip(st, ok) if o), name
    motion = {k: v for k, v in peak.items() if "trot" in k or "bounce" in k or "keyframes" in k}
    assert motion and max(motion.values()) < 15.0, motion


def test_armed_replay_refuses_without_a_measured_mount(fake, tmp_path, monkeypatch):
    fake.configure()
    monkeypatch.setattr(rn, "IMU_MOUNT_JSON", tmp_path / "missing_imu_mount.json")
    pol = make_policy(tmp_path)
    acts = tmp_path / "acts.npz"
    np.savez(acts, raw_actions=np.zeros((10, 12), dtype=np.float32))
    with pytest.raises(SystemExit, match="IMU mounting"):
        rn.main(["--port", fake.port, "replay", "--policy", str(pol), "--actions", str(acts), "--seconds", "0.2",
                 "--hold-before", "0.2", "--hold-after", "0.2", "--log", str(tmp_path / "take")])
    assert not fake.armed_seen


def make_omni_policy(tmp_path):
    d = make_policy(tmp_path)
    z = dict(np.load(d / "policy.npz"))
    z["meta"] = json.dumps({"contract_sha256": rn.sha256(CONTRACT_JSON), "omni": True})
    np.savez(d / "policy.npz", **z)
    return d


def test_heading_loop_steers_against_a_turn(fake, tmp_path):
    """A turn starts after the gyro bias is measured; the loop's wz opposes it and reaches the policy's command."""
    import threading
    pol = make_omni_policy(tmp_path)
    log = tmp_path / "take"
    threading.Timer(1.3, lambda: fake.gyr.__setitem__(2, 300)).start()      # 30 deg/s on the sensor's z
    rc = rn.main(["--port", fake.port, "run", "--policy", str(pol), "--seconds", "1.0", "--mount", "flip_x",
                  "--hold-before", "0.2", "--hold-after", "0.2", "--dry", "--heading-loop", "--log", str(log)])
    assert rc == 0
    z = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(z["meta"]))
    assert meta["heading_loop"]["kp"] == 2.0
    yaw_rate = z["obs"][:, 2]                                                # the body's yaw rate, as observed
    turning = np.abs(yaw_rate) > 0.1
    assert turning.sum() > 10
    assert np.all(np.sign(z["wz_cmd"][turning][-5:]) == -np.sign(yaw_rate[turning][-5:]))
    assert np.allclose(z["obs"][:, 8], z["wz_cmd"])                         # the command slot carries it


def test_heading_loop_records_its_law(fake, tmp_path):
    """The take's meta names the loop that ran (gains, limits, anti-windup) and the integrator is logged per step."""
    import threading
    pol = make_omni_policy(tmp_path)
    log = tmp_path / "take"
    threading.Timer(1.3, lambda: fake.gyr.__setitem__(2, 300)).start()
    rc = rn.main(["--port", fake.port, "run", "--policy", str(pol), "--seconds", "1.0", "--mount", "flip_x",
                  "--hold-before", "0.2", "--hold-after", "0.2", "--dry", "--heading-loop", "--heading-aw", "cond",
                  "--heading-wz-max", "0.5", "--log", str(log)])
    assert rc == 0
    z = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(z["meta"]))
    assert meta["heading_loop"] == {"kp": 2.0, "ki": 0.5, "wz_max": 0.5, "i_max": 0.5, "tau_g_s": 1.0,
                                    "antiwindup": "cond"}
    assert len(z["heading_integ"]) == len(z["wz_cmd"]) > 10
    assert np.all(np.abs(z["wz_cmd"]) <= 0.5 + 1e-9)


def test_heading_loop_refused_for_a_forward_policy(fake, tmp_path):
    rc, _ = run(fake, tmp_path, "--dry", "--heading-loop")
    assert rc == 2


def test_history_policy_runs(fake, tmp_path):
    """A policy exported with a 5-step history (225 inputs) gets the stacked observation."""
    d = make_omni_policy(tmp_path)
    z = dict(np.load(d / "policy.npz"))
    z["W0"] = np.zeros((12, 225), dtype=np.float32)
    z["fixture_obs"] = np.zeros((2, 225), dtype=np.float32)
    z["meta"] = json.dumps({"contract_sha256": rn.sha256(CONTRACT_JSON), "omni": True, "history": 5})
    np.savez(d / "policy.npz", **z)
    log = tmp_path / "take"
    rc = rn.main(["--port", fake.port, "run", "--policy", str(d), "--seconds", "0.3", "--mount", "flip_x",
                  "--hold-before", "0.2", "--hold-after", "0.2", "--dry", "--log", str(log)])
    assert rc == 0
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert meta["stop_reason"] == "completed" and meta["history"] == 5


def test_contract_stand_is_its_calibration():
    """The contract's stand ticks and positions are what its offsets give at 0 deg (the firmware's arithmetic), and
    `runner check` accepts a policy exported before a recalibration the contract lists as superseded."""
    ticks = C.ticks_from_servo_deg(np.zeros(12))
    assert np.array_equal(ticks, C.stand_ticks) and np.array_equal(C.dd_from_ticks(ticks), C.stand_dd)
    superseded = [c["sha256"] for c in C.meta.get("calibrations_superseded", [])]
    pol = REPO / "policies" / "bam_t5ub06_terrain_h10_v2"
    want = json.loads(str(np.load(pol / "policy.npz")["meta"]))["contract_sha256"]
    assert want == rn.sha256(CONTRACT_JSON) or want in superseded
    assert rn.main(["check", "--policy", str(pol)]) == 0


# ---- the classic gait (minipupperesp's Stanford trot, firmware_gait) -------------------------------------------------
def gait(fake, tmp_path, *extra, seconds=0.3):
    log = tmp_path / "gtake"
    argv = ["--port", fake.port, "gait", "--seconds", str(seconds), "--mount", "flip_x",
            "--hold-before", "0.2", "--hold-after", "0.2", "--log", str(log), *extra]
    return rn.main(argv), log


def host_gait_dd(n, vx_mm_s):
    from mp3_walk_policy import firmware_gait
    g = firmware_gait.Gait(C.offsets_deg)
    g.reset()
    return np.array([C.dd_from_ticks(g.ticks(g.step(vx_mm_s))) for _ in range(n)], dtype=float)


def after_sends(fake, n, fn):
    """Call fn once the fake has run n ticks armed (5 ms each: the 2 s ramp and 0.2 s hold are 440; 3 per gait tick)."""
    import threading

    def wait():
        t0 = time.monotonic()
        while len(fake.armed_seen) < n and time.monotonic() - t0 < 20:
            time.sleep(0.01)
        fn()
    threading.Thread(target=wait, daemon=True).start()


def test_gait_sends_the_firmware_trot(fake, tmp_path):
    fake.configure()
    rc, log = gait(fake, tmp_path, "--vx", "0.10", seconds=0.6)
    assert rc == 0 and fake.armed_seen
    pol = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(pol["meta"]))
    assert meta["stop_reason"] == "completed" and meta["policy"] == "firmware_gait"
    n = len(pol["tick"])
    assert 38 <= n <= 42                                                # 0.6 s of 15 ms gait ticks
    assert np.all(np.diff(pol["tick"]) == 3)                            # one gait tick per 3 STATE ticks
    assert np.array_equal(pol["dd"], host_gait_dd(n, 100.0))           # exactly the firmware's targets
    assert not np.any(pol["clipped"])
    sent = [t for _, t in fake.armed_seen]
    assert any(np.array_equal(np.asarray(t), pol["dd"][10]) for t in sent)
    assert not fake.armed


def test_gait_dry_never_arms(fake, tmp_path):
    rc, log = gait(fake, tmp_path, "--dry", "--heading-loop")
    assert rc == 0 and not fake.armed_seen
    pol = np.load(log.with_suffix(".policy.npz"))
    assert len(pol["heading_psi"]) == len(pol["tick"]) == len(pol["wz_cmd"])


def test_gait_command_file_changes_the_command(fake, tmp_path):
    fake.configure()
    cmd = tmp_path / "gtake.cmd"
    after_sends(fake, 530, lambda: cmd.write_text("0.05 0 0.3\n"))            # 30 gait ticks in
    rc, log = gait(fake, tmp_path, "--vx", "0.10", seconds=1.5)
    c = np.load(log.with_suffix(".policy.npz"))["command"]
    assert np.allclose(c[0], [0.10, 0, 0]) and np.allclose(c[-1], [0.05, 0, 0.3])


def test_gait_stop_file_ramps_to_the_stand(fake, tmp_path):
    fake.configure()
    after_sends(fake, 530, lambda: (tmp_path / "gtake.stop").touch())
    rc, log = gait(fake, tmp_path, "--vx", "0.10", seconds=3.0)
    pol = np.load(log.with_suffix(".policy.npz"))
    meta = json.loads(str(pol["meta"]))
    assert 20 < len(pol["tick"]) < 100                                  # stopped while walking
    assert meta["stop_reason"] == "stop: stop file" and not fake.armed
    assert np.array_equal(np.asarray(fake.armed_seen[-1][1]), np.asarray(C.stand_dd, dtype=float))


def test_tilt_cut_can_be_turned_off(fake, tmp_path):
    fake.configure()
    after_sends(fake, 470, lambda: setattr(fake, "acc", [0, 700, -700]))      # 45 deg, while walking
    rc, log = gait(fake, tmp_path, "--vx", "0.10", "--tilt-cut", "0", seconds=0.6)
    meta = json.loads(str(np.load(log.with_suffix(".policy.npz"))["meta"]))
    assert meta["stop_reason"] == "completed"


def make_heading_policy(tmp_path, history=10):
    """An omni policy with the heading term (meta obs_terms_extra): 46 inputs per frame."""
    from mp3_walk_policy.contract import HEADING_TERM, obs_size
    d = make_omni_policy(tmp_path)
    z = dict(np.load(d / "policy.npz"))
    width = obs_size(True) * max(history, 1)
    z["W0"] = np.zeros((12, width), dtype=np.float32)
    z["fixture_obs"] = np.zeros((2, width), dtype=np.float32)
    z["meta"] = json.dumps({"contract_sha256": rn.sha256(CONTRACT_JSON), "omni": True, "history": history,
                            "obs_terms_extra": [HEADING_TERM]})
    np.savez(d / "policy.npz", **z)
    return d


def test_heading_term_policy_check(tmp_path, capsys):
    """runner check accepts a heading-term policy (460 = 46 x 10) and fails one whose width disagrees with its meta."""
    d = make_heading_policy(tmp_path)
    assert rn.main(["check", "--policy", str(d)]) == 0
    assert "460 = 46 x 10" in capsys.readouterr().out
    z = dict(np.load(d / "policy.npz"))
    z["W0"] = np.zeros((12, 450), dtype=np.float32)
    z["fixture_obs"] = np.zeros((2, 450), dtype=np.float32)
    np.savez(d / "policy.npz", **z)
    assert rn.main(["check", "--policy", str(d)]) == 1


def test_heading_term_policy_gets_the_error_as_its_last_block(fake, tmp_path, monkeypatch):
    """A heading-term policy with a 10-step history: each step's observation is 46 floats, the last the heading loop's
    error as observed (contract.heading_obs of the error its wz came from), and the network's input is 460 floats whose
    last 10 are that error's frames, oldest to newest, after the 450 of the other terms."""
    import threading
    from mp3_walk_policy.contract import HistoryStacker, heading_obs
    d = make_heading_policy(tmp_path)
    seen = []
    orig = rn.NumpyPolicy.__call__
    monkeypatch.setattr(rn.NumpyPolicy, "__call__", lambda self, x: (seen.append(np.array(x)), orig(self, x))[1])
    threading.Timer(1.3, lambda: fake.gyr.__setitem__(2, 300)).start()      # 30 deg/s on the sensor's z
    log = tmp_path / "take"
    rc = rn.main(["--port", fake.port, "run", "--policy", str(d), "--seconds", "1.0", "--mount", "flip_x",
                  "--hold-before", "0.2", "--hold-after", "0.2", "--dry", "--heading-loop", "--log", str(log)])
    assert rc == 0
    z = np.load(log.with_suffix(".policy.npz"))
    assert json.loads(str(z["meta"]))["heading_obs"] is True
    obs, err = z["obs"], z["heading_err"]
    assert obs.shape[1] == 46 and len(err) == len(obs) > 20
    assert np.allclose(obs[:, 45], [heading_obs(e) for e in err], atol=1e-6)
    assert np.abs(obs[:, 45]).max() > 0.05                                # the turn moved it
    assert np.all(np.sign(obs[:, 45][np.abs(obs[:, 45]) > 0.05]) == -np.sign(obs[:, 2][np.abs(obs[:, 45]) > 0.05]))
    assert len(seen) == len(obs) and all(x.shape == (460,) for x in seen)
    for k in range(len(obs)):
        newest = obs[max(0, k - 9):k + 1, 45]
        block = seen[k][450:]
        assert np.allclose(block[10 - len(newest):], newest)              # oldest to newest
        assert np.all(block[:10 - len(newest)] == obs[0, 45])             # the first observation fills the slots
    ref = HistoryStacker(10)                                              # the other terms: as without the term
    for k in range(len(obs)):
        assert np.array_equal(seen[k][:450], ref(obs[k, :45]))


def test_heading_term_is_zero_without_the_loop(fake, tmp_path):
    d = make_heading_policy(tmp_path, history=0)
    log = tmp_path / "take"
    rc = rn.main(["--port", fake.port, "run", "--policy", str(d), "--seconds", "0.3", "--mount", "flip_x",
                  "--hold-before", "0.2", "--hold-after", "0.2", "--dry", "--log", str(log)])
    assert rc == 0
    obs = np.load(log.with_suffix(".policy.npz"))["obs"]
    assert obs.shape[1] == 46 and not obs[:, 45].any()
