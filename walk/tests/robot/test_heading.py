"""The runner's heading loop: it integrates the yaw about gravity (also on a slope, also with a vibrating
accelerometer) and, through the yaw-rate command, holds a line against a constant turn of the robot's own."""
import numpy as np
import pytest

from mp3_walk_policy.heading import pi_step
from mp3_walk_policy.runner import HeadingLoop

FRAME_DT, STEP_FRAMES = 0.005, 4


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _walk(drift, seconds=10.0, pitch=0.0, accel_noise=0.0, kp=2.0, ki=0.5, seed=0, antiwindup="none", kick=None):
    """A robot that turns at ``drift`` plus the commanded wz (applied one policy step late), walking 0.10 m/s.
    ``kick``: (t0, duration, rate) a yaw-rate disturbance (an edge strike) during [t0, t0 + duration)."""
    rng = np.random.default_rng(seed)
    loop = HeadingLoop(kp=kp, ki=ki, antiwindup=antiwindup)
    R = _rot_y(pitch)                                 # body pitched on a slope: world = R @ body
    yaw, y, wz_applied, wz_next = 0.0, 0.0, 0.0, 0.0
    for f in range(int(seconds / FRAME_DT)):
        t = f * FRAME_DT
        rate = drift + wz_applied + (kick[2] if kick and kick[0] <= t < kick[0] + kick[1] else 0.0)
        gyro_body = R.T @ np.array([0.0, 0.0, rate])
        grav_body = R.T @ np.array([0.0, 0.0, -1.0]) + rng.uniform(-accel_noise, accel_noise, 3)
        loop.push(gyro_body, grav_body / np.linalg.norm(grav_body))
        yaw += rate * FRAME_DT
        y += 0.10 * np.sin(yaw) * FRAME_DT
        if f % STEP_FRAMES == STEP_FRAMES - 1:
            wz_applied, wz_next = wz_next, loop.wz()
    return yaw, y, loop


@pytest.mark.parametrize("antiwindup", ["none", "cond"])
def test_holds_the_line_against_the_right_drift(antiwindup):
    drift = -np.radians(64.0) / 10.0                  # the worst policy turn measured on SKU2 (64 deg in 10 s)
    yaw, y, _ = _walk(drift, antiwindup=antiwindup)
    assert abs(np.degrees(yaw)) < 2.0
    assert abs(y) < 0.10 * 1.0                         # under 10 % of the 1 m walked
    yaw_open, y_open, _ = _walk(drift, kp=0.0, ki=0.0)
    assert abs(np.degrees(yaw_open)) > 60.0            # the same robot without the loop


def test_heading_about_gravity_on_a_slope():
    _, _, loop = _walk(0.05, seconds=4.0, pitch=np.radians(20.0), kp=0.0, ki=0.0)
    assert abs(loop.psi - 0.05 * 4.0) < 0.01 * 0.2


def test_vibrating_accelerometer():
    _, _, loop = _walk(0.05, seconds=4.0, pitch=np.radians(10.0), accel_noise=0.3, kp=0.0, ki=0.0)
    assert abs(loop.psi - 0.05 * 4.0) < 0.03 * 0.2


def test_command_is_clipped():
    loop = HeadingLoop(wz_max=0.4)
    loop.psi = 1.0
    assert loop.wz() == -0.4



def test_antiwindup_overshoots_less_after_an_edge_strike():
    peaks = {}
    for aw in ("none", "cond"):
        loop = HeadingLoop(antiwindup=aw)
        yaw, wz_applied, wz_next, peak = 0.0, 0.0, 0.0, 0.0
        for f in range(int(8.0 / FRAME_DT)):
            t = f * FRAME_DT
            rate = wz_applied + (-1.0 if 1.0 <= t < 1.5 else 0.0)
            loop.push(np.array([0.0, 0.0, rate]), np.array([0.0, 0.0, -1.0]))
            yaw += rate * FRAME_DT
            if t > 1.5:
                peak = max(peak, yaw)
            if f % STEP_FRAMES == STEP_FRAMES - 1:
                wz_applied, wz_next = wz_next, loop.wz()
        peaks[aw] = np.degrees(peak)
        assert abs(np.degrees(yaw)) < 3.0                                   # both return to the line
    assert peaks["cond"] < 0.5 * peaks["none"]
    assert peaks["none"] > 3.0                                              # without anti-windup it overshoots


def test_pi_step_laws():
    """'none' integrates always (the clamp only); 'cond' holds the integrator only while saturated in the error's direction;
    the law gives the same numbers on numpy arrays as on floats."""
    rng = np.random.default_rng(1)
    for aw in ("none", "cond"):
        integ_f, integ_a = 0.0, np.zeros(1)
        for err in rng.uniform(-0.6, 0.6, 500):
            wz_f, integ_f = pi_step(err, integ_f, 0.02, antiwindup=aw)
            wz_a, integ_a = pi_step(np.array([err]), integ_a, 0.02, antiwindup=aw)
            assert np.isclose(float(wz_f), wz_a[0]) and np.isclose(float(integ_f), integ_a[0])
    integ = 0.0
    for err in rng.uniform(-0.6, 0.6, 500):                                # 'none': no anti-windup, the clamp only
        wz, new = pi_step(err, integ, 0.02, antiwindup="none")
        ref_i = float(np.clip(integ + err * 0.02, -0.5, 0.5))
        assert np.isclose(float(new), ref_i) and np.isclose(float(wz), float(np.clip(2 * err + 0.5 * ref_i, -0.4, 0.4)))
        integ = ref_i
    wz, new = pi_step(0.3, 0.2, 0.02, antiwindup="cond")                    # saturated, same sign: held
    assert float(new) == 0.2 and float(wz) == 0.4
    wz, new = pi_step(-0.3, 0.5, 0.02, antiwindup="cond")                   # error against the output: integrates
    assert np.isclose(float(new), 0.5 - 0.006)


def test_lateral_loop_returns_a_displaced_robot_to_its_line():
    """A heading error of 30 deg then the heading loop: without the lateral loop the robot ends parallel to its line but off
    it; with it (heading.lateral_step biasing the target) it ends nearer the line. The law clips its offset."""
    from mp3_walk_policy.heading import LAT_MAX, lateral_step

    def run(k):
        dt, v, psi, y, integ, y_hat = 0.02, 0.10, np.radians(30.0), 0.0, 0.0, 0.0
        for _ in range(int(8.0 / dt)):
            target = 0.0
            if k:
                target, y_hat = lateral_step(y_hat, psi, v, dt, k)
            wz, integ = pi_step(target - psi, integ, dt, antiwindup="cond")
            psi += wz * dt
            y += v * np.sin(psi) * dt
        return y
    assert abs(run(3.0)) < 0.4 * abs(run(0.0))
    off, _ = lateral_step(1.0, 0.0, 0.1, 0.02, 3.0)
    assert off == pytest.approx(-LAT_MAX)


def test_heading_error_sign_wrap_and_clip():
    """The error the loop uses and a heading-term policy observes: + = turn left to correct (the robot turned right),
    wrapped to [-pi, pi), observed clipped to ±HEADING_CLIP rad and scaled to [-1, 1]."""
    from mp3_walk_policy.contract import HEADING_CLIP, heading_obs
    loop = HeadingLoop()
    for psi, want in ((-0.3, 0.3), (0.3, -0.3), (2 * np.pi - 0.1, 0.1), (-2 * np.pi + 0.1, -0.1),
                      (np.pi - 0.05, -(np.pi - 0.05))):
        loop.psi = psi
        assert loop.error() == pytest.approx(want)
        loop.wz()
        assert loop.err == pytest.approx(want)                  # wz() keeps the error it used
    assert heading_obs(0.4) == pytest.approx(0.5) and heading_obs(-0.2) == pytest.approx(-0.25)
    assert heading_obs(HEADING_CLIP + 1.0) == 1.0 and heading_obs(-3.0) == -1.0
    loop.psi = -0.3
    assert np.sign(loop.wz()) == np.sign(heading_obs(loop.err)) == 1.0     # the loop and the term agree in sign


def test_gyro_bias_drifts_the_observed_error():
    """A residual gyro bias b (rad/s) about the vertical integrates into the heading: standing still, the observed
    error after T s is -b T (until the loop turns the body; here open loop), scaled by 1 / HEADING_CLIP."""
    from mp3_walk_policy.contract import HEADING_CLIP, heading_obs
    loop, b, T = HeadingLoop(), 0.01, 5.0
    for _ in range(int(T / FRAME_DT)):
        loop.push(np.array([0.0, 0.0, b]), np.array([0.0, 0.0, -1.0]))
    assert loop.error() == pytest.approx(-b * T, rel=1e-6)
    assert heading_obs(loop.error()) == pytest.approx(-b * T / HEADING_CLIP, rel=1e-6)


def test_history_stacker_appends_the_heading_block():
    """contract.HistoryStacker with the heading term: the 45-float terms' blocks exactly as without it, then the
    heading error's frames (oldest to newest) as one last block; assemble_observation puts it 46th."""
    from mp3_walk_policy.contract import HistoryStacker, assemble_observation, obs_size
    rng = np.random.default_rng(0)
    plain, with_h = HistoryStacker(10), HistoryStacker(10, heading=True)
    hs = []
    for k in range(13):
        o = rng.normal(0, 0.5, 45).astype(np.float32)
        h = float(rng.uniform(-1, 1))
        hs.append(h)
        o46 = assemble_observation(o[0:3], o[3:6], o[6:9], o[9:21], o[21:33], o[33:45], h)
        assert o46.shape == (obs_size(True),) and np.array_equal(o46[:45], o) and o46[45] == np.float32(h)
        x, x0 = with_h(o46), plain(o)
        assert x.shape == (460,) and np.array_equal(x[:450], x0)
        last = hs[-10:]
        assert np.allclose(x[450 + 10 - len(last):], last) and np.all(x[450:450 + 10 - len(last)] == np.float32(hs[0]))
    assert assemble_observation(o[0:3], o[3:6], o[6:9], o[9:21], o[21:33], o[33:45]).shape == (45,)
