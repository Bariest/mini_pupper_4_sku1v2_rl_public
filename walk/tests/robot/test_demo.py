"""The demo command against mp3-link's fake ESP32 (numpy and pyserial only, as on the Pi)."""
import pytest

pytest.importorskip("serial")

from test_runner import FakeSku2, make_policy                       # noqa: E402  (sys.path set by test_runner)

from mp3link import protocol as p                                   # noqa: E402

from mp3_walk_policy import demo                                    # noqa: E402
from mp3_walk_policy import runner as rn                            # noqa: E402


@pytest.fixture
def fake():
    f = FakeSku2()
    yield f
    f.close()


def test_speed_bounds():
    for bad, fast in ((0.0, False), (0.04, True), (0.21, False), (0.26, True)):
        with pytest.raises(SystemExit):
            demo.check_speed(bad, fast)
    demo.check_speed(0.05, False)
    demo.check_speed(0.20, False)
    demo.check_speed(0.25, True)


def test_armed_walk_needs_a_terminal(tmp_path, monkeypatch):
    pol = make_policy(tmp_path)
    monkeypatch.setattr(demo.sys, "stdin", open("/dev/null"))
    for secs in ("0", "5"):
        with pytest.raises(SystemExit, match="terminal"):
            demo.main(["--port", "/nonexistent", str(pol), "--seconds", secs])
    with pytest.raises(SystemExit, match="seconds"):
        demo.main(["--port", "/nonexistent", str(pol), "--seconds", "nan"])


def test_resolve_by_prefix(tmp_path, monkeypatch):
    for name in ("stock_v3n2_s1", "stock_v3n2_s2", "bam_t5ub06_v3n2_s1"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "policy.npz").write_bytes(b"")
    monkeypatch.setattr(demo, "POLICIES", tmp_path)
    assert demo.resolve("bam").name == "bam_t5ub06_v3n2_s1"
    assert demo.resolve("stock_v3n2_s2").name == "stock_v3n2_s2"
    with pytest.raises(SystemExit, match="ambiguous"):
        demo.resolve("stock")
    with pytest.raises(SystemExit, match="not found"):
        demo.resolve("nothing")


class FakeTty:
    """A terminal on which Enter is never pressed (select() needs a real file descriptor)."""

    def __init__(self):
        self.f = open("/dev/null")

    def isatty(self):
        return True

    def fileno(self):
        return self.f.fileno()

    def readline(self):
        return ""


def test_writes_the_configuration_then_walks(fake, tmp_path, monkeypatch):
    """After a power-up the servos run their flash gains: the demo writes the configuration, then the runner arms."""
    pol = make_policy(tmp_path)
    ids = {name: i for i, name in enumerate(p.PARAMS)}
    rc = demo.main(["--port", fake.port, str(pol), "--seconds", "0.3", "--hold-before", "0.2", "--dry"])
    assert rc == 0 and not fake.armed_seen                     # --dry: the configuration written, nothing armed
    assert fake.params[1][ids["kp_position"]] == pytest.approx(80.0)
    fake.params[1][ids["kp_position"]] = 0.0                       # as after a power cycle
    monkeypatch.setattr(demo.sys, "stdin", FakeTty())
    rc = demo.main(["--port", fake.port, str(pol), "--seconds", "0.3", "--hold-before", "0.2"])
    assert rc == 0 and fake.armed_seen and not fake.armed
    for s in range(1, 13):
        for name, val in rn.IDENTIFIED.items():
            assert fake.params[s][ids[name]] == pytest.approx(val, rel=1e-4)
