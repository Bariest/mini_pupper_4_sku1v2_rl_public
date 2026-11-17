"""``tools/chain_delay.py`` on a 10 s policy walk logged on SKU2: the chain is 10 ms. Numpy only."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import chain_delay  # noqa: E402

TAKE = REPO / "tests/data/sku2_walk_log"


def test_walk_chain_is_10_ms():
    r = chain_delay.summarise(str(TAKE), 10.0)
    assert r["steps"] == 498
    assert abs(r["joint_ms_median"] - 10.0) < 0.05
    assert r["off_frac"] < 0.01
    assert 5.5 < r["imu_age_ms_median"] < 7.5


def test_wrong_expectation_flags_every_step():
    assert chain_delay.summarise(str(TAKE), 15.0)["off_frac"] > 0.99
