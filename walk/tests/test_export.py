"""The numpy policy reproduces the torch actor (the export the robot runs)."""
import numpy as np

from mp3_walk_policy.export import NumpyPolicy, export


def test_numpy_equals_torch(tmp_path):
    import torch
    from mp3_walk_policy.distill import mlp
    net = mlp(45, 12)
    sd = {f"mlp.{k}": v for k, v in net.state_dict().items()}
    sd["distribution.std_param"] = torch.full((12,), 0.1)
    ck = tmp_path / "m.pt"
    torch.save({"actor_state_dict": sd}, ck)
    path = export(ck, tmp_path / "exp", scale=[0.1] * 12)
    p = NumpyPolicy(path)
    d = np.load(path)
    assert np.abs(p(d["fixture_obs"]) - d["fixture_act"]).max() < 1e-5
    with torch.no_grad():
        x = np.random.default_rng(5).normal(size=(10, 45)).astype(np.float32)
        assert np.abs(p(x) - net(torch.from_numpy(x)).numpy()).max() < 1e-5


def test_heading_obs_export_meta_and_width(tmp_path):
    """heading_obs: the first layer is 46 x history wide and the meta names the term (obs_terms_extra), which the
    runner reads (contract.has_heading_obs); the plain export's meta does not."""
    import json
    import pytest
    import torch
    from mp3_walk_policy.contract import has_heading_obs
    from mp3_walk_policy.distill import mlp
    for width, kw, want in ((460, {"heading_obs": True, "history": 10}, True), (450, {"history": 10}, False)):
        sd = {f"mlp.{k}": v for k, v in mlp(width, 12).state_dict().items()}
        ck = tmp_path / f"m{width}.pt"
        torch.save({"actor_state_dict": sd}, ck)
        p = NumpyPolicy(export(ck, tmp_path / f"exp{width}", scale=[0.1] * 12, omni=True, **kw))
        assert has_heading_obs(p.meta) is want and json.dumps(p.meta).count("heading_error") == (2 if want else 0)
    with pytest.raises(AssertionError):
        export(tmp_path / "m450.pt", tmp_path / "bad", scale=[0.1] * 12, history=10, heading_obs=True)
