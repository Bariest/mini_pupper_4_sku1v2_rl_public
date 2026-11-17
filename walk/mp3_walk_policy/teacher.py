"""The teacher: minipupperesp's own trot (the firmware's Stanford gait and IK, built on the host by
``firmware_gait.py``), tabulated at the policy's 50 Hz.

The firmware steps its gait every 15 ms; a cycle is 24 ticks (360 ms) = 18 policy steps. At policy step n the teacher's
target is the firmware's target in force at 20 n ms: tick floor(4 n / 3) of the cycle (a zero-order hold, as the robot
would hold it). Each tick is the firmware's own servo tick (``servo_write`` with SKU2's calibration offsets),
stored as the policy target that the contract's decoder turns back into exactly that tick (the tick's centre), so the
simulated servo receives the dd the robot's firmware would send.

The table covers vx 50..200 mm/s in 25 mm/s steps (the firmware's full stick is 200 mm/s, stanford_gait.h:63), each the
steady cycle after 240 ticks of warm-up from the gait's reset (every vx starts the cycle at the same tick), and is
interpolated linearly in vx.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths
from .contract import GAIT_CYCLE_STEPS, Contract

TABLE = Path(__file__).resolve().parent / "data" / "teacher_trot.npz"
VX_GRID_MM_S = np.arange(50.0, 200.0 + 1e-9, 25.0)
WARMUP_TICKS = 240
TICKS_PER_CYCLE = 24


def _gait_module():
    from . import firmware_gait
    return firmware_gait


def build_table(contract: Contract | None = None) -> dict:
    c = contract or Contract.load()
    fg = _gait_module()
    g = fg.Gait(c.offsets_deg)
    stand_ticks = g.ticks(fg.neutral_feet())
    assert np.array_equal(stand_ticks, c.stand_ticks), (stand_ticks, c.stand_ticks)
    pick = (4 * np.arange(GAIT_CYCLE_STEPS)) // 3
    rel = np.zeros((len(VX_GRID_MM_S), GAIT_CYCLE_STEPS, 12))
    ticks_all = np.zeros((len(VX_GRID_MM_S), TICKS_PER_CYCLE, 12), dtype=int)
    for k, vx in enumerate(VX_GRID_MM_S):
        g.reset()
        for _ in range(WARMUP_TICKS):
            g.step(vx)
        ticks = np.array([g.ticks(g.step(vx)) for _ in range(TICKS_PER_CYCLE)])
        ticks_all[k] = ticks
        rel[k] = c.rel_from_ticks(ticks[pick])          # re-quantises to exactly these ticks (contract)
    sources = {"gait_host.c": fg.SOURCES[0], "minipupperesp/main/stanford_gait.c": fg.SOURCES[1],
               "minipupperesp/main/stanford_kinematics.c": fg.SOURCES[2]}
    return {"vx_mm_s": VX_GRID_MM_S, "rel": rel, "ticks": ticks_all, "pick": pick,
            "sources": {k: paths.sha256(p) for k, p in sources.items()}}


def save_table(path: Path = TABLE) -> dict:
    import json
    t = build_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vx_mm_s=t["vx_mm_s"], rel=t["rel"], ticks=t["ticks"], pick=t["pick"],
                        sources=json.dumps(t["sources"]))
    return t


OMNI_TABLE = Path(__file__).resolve().parent / "data" / "teacher_omni.npz"
OMNI_VXY_GRID_MM_S = np.arange(-140.0, 140.0 + 1e-9, 20.0)
OMNI_WZ_GRID_RAD_S = np.arange(-0.8, 0.8 + 1e-9, 0.2)


def build_omni_table(contract: Contract | None = None) -> dict:
    """The omni teacher: the same firmware trot on a (vx, vy, wz) grid, each the steady cycle after the same warm-up
    from the gait's reset. ``rel`` is (n_vx, n_vy, n_wz, 18, 12)."""
    c = contract or Contract.load()
    fg = _gait_module()
    g = fg.Gait(c.offsets_deg)
    pick = (4 * np.arange(GAIT_CYCLE_STEPS)) // 3
    vg, wg = OMNI_VXY_GRID_MM_S, OMNI_WZ_GRID_RAD_S
    rel = np.zeros((len(vg), len(vg), len(wg), GAIT_CYCLE_STEPS, 12), dtype=np.float32)
    for i, vx in enumerate(vg):
        for j, vy in enumerate(vg):
            for k, wz in enumerate(wg):
                g.reset()
                for _ in range(WARMUP_TICKS):
                    g.step(vx, vy, wz)
                ticks = np.array([g.ticks(g.step(vx, vy, wz)) for _ in range(TICKS_PER_CYCLE)])
                rel[i, j, k] = c.rel_from_ticks(ticks[pick])
    sources = {"gait_host.c": fg.SOURCES[0], "minipupperesp/main/stanford_gait.c": fg.SOURCES[1],
               "minipupperesp/main/stanford_kinematics.c": fg.SOURCES[2]}
    return {"vxy_mm_s": vg, "wz_rad_s": wg, "rel": rel, "pick": pick,
            "sources": {k: paths.sha256(p) for k, p in sources.items()}}


def save_omni_table(path: Path = OMNI_TABLE) -> dict:
    import json
    t = build_omni_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vxy_mm_s=t["vxy_mm_s"], wz_rad_s=t["wz_rad_s"], rel=t["rel"], pick=t["pick"],
                        sources=json.dumps(t["sources"]))
    return t


class TorchOmniTeacher:
    """``target_rel(cmd, step)``: the omni teacher's target for each env, trilinear in (vx m/s, vy m/s, wz rad/s)."""

    def __init__(self, device, path: Path = OMNI_TABLE):
        import torch
        d = np.load(path)
        self.axes = [torch.tensor(d["vxy_mm_s"] / 1000.0, dtype=torch.float32, device=device),
                     torch.tensor(d["vxy_mm_s"] / 1000.0, dtype=torch.float32, device=device),
                     torch.tensor(d["wz_rad_s"], dtype=torch.float32, device=device)]
        self.rel = torch.tensor(d["rel"], dtype=torch.float32, device=device)      # (nx, ny, nw, 18, 12)

    def target_rel(self, cmd, step):
        import torch
        lo, w = [], []
        for a, ax in enumerate(self.axes):
            v = cmd[:, a].clamp(float(ax[0]), float(ax[-1])).contiguous()
            hi = torch.searchsorted(ax, v).clamp(1, len(ax) - 1)
            lo.append(hi - 1)
            w.append(((v - ax[hi - 1]) / (ax[hi] - ax[hi - 1])).unsqueeze(-1))
        ph = torch.remainder(step, GAIT_CYCLE_STEPS)
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dw in (0, 1):
                    wt = ((w[0] if dx else 1 - w[0]) * (w[1] if dy else 1 - w[1]) * (w[2] if dw else 1 - w[2]))
                    out = out + wt * self.rel[lo[0] + dx, lo[1] + dy, lo[2] + dw, ph]
        return out


class TorchTeacher:
    """``target_rel(vx_m_s, step)``: the teacher's policy-coordinate target for each env (torch)."""

    def __init__(self, device, path: Path = TABLE):
        import torch
        d = np.load(path)
        self.vx = torch.tensor(d["vx_mm_s"] / 1000.0, dtype=torch.float32, device=device)
        self.rel = torch.tensor(d["rel"], dtype=torch.float32, device=device)        # (n_vx, 18, 12)

    def target_rel(self, vx, step):
        import torch
        v = vx.clamp(float(self.vx[0]), float(self.vx[-1]))
        hi = torch.searchsorted(self.vx, v.contiguous()).clamp(1, len(self.vx) - 1)
        lo = hi - 1
        w = ((v - self.vx[lo]) / (self.vx[hi] - self.vx[lo])).unsqueeze(-1)
        ph = torch.remainder(step, GAIT_CYCLE_STEPS)
        return (1 - w) * self.rel[lo, ph] + w * self.rel[hi, ph]
