"""A gallery of the low-poly generator's surfaces (``terrain.low_poly_heights``) with SKU2 standing on each, for scale.

    MUJOCO_GL=egl python -B tools/terrain_gallery.py --out runs/preview/gallery.png

Rows: relief (peak to valley); columns: feature size (point spacing). The last row varies the slope and shows a
periodic (printable) tile. Each tile is 0.6 m square; the robot stands at its centre (posed, not simulated).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain_preview import add_robots, pose_stand, shoot  # noqa: E402

from mp3_walk_policy import terrain  # noqa: E402

SIZE, PITCH = 0.6, 0.005


def tile(amplitude, feature, slope=0.0, periodic=False, seed=0):
    h = terrain.low_poly_heights(np.random.default_rng(seed), (SIZE, SIZE), PITCH, amplitude, feature,
                                 periodic=periodic, slope_deg=slope)
    return h


def render(h, title):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    spec.worldbody.add_light(pos=[0.3, 0.3, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    terrain.add_hfield(spec, h, PITCH, 0.01)
    c = (np.array(h.shape[::-1]) - 1) * PITCH / 2
    z = terrain.spawn_height(h, PITCH, c[0], c[1])
    prefixes = add_robots(spec, [np.array([c[0], c[1], z])])
    for key in list(spec.keys):
        spec.delete(key)
    m = spec.compile()
    d = mujoco.MjData(m)
    pose_stand(m, d, prefixes)
    mujoco.mj_forward(m, d)
    look = np.array([c[0], c[1], z + 0.03])
    return [(f"{title}", shoot(m, d, look, 0.75, 135, -28, size=(520, 360))),
            (f"{title}, side", shoot(m, d, look, 0.55, 90, -6, size=(520, 360)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = []
    for amp in (0.010, 0.020, 0.030):
        for feat in (0.08, 0.04):
            cases.append((f"relief {amp * 1000:.0f} mm, features {feat * 1000:.0f} mm", tile(amp, feat)))
    cases.append(("relief 15 mm, features 60 mm, slope 10°", tile(0.015, 0.06, slope=10.0)))
    cases.append(("printable: relief 20 mm, features 50 mm, periodic", tile(0.020, 0.05, periodic=True)))
    panels = []
    for title, h in cases:
        panels += render(h, title)
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (t, img) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=80)
    print("wrote", a.out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
