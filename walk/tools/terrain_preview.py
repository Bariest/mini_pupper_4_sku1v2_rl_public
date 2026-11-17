"""Look at a terrain set before training on it: mjlab's terrain generator builds the grid exactly as training will, SKU2
is placed at the stand on chosen tiles, and the views are rendered offscreen to one PNG.

    MUJOCO_GL=egl python -B tools/terrain_preview.py --out runs/preview/rough_basic.png         # overview + close-ups
    python -B tools/terrain_preview.py --only low_poly --rows 5 --levels 0 2 4 --out lp.png
    python -B tools/terrain_preview.py --preset rough --out rough.png                         # another preset
    python -B tools/terrain_preview.py --strip --out strip.png                                # the printed-tile strip

Close-ups show the robot on each terrain type at the chosen difficulty levels (rows), from the side and at an angle.
Prints each tile's spawn origin and the contacts of the robot placed there (nothing is simulated: the robot is posed).
"""
from __future__ import annotations

import argparse
import json
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
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg  # noqa: E402

from mp3_walk_policy import sku2, terrain  # noqa: E402
from mp3_walk_policy.contract import Contract  # noqa: E402


def build(sub_terrains: dict, rows: int, seed: int, size: float):
    cfg = TerrainGeneratorCfg(size=(size, size), sub_terrains=sub_terrains, num_rows=rows, curriculum=True,
                              border_width=0.5, seed=seed)
    gen = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    gen.compile(spec)
    return spec, gen


def add_robots(spec: mujoco.MjSpec, places: list[np.ndarray], yaw: float = 0.0) -> list[str]:
    prefixes = []
    for k, p in enumerate(places):
        robot = sku2.get_spec(strip_visuals=False)
        robot.body("base").pos = [0.0, 0.0, 0.0]   # the MJCF places the base 0.2 m up; the frame sets the height
        robot.compile()  # settles get_spec's keyframe deletion before attaching (else each copy brings 'stand')
        frame = spec.worldbody.add_frame(pos=[float(p[0]), float(p[1]), float(p[2]) + sku2.INIT_BASE_Z],
                                         quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        prefix = f"r{k}/"
        spec.attach(robot, frame=frame, prefix=prefix)
        prefixes.append(prefix)
    return prefixes


def pose_stand(model, data, prefixes) -> None:
    c = Contract.load()
    for pre in prefixes:
        for name, q in zip(c.joint_order, c.stand_joint_rad):
            data.qpos[model.joint(pre + name).qposadr[0]] = q


def shoot(model, data, lookat, distance, az, el, size=(640, 400)) -> np.ndarray:
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)  # mjlab's lights are spaced for 8 m tiles
    model.vis.headlight.diffuse[:] = (0.55, 0.55, 0.55)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])
    r = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, az, el
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    r.update_scene(data, camera=cam, scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=10, help="difficulty levels")
    ap.add_argument("--levels", type=int, nargs="+", help="rows to show close up (default: first, middle, last)")
    ap.add_argument("--only", nargs="+", help="sub-terrain names to include")
    ap.add_argument("--strip", action="store_true", help="the printed-tile strip instead of a preset")
    ap.add_argument("--preset", default="rough_basic", choices=tuple(terrain.PRESETS),
                    help="the sub-terrain set (terrain.PRESETS)")
    ap.add_argument("--layout", default="mirror_x")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=float, default=terrain.TILE_SIZE)
    ap.add_argument("--json", help="write tile origins and robot contacts")
    args = ap.parse_args()

    if args.strip:
        subs = {"strip": terrain.TileStripCfg(size=(args.size, args.size), layout=args.layout)}
        args.rows, levels = 1, [0]
    else:
        subs = terrain.PRESETS[args.preset](args.size)
        if args.only:
            subs = {k: v for k, v in subs.items() if k in args.only}
        levels = args.levels or sorted({0, args.rows // 2, args.rows - 1})
    spec, gen = build(subs, args.rows, args.seed, args.size)
    names = list(subs)
    places = [(r, c, gen.terrain_origins[r, c]) for c in range(len(names)) for r in levels]
    prefixes = add_robots(spec, [p for _, _, p in places])
    for key in list(spec.keys):  # each attached robot brings its keyframe; the pose is set below instead
        spec.delete(key)
    model = spec.compile()
    data = mujoco.MjData(model)
    pose_stand(model, data, prefixes)
    mujoco.mj_forward(model, data)

    report = []
    for (r, c, p), pre in zip(places, prefixes):
        robot_geoms = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith(pre)}
        n = sum(1 for i in range(data.ncon) if data.contact[i].geom1 in robot_geoms or data.contact[i].geom2 in robot_geoms)
        minpen = min([data.contact[i].dist for i in range(data.ncon)
                      if data.contact[i].geom1 in robot_geoms or data.contact[i].geom2 in robot_geoms] or [0.0])
        report.append({"terrain": names[c], "row": r, "origin": [round(float(x), 4) for x in p], "contacts": n,
                       "deepest_mm": round(1000 * minpen, 2)})
        print(f"{names[c]:12s} row {r:2d}: origin z {p[2] * 1000:6.1f} mm, robot contacts {n:2d} "
              f"(deepest {1000 * minpen:.1f} mm)")

    panels = []
    extent = np.array([gen.cfg.num_rows * args.size, len(names) * args.size])
    panels.append(("overview from above (difficulty rises along x, left to right; columns bottom to top: "
                   + ", ".join(names) + ")",
                   shoot(model, data, [0, 0, 0], 0.75 * extent.max(), 90, -89.9, size=(1280, 720))))
    for (r, c, p), pre in zip(places, prefixes):
        base = data.xpos[model.body(pre + "base").id]
        for az, el, tag in ((90, -12, "side"), (135, -30, "iso")):
            panels.append((f"{names[c]} row {r} ({tag})", shoot(model, data, base, 0.55, az, el)))
    cols = 4
    nrow = (len(panels) - 1 + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4.2, 9.5 + nrow * 2.9))
    gs = fig.add_gridspec(1 + nrow, cols, height_ratios=[3.2] + [1] * nrow)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(panels[0][1])
    ax.set_title(panels[0][0], fontsize=9)
    ax.axis("off")
    for k, (t, img) in enumerate(panels[1:]):
        ax = fig.add_subplot(gs[1 + k // cols, k % cols])
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"wrote {args.out}; model: {model.nhfield} heightfields, {model.ngeom} geoms")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
