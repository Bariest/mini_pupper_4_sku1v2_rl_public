"""The SKU1 v2 robot as training builds it (MP3_ROBOT=sku1v2): which model file, the total mass, the share on the rear
feet at the stand and the hip height, against data/sku1v2_masses.json. check.ps1 runs it; it needs no GPU.

    $env:MP3_ROBOT = "sku1v2"; & $PY -B tools\\sku1v2_check.py [--masses sku1v2_masses_405g.json]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                                            # noqa: E402
import numpy as np                                                       # noqa: E402


def main() -> int:
    from mp3_walk_policy import paths
    from mp3_walk_policy.contract import Contract
    from mp3_walk_policy.sku2 import FOOT_SITES, LEGS, get_spec, masses
    from mp3_walk_policy.terrain_task import TerrainOptions
    if paths.ROBOT != "sku1v2":
        print(f"MP3_ROBOT={os.environ.get('MP3_ROBOT')}: set MP3_ROBOT=sku1v2")
        return 1
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--masses", help="a mass file in mp3_walk_policy/data (default: the terrain task's)")
    a = ap.parse_args()
    opt = TerrainOptions(**({"masses_json": a.masses} if a.masses else {}))
    m_file = masses(opt.masses_json)
    spec = get_spec(strip_visuals=True, masses_json=opt.masses_json, self_collision=True)
    m = spec.compile()
    d = mujoco.MjData(m)
    c = Contract.load()
    for n, q in zip(c.joint_order, c.stand_joint_rad):
        d.qpos[m.joint(n).qposadr[0]] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mass = m.body_mass[1:]
    total = float(mass.sum())
    com_x = float((mass * d.xipos[1:, 0]).sum() / total)
    fx = {leg: d.site_xpos[m.site(s).id] for leg, s in zip(LEGS, FOOT_SITES)}
    xf = 0.5 * (fx["FR"][0] + fx["FL"][0])
    xr = 0.5 * (fx["RR"][0] + fx["RL"][0])
    rear = (xf - com_x) / (xf - xr)
    foot_r = float(m.geom_size[m.geom("FR_foot").id][0])
    hip_h = float(d.xpos[m.body("FR_thigh").id][2] - fx["FR"][2]) + foot_r     # above the ground
    print(f"model    {paths.ROBOT_XML}  (sha256 {paths.sha256(paths.ROBOT_XML)[:12]})")
    print(f"masses   {opt.masses_json}: {m_file['source']}")
    print(f"built    {1000 * total:.1f} g, rear feet {100 * rear:.1f} %, front {100 * (1 - rear):.1f} %, "
          f"hip axis {1000 * hip_h:.1f} mm above the ground, base {1000 * m.body_mass[m.body('base').id]:.1f} g")
    ok = abs(total - m_file["total_kg"]) < 1e-4 and abs(rear - m_file["rear_share"]) < 0.005 and paths.SKU1_BODY
    print("robot OK" if ok else "robot NOT as the mass file says")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
