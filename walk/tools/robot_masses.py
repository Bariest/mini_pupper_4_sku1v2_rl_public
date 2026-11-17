"""Write data/<robot>_masses.json from the robot on a kitchen scale.

The training model takes the robot's real masses (sku2.apply_masses): each MD01 module and each leg as SKU2 weighed
them (the same parts on SKU1 and SKU2), the base gets the rest of the total, and the base's centre of mass is moved
along x until the model, standing on the contract's stand, puts the weighed share on the rear feet.

Weigh the robot standing (servos on, at its stand), battery in:
  total  the whole robot on the scale;
  front  only the front feet on the scale, the rear feet on something of the scale's height;
  rear   only the rear feet on the scale, the front feet on that support.
front + rear is usually a few grams off the total; only their ratio is used.

    python tools/robot_masses.py --robot sku1v2 --total-g 390 --front-g 195 --rear-g 195     # SKU1 v2 (the shipped file)
    python tools/robot_masses.py --robot sku1v2 --total-g 390          # no split: SKU2's 56.6 % on the rear

Run it with MP3_ROBOT unset or equal to --robot (it picks the model: sku1v2 = minipupperesp's SKU1 model). Check
against SKU2: --robot sku2 --total-g 450 --front-g 197.3 --rear-g 257.8 gives SKU2's base_com_x_m (-0.02313) on
mpx-sim's model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

SKU2_REAR_SHARE = 257.8 / (197.3 + 257.8)
MODULE_KG = 0.06682      # a bare MD01 module (sku2_masses.json)
LEG_KG = 0.0065          # a leg's thigh and calf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=("sku2", "sku1v2"))
    ap.add_argument("--total-g", type=float, required=True)
    ap.add_argument("--front-g", type=float)
    ap.add_argument("--rear-g", type=float)
    ap.add_argument("--out", help="default: mp3_walk_policy/data/<robot>_masses.json")
    a = ap.parse_args()
    if os.environ.get("MP3_ROBOT", a.robot).strip().lower() != a.robot:
        sys.exit(f"MP3_ROBOT={os.environ['MP3_ROBOT']} but --robot {a.robot}: unset one of them")
    os.environ["MP3_ROBOT"] = a.robot
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mp3_walk_policy import paths
    from mp3_walk_policy.contract import Contract
    from mp3_walk_policy.sku2 import FOOT_SITES, LEGS, apply_masses

    if (a.front_g is None) != (a.rear_g is None):
        sys.exit("give both --front-g and --rear-g, or neither")
    if a.front_g is not None:
        if a.front_g <= 0 or a.rear_g <= 0:
            sys.exit("the front and rear weights must be positive")
        share = a.rear_g / (a.front_g + a.rear_g)
        split = f"the front and rear foot pairs {a.front_g:g} / {a.rear_g:g} g at the stand"
    else:
        share = SKU2_REAR_SHARE
        split = f"no front/rear weighing: SKU2's rear share {100 * share:.1f} %"
    total = a.total_g / 1000.0
    if not 0.2 < total < 2.0:
        sys.exit(f"{a.total_g:g} g does not look like a Mini Pupper")

    spec = mujoco.MjSpec.from_file(str(paths.ROBOT_XML))
    spec.delete(spec.body("g_base"))
    for k in list(spec.keys):
        spec.delete(k)
    for x in list(spec.actuators):
        spec.delete(x)
    for x in list(spec.sensors):
        spec.delete(x)
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    apply_masses(spec, {"total_kg": total, "module_kg": MODULE_KG, "leg_kg": LEG_KG, "base_com_x_m": 0.0})
    if spec.body("base").mass <= 0:
        sys.exit(f"{a.total_g:g} g leaves nothing for the base")

    def com_and_feet(base_com_x):
        ipos = np.array(spec.body("base").ipos)
        ipos[0] = base_com_x
        spec.body("base").ipos = ipos
        m = spec.compile()
        d = mujoco.MjData(m)
        c = Contract.load()
        for n, q in zip(c.joint_order, c.stand_joint_rad):
            d.qpos[m.joint(n).qposadr[0]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mass = m.body_mass[1:]
        com_x = float((mass * d.xipos[1:, 0]).sum() / mass.sum())
        fx = {leg: float(d.site_xpos[m.site(s).id][0]) for leg, s in zip(LEGS, FOOT_SITES)}
        return com_x, 0.5 * (fx["FR"] + fx["FL"]), 0.5 * (fx["RR"] + fx["RL"]), float(mass.sum())

    c0, xf, xr, msum = com_and_feet(0.0)
    c1, _, _, _ = com_and_feet(0.01)
    target = xf - share * (xf - xr)                       # the COM x that loads the rear pair with `share`
    base_com_x = 0.01 * (target - c0) / (c1 - c0)          # the COM is linear in the base's COM x
    check, _, _, _ = com_and_feet(base_com_x)
    rear_back = (xf - check) / (xf - xr)
    out = {"robot": a.robot, "total_kg": round(total, 6), "module_kg": MODULE_KG, "leg_kg": LEG_KG,
           "rear_share": round(share, 4), "base_com_x_m": round(base_com_x, 5),
           "source": f"{a.robot.upper()}: {a.total_g:g} g total, {split}; a leg 6.5 g and a bare MD01 module "
                     f"66.82 g as SKU2 (the same parts); the base COM x is solved for the rear share "
                     f"(tools/robot_masses.py, model {paths.ROBOT_XML.name} sha256 {paths.sha256(paths.ROBOT_XML)[:12]})"}
    if a.front_g is not None:
        out["front_pair_kg"], out["rear_pair_kg"] = round(a.front_g / 1000.0, 6), round(a.rear_g / 1000.0, 6)
    path = Path(a.out) if a.out else paths.DATA / f"{a.robot}_masses.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"model mass {msum * 1000:.1f} g, rear share back {100 * rear_back:.2f} %, written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
