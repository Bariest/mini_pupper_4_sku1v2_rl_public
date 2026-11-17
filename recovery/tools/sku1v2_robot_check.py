#!/usr/bin/env python
"""The SKU1 v2 robot as the training builds it (MP4R_ROBOT=sku1v2): the model, the weighed masses, the stand, the
contract, the joint limits and the hip envelope. CPU only (MuJoCo), a second.

    MP4R_ROBOT=sku1v2 python -B tools/sku1v2_robot_check.py
"""
import json
import sys

import mujoco

from mp4_recovery import limits, paths
from mp4_recovery.contract import Contract
from mp4_recovery.robot import sku2


def main() -> int:
    if paths.ROBOT != "sku1v2":
        raise SystemExit("set MP4R_ROBOT=sku1v2 first")
    m = sku2.get_spec(limits=True, self_collision=True).compile()
    c = Contract.load()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.2
    for jn, v in zip(c.joint_order, c.stand_joint_rad):
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    feet = {leg: d.geom_xpos[m.geom(f"{leg}_foot").id] for leg in sku2.LEGS}
    rad = float(m.geom_size[m.geom("FR_foot").id][0])
    rear = (feet["FR"][0] - com[0]) / (feet["FR"][0] - feet["RR"][0])
    z_stand = 0.2 - min(f[2] for f in feet.values()) + rad
    mj = json.loads(paths.MASSES_JSON.read_text())
    print(f"model {paths.SKU2_XML.relative_to(paths.REPO)} (sha256 {paths.sha256(paths.SKU2_XML)[:12]})")
    print(f"built {1000 * m.body_mass.sum():.1f} g, rear feet {100 * rear:.2f} % (weighed: "
          f"{1000 * mj['total_kg']:.0f} g, {100 * mj['rear_share']:.2f} %)")
    print(f"base origin {1000 * z_stand:.1f} mm above the floor at the stand; the starts at {1000 * sku2.INIT_BASE_Z:.1f} mm")
    print(f"contract {paths.CONTRACT_JSON.name} (sha256 {paths.sha256(paths.CONTRACT_JSON)[:12]}): offsets "
          f"{[round(float(v), 2) for v in c.offsets_deg]}")
    b = limits.decoder_bounds("getup")
    print(f"joint limits {limits.LIMITS_JSON.name} (estimated from SKU2's), hip envelope "
          f"{'yes' if b.hip_envelope else 'NO'}")
    ok = abs(m.body_mass.sum() - mj["total_kg"]) < 1e-4 and abs(rear - mj["rear_share"]) < 2e-3 and \
        abs(z_stand - sku2.INIT_BASE_Z) < 0.003
    print("robot OK" if ok else "ROBOT NOT AS EXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
