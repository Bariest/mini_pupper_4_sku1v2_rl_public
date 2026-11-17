"""Compare fitted MD01 models on the quantities the data can identify.

Raw kt and R are a gauge in position data (only kt/R and kt*kt_emf/R are
visible), so two fits can disagree on both while describing the same actuator.
This prints, for each params file, what a joint actually feels in the robot's
configuration (control_mode 1, kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V):

- stiffness: motor torque / error at rest, secant at 1, 2 and 5 deg [Nm/rad]
  (1 deg sits in the conduction knee, where a holding joint works)
- damping at a hold (5 deg of error, +-0.5 rad/s): the motor's (back-EMF + D
  term) and the fitted viscous friction [Nm.s/rad]; the idle bridge's braking
  kt*kt_emf/R
- torque ceiling with the duty saturated, at 0, 2 and 5 rad/s [Nm]
- static friction budget at rest holding 0.05, 0.13, 0.25 and 0.36 Nm (a
  holding joint's load and the bench blocks' gravity torques) [Nm]
- armature [kg m^2], and the rig terms q_offset and command_delay

    .venv/bin/python bam/mangdang/tools/compare_params.py bam/params/md01/id_3/m6.json bam/params/md01/id_2/m6.json
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bam.model import load_model  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("params", nargs="+", help="params JSON files")
parser.add_argument("--kp", type=float, default=80.0)
parser.add_argument("--kd", type=float, default=400.0)
args = parser.parse_args()

ROBOT = {
    "mass": 0.12, "arm_mass": 0.016, "length": 0.10, "kp": args.kp, "kd": args.kd, "vin": 12.0,
    "at32_kff_current": 0.001, "at32_max_pwm_duty_cycle": 0.99, "at32_kd_position": args.kd,
    "at32_control_mode": 1, "cur_cap_ma": 1500, "supply_v": 12.0, "dt": 0.005, "entries": [],
}
DT = 0.005


def invariants(path):
    model = load_model(path)
    act = model.actuator
    act.load_log(dict(ROBOT))

    def motor_torque(error, dq):
        control = act.compute_control(error, 0.0, dq, DT)
        return float(act.compute_torque(control, True, 0.0, dq))

    out = {}
    for deg in (1.0, 2.0, 5.0):
        e = math.radians(deg)
        out[f"stiffness @{deg:g} deg [Nm/rad]"] = motor_torque(e, 0.0) / e
    e = math.radians(5.0)
    out["motor damping at a hold [Nm.s/rad]"] = -(motor_torque(e, 0.5) - motor_torque(e, -0.5)) / 1.0
    out["viscous friction [Nm.s/rad]"] = model.friction_viscous.value if hasattr(model, "friction_viscous") else 0.0
    kt_emf = model.kt_emf.value if hasattr(model, "kt_emf") else model.kt.value
    out["idle braking kt*kt_emf/R [Nm.s/rad]"] = model.kt.value * kt_emf / model.R.value
    for w in (0.0, 2.0, 5.0):
        out[f"ceiling @{w:g} rad/s [Nm]"] = motor_torque(math.radians(60.0), w)
    for load in (0.05, 0.13, 0.25, 0.36):
        frictionloss, _ = model.compute_frictions(load, -load, 0.0)
        out[f"static band holding {load:g} Nm [Nm]"] = float(frictionloss)
    out["armature [kg m^2]"] = model.armature.value
    out["q_offset [rad]"] = model.q_offset.value
    out["command_delay [s]"] = model.command_delay.value if hasattr(model, "command_delay") else 0.0
    out["kt / kt_emf / R (gauge)"] = f"{model.kt.value:.3f} / {kt_emf:.3f} / {model.R.value:.2f}"
    return out


columns = [(Path(p).parent.name + "/" + Path(p).stem, invariants(p)) for p in args.params]
width = max(len(k) for k in columns[0][1])
print(" " * width + "".join(f"  {name:>18s}" for name, _ in columns))
for key in columns[0][1]:
    cells = []
    for _, inv in columns:
        v = inv[key]
        cells.append(f"  {v:>18.4g}" if isinstance(v, float) else f"  {v:>18s}")
    print(f"{key:<{width}s}" + "".join(cells))
