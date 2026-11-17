"""Walk a deployed policy on SKU2 by hand, from its Raspberry Pi (numpy and mp3link only).

    ssh -t <pi> ~/walk/tools/walk list
    ssh -t <pi> ~/walk/tools/walk <policy> [--speed 0.10] [--seconds 20 | 0] [--allow-fast] [--log NAME] [--dry]

<policy> is a directory under ~/walk/policies (a unique prefix is enough). The command writes the servo
configuration the policies were trained for (mode 1, kp 80, kd 400, kff 0.001, max_pwm 0.99; RAM only), then runs
``runner run``: 3 s at the stand, the policy walking forward at --speed for --seconds (0 = until Enter), the ramp
back to the stand, 3 s at the stand, and the servos idle (the body drops about 7 cm).

The policies were trained to walk forward at 0.05-0.20 m/s only: no turning, no sideways command. On SKU2 they tend to
drift right, so a long walk curves; keep the floor clear to the right.

Stopping:
- Enter: the walk ends, the robot ramps to the stand, holds and sits (needs a terminal: an armed walk refuses to
  start without one, so use ssh -t).
- Ctrl-C: the servos idle at once (the body drops).
- Tipping the robot past 35 deg idles the servos at once. Lifting it level does NOT stop it.
- If the network drops mid-walk, Enter cannot reach the Pi: tip the robot.
The runner also refuses to start above 45 degC, without the IMU or over a latched fault, and stops at 55 degC.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import runner as rn

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "policies"
LOGS = ROOT / "logs" / "demo"
SPEED_MIN, SPEED_MAX, SPEED_FAST = 0.05, 0.20, 0.25          # the trained range; --allow-fast up to SPEED_FAST
UNTIL_ENTER_S = 1.0e6


def deployed() -> list[str]:
    return sorted(d.name for d in POLICIES.iterdir() if (d / "policy.npz").exists()) if POLICIES.is_dir() else []


def resolve(name: str) -> Path:
    if (Path(name) / "policy.npz").exists():
        return Path(name)
    names = deployed()
    hits = [n for n in names if n == name] or [n for n in names if n.startswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"policy {name!r}: {'ambiguous: ' + ', '.join(hits) if hits else 'not found'}; "
                         f"deployed: {', '.join(names) or 'none'}")
    return POLICIES / hits[0]


def check_speed(speed: float, allow_fast: bool) -> None:
    if not math.isfinite(speed) or speed < SPEED_MIN:
        raise SystemExit(f"--speed {speed}: the policies were trained at {SPEED_MIN}-{SPEED_MAX} m/s")
    if speed > SPEED_MAX:
        if not allow_fast or speed > SPEED_FAST:
            raise SystemExit(f"--speed {speed}: above the trained {SPEED_MAX} m/s needs --allow-fast "
                             f"(at most {SPEED_FAST})")
        print(f"WARNING: {speed} m/s is above the trained range; it may stumble or turn hard", flush=True)


def log_name(name: str) -> str:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("NAME: letters, digits, _ and - only")
    return name


def write_config(link) -> None:
    from mp3link.cli import cmd_config
    v = rn.IDENTIFIED
    cmd_config(link, argparse.Namespace(ids="1-12", mode=v["control_mode"], kp=v["kp_position"],
                                        kd=v["kd_position"], kff=v["kff_current"], max_pwm=v["max_pwm_duty_cycle"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="walk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", help="a deployed policy (name or unique prefix), or 'list'")
    ap.add_argument("--speed", type=float, default=0.10, help="forward m/s (0.05-0.20)")
    ap.add_argument("--seconds", type=float, default=20.0, help="walking time; 0 = until Enter")
    ap.add_argument("--allow-fast", action="store_true", help=f"allow up to {SPEED_FAST} m/s (untrained)")
    ap.add_argument("--log", type=log_name, help="save the run under ~/walk/logs/demo/NAME")
    ap.add_argument("--dry", action="store_true",
                    help="never arm: the servos stay idle (the configuration is still written)")
    ap.add_argument("--hold-before", type=float, default=3.0)
    ap.add_argument("--port", default="/dev/ttyAMA2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    a = ap.parse_args(argv)
    if a.policy == "list":
        print("\n".join(deployed()) or "no policy deployed")
        return 0
    pdir = resolve(a.policy)
    check_speed(a.speed, a.allow_fast)
    if not math.isfinite(a.seconds) or a.seconds < 0:
        raise SystemExit(f"--seconds {a.seconds}: a number of seconds, or 0 for until Enter")
    if not a.dry and not sys.stdin.isatty():
        raise SystemExit("an armed walk needs a terminal for Enter, the graceful stop: use ssh -t")
    seconds = a.seconds if a.seconds > 0 else UNTIL_ENTER_S
    if a.log and (LOGS / f"{a.log}.npz").exists():
        raise SystemExit(f"log {a.log} exists: choose another name")
    from mp3link.link import Link
    from .export import NumpyPolicy
    pol = NumpyPolicy(pdir / "policy.npz")
    link = Link(a.port, a.baud)
    try:
        try:
            link.wait_state(1.0)
        except Exception as exc:
            raise SystemExit(f"no STATE from the ESP32 ({exc}): is it running mp3-link? Check with `mp3link ping`")
        write_config(link)
        run = argparse.Namespace(cmd="run", policy=str(pdir), vx=a.speed, vy=0.0, wz=0.0, seconds=seconds, mount=None, dry=a.dry,
                                 cap=rn.CAP_MA, hold_before=a.hold_before, hold_after=3.0, clear=False,
                                 log=str(LOGS / a.log) if a.log else None)
        until = "until Enter" if seconds == UNTIL_ENTER_S else f"{seconds:g} s"
        print(f"{pdir.name}: {a.speed:.2f} m/s, {until}{' (dry: servos idle)' if a.dry else ''}. "
              "Enter: stand and sit; Ctrl-C: idle at once; tip past 35 deg: idle at once; a level lift does not stop it",
              flush=True)
        return rn._run_common(link, run, lambda obs, k: pol(obs), {"cmd": "demo"})
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
