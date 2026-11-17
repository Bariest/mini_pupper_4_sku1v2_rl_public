"""Where the package's inputs live.

The robot (read once, when Python starts): ``MP4R_ROBOT=sku2`` (the default: the Mini Pupper v4 SKU2 this repository was
made for) or ``MP4R_ROBOT=sku1v2`` (SKU1 v2: SKU1's body, minipupperesp's ``sim/model/pupper.xml`` in ``robot/assets/sku_1``,
the same legs, MD01 modules, IMU and servo mapping, 405 g as weighed). Each robot has its own model, masses, contract, joint
limits, hip envelope and searched motions (``data/<robot>_*.json``, ``data/searched_motions[_<robot>]``)."""
import hashlib
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]     # the source checkout: assumes the editable install the installer makes
DATA = PACKAGE / "data"
ASSETS = PACKAGE / "robot" / "assets"
ROBOTS = ("sku2", "sku1v2")
ROBOT = os.environ.get("MP4R_ROBOT", "sku2").strip().lower() or "sku2"
if ROBOT not in ROBOTS:
    raise ValueError(f"MP4R_ROBOT={ROBOT!r}: one of {', '.join(ROBOTS)}")
SKU1V2 = ROBOT == "sku1v2"
SKU2_XML = ASSETS / ("sku_1" if SKU1V2 else "sku_2") / "pupper.xml"      # the robot's model (the name is historical)
ROBOT_XML = SKU2_XML
MASSES_JSON = DATA / ("sku1v2_masses_405g.json" if SKU1V2 else "sku2_masses.json")
TASK_MASSES = "sku1v2_masses_405g.json" if SKU1V2 else "sku2_masses_478g.json"   # the tasks' default mass file
CONTRACT_JSON = DATA / f"{ROBOT}_contract.json"
LIMITS_JSON = DATA / f"{ROBOT}_joint_limits.json"
HIP_ENVELOPE_JSON = DATA / f"{ROBOT}_hip_envelope.json"
SEARCHED_MOTIONS = PACKAGE.parents[1] / "data" / ("searched_motions_sku1v2" if SKU1V2 else "searched_motions")
RUNS = REPO / "runs"
POLICIES = REPO / "policies"


def git(*args, cwd=None) -> str:
    """``git args`` in ``cwd`` (default: the repository), its stdout stripped; "" without git or outside a checkout."""
    import subprocess
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(cwd or REPO)).stdout.strip()
    except OSError:
        return ""


TEXT_SUFFIXES = {".xml", ".json", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout gives the same digest."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()
