"""Where the inputs live: this repository's data, and the sibling checkouts of mpx-sim (the SKU2 model and the stock
servo profile), bam (the MD01 actuator models) and minipupperesp (the firmware gait, for the teacher). Override a
sibling's location with MPX_SIM_DIR, BAM_DIR or MINIPUPPERESP_DIR.

The robot (read once, when Python starts): MP3_ROBOT=sku2 (the default: mpx-sim's SKU2 model and stock servo profile,
data/sku2_masses.json; the terrain task's default mass file is sku2_masses_478g.json) or MP3_ROBOT=sku1v2 (SKU1's body:
minipupperesp's sim/model/pupper.xml, built from sku_1_urdf_fixed, its stock servo profile, and
data/sku1v2_masses.json: 390 g, 195 / 195 g on the front and rear feet). Both share the legs, the MD01 modules, the
IMU and the servo mapping, so the contract (data/sku2_contract.json) and the BAM actuator models are the same; they
differ in the body and its masses. mpx-sim is not needed for sku1v2. See docs/SKU1V2_STAIRS.md."""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "mp3_walk_policy" / "data"
MPX_SIM = Path(os.environ.get("MPX_SIM_DIR", REPO.parent / "mpx-sim"))
BAM = Path(os.environ.get("BAM_DIR", REPO.parent / "bam"))
MINIPUPPERESP = Path(os.environ.get("MINIPUPPERESP_DIR", REPO.parent / "minipupperesp"))
ROBOTS = ("sku2", "sku1v2")
ROBOT = os.environ.get("MP3_ROBOT", "sku2").strip().lower()
if ROBOT not in ROBOTS:
    raise ValueError(f"MP3_ROBOT={ROBOT!r}: one of {', '.join(ROBOTS)}")
SKU1_BODY = ROBOT == "sku1v2"                     # minipupperesp's SKU1 model
SKU2_XML = MPX_SIM / "src/mpx_sim/model/sku_2/pupper.xml"
SKU1_XML = MINIPUPPERESP / "sim/model/pupper.xml"
ROBOT_XML = SKU1_XML if SKU1_BODY else SKU2_XML
SERVO_PROFILES = (MINIPUPPERESP / "sim/model/servo_profiles.json" if SKU1_BODY
                  else MPX_SIM / "src/mpx_sim/model/servo_profiles.json")
# MP3_MASSES=<a file in mp3_walk_policy/data> (read once, when Python starts): every task's mass file, the flat walks'
# and the terrain's, e.g. sku1v2_masses_405g.json for SKU1 v2 as weighed; unset, the robot's own as below
MASSES_OVERRIDE = os.environ.get("MP3_MASSES", "").strip()
if MASSES_OVERRIDE and not (DATA / MASSES_OVERRIDE).is_file():
    raise FileNotFoundError(f"MP3_MASSES={MASSES_OVERRIDE}: no such file in {DATA}")
MASSES_JSON = DATA / (MASSES_OVERRIDE or f"{ROBOT}_masses.json")
# the terrain task's default mass file (terrain_task.TerrainOptions.masses_json)
TERRAIN_MASSES = MASSES_OVERRIDE or ("sku1v2_masses.json" if SKU1_BODY else "sku2_masses_478g.json")
RUNS = REPO / "runs"


TEXT_SUFFIXES = {".xml", ".json", ".c", ".h", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout (core.autocrlf) gives
    the same digest as a Linux one."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()
