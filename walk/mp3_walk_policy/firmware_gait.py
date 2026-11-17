"""minipupperesp's own Stanford trot and IK, built on the host (the teacher's source).

The library is compiled from the sibling minipupperesp checkout's main/stanford_gait.c and stanford_kinematics.c
(unchanged) plus gait_host.c (the gait step and servo_write's tick formula, main.c:2526), with -ffp-contract=off,
into ~/.cache/mp3_walk_policy_gait/ (keyed by the sources' hash). Needs gcc, or on Windows MSVC (Build Tools, found
with vswhere). On Windows a prebuilt library for the same sources, data/libgait_win64_<hash>.dll (gcc/mingw-w64,
-O2 -ffp-contract=off, as the Linux and Pi builds), is used first, so no compiler is needed for it. The hash reads
the sources with LF line endings, so a Windows (CRLF) and a Linux copy of the same sources give the same hash.
"""
import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import paths

HERE = Path(__file__).resolve().parent
FW = paths.MINIPUPPERESP / "main"
SOURCES = [HERE / "gait_host.c", FW / "stanford_gait.c", FW / "stanford_kinematics.c"]
NEUTRAL_Z, CLEARANCE, DT = 70.0, 20.0, 0.015


EXPORTS = ("gh_reset", "gh_step", "gh_ticks")


def _vcvars64() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    root = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return Path(root) / "VC/Auxiliary/Build/vcvars64.bat"


def _build_msvc(lib: Path) -> None:
    """MSVC's /fp:precise does not contract a*b+c into an FMA (the gcc build's -ffp-contract=off)."""
    exports = " ".join(f"/EXPORT:{e}" for e in EXPORTS)
    srcs = " ".join(f'"{p}"' for p in SOURCES)
    cmd = (f'call "{_vcvars64()}" >nul && cl /nologo /O2 /fp:precise /LD /I"{FW}" {srcs} '
           f'/Fe"{lib}" /link {exports}')
    subprocess.run(cmd, shell=True, check=True, cwd=lib.parent)


def sources_hash() -> str:
    missing = [str(p) for p in SOURCES + [FW / "stanford_gait.h", FW / "stanford_kinematics.h"] if not p.exists()]
    if missing:
        raise FileNotFoundError("the firmware gait's sources are missing (minipupperesp/main, MINIPUPPERESP_DIR): "
                                + ", ".join(missing))
    return hashlib.sha256(b"".join(p.read_bytes().replace(b"\r\n", b"\n")
                                   for p in SOURCES + [FW / "stanford_gait.h",
                                                       FW / "stanford_kinematics.h"])).hexdigest()[:12]


def library():
    h = sources_hash()
    windows = sys.platform == "win32"
    lib = Path.home() / ".cache/mp3_walk_policy_gait" / f"libgait_{h}.{'dll' if windows else 'so'}"
    prebuilt = HERE / "data" / f"libgait_win64_{h}.dll"
    if windows and not lib.exists() and prebuilt.exists():
        lib = prebuilt
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        if windows:
            _build_msvc(lib)
        else:
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-ffp-contract=off", f"-I{FW}", *map(str, SOURCES),
                            "-lm", "-o", str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    f = ctypes.c_float
    so.gh_reset.argtypes = [f]
    so.gh_step.argtypes = [f, f, f, f, f, ctypes.POINTER(f)]
    so.gh_ticks.argtypes = [ctypes.POINTER(f), ctypes.POINTER(f), ctypes.POINTER(ctypes.c_int)]
    return so


class Gait:
    def __init__(self, offsets_deg):
        self.so = library()
        self.off = (ctypes.c_float * 13)(0.0, *[float(v) for v in offsets_deg])

    def reset(self, height=NEUTRAL_Z):
        self.so.gh_reset(height)

    def step(self, vx=0.0, vy=0.0, wz=0.0, height=NEUTRAL_Z, clearance=CLEARANCE):
        feet = (ctypes.c_float * 12)()
        self.so.gh_step(vx, vy, wz, height, clearance, feet)
        return np.array(feet[:]).reshape(4, 3)

    def ticks(self, feet):
        fe = (ctypes.c_float * 12)(*[float(v) for v in np.asarray(feet).ravel()])
        out = (ctypes.c_int * 13)()
        self.so.gh_ticks(fe, self.off, out)
        return np.array(out[1:])


LATERAL_MM = 49.5          # SG_ORIGIN_Y: LEG_LR 23.5 + ABDUCTION_OFFSET 26 (stanford_gait.h); foot under it at neutral


def neutral_feet(dx=0.0, dz=0.0, roll_deg=0.0):
    """The four feet (FR, FL, RR, RL) in each hip's frame (x forward, y left, z body height above the foot) for a
    body shifted by dx (feet move the other way), raised by dz, and rolled by roll_deg (left side up) about its x
    axis at hip level, with the feet fixed on the ground."""
    feet = np.zeros((4, 3))
    feet[:, 0] = -dx
    side = np.array([-1.0, 1.0, -1.0, 1.0])
    y, zb = side * LATERAL_MM, -(NEUTRAL_Z + dz) * np.ones(4)       # body frame, z up
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    y2, z2 = c * y + s * zb, -s * y + c * zb                         # the fixed feet seen from the rolled body
    feet[:, 1] = y2 - side * LATERAL_MM
    feet[:, 2] = -z2
    return feet
