"""Silence mujoco_warp's heightfield-overflow warning in the venv it runs from (idempotent; ``--check`` only reports).

    python tools/patch_mujoco_warp.py [--check]

When a geom touches more than 50 heightfield prisms (a fallen body on fine terrain), mujoco_warp 3.7.0.1 skips the rest
(``continue``) and prints a warning from the GPU kernel, for every geom pair and every physics substep. In training with
thousands of falling robots that is millions of lines: it slows the run several-fold and fills the disk (5.3 million
lines in one 30-iteration run). The patch removes only the print; the skip, and so the physics, is unchanged. The feet
never reach the cap (a 4.9 mm sphere on 5 mm cells). Re-run after reinstalling mujoco_warp. ``is_patched()`` is the
check ``scripts/train_terrain.py`` runs before training.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import mujoco_warp

PATH = Path(mujoco_warp.__file__).parent / "_src" / "collision_convex.py"
OLD = '''          if count >= MJ_MAXCONPAIR:
            wp.printf(
              "height field collision overflow, number of collisions >= %u - please adjust resolution: \\n decrease the number of hfield rows/cols or modify size of colliding geom\\n",
              MJ_MAXCONPAIR,
            )
            continue'''
NEW = '''          if count >= MJ_MAXCONPAIR:
            # (warning print removed; the contact is still skipped)
            continue'''
# the patched block, whatever its comment says
PATCHED = re.compile(r"if count >= MJ_MAXCONPAIR:\n[ ]*#[^\n]*\n[ ]*continue")


def is_patched(src: str | None = None) -> bool | None:
    """True when the warning print is removed, False when the original block is there, None when neither is found
    (another mujoco_warp version)."""
    src = PATH.read_text() if src is None else src
    if OLD in src:
        return False
    return True if PATCHED.search(src) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = PATH.read_text()
    print(f"{PATH} (mujoco_warp {getattr(mujoco_warp, '__version__', '?')}), sha256 "
          f"{hashlib.sha256(src.encode()).hexdigest()[:16]}")
    state = is_patched(src)
    if state:
        print("patched")
        return 0
    if state is None:
        print("the expected block is not there (another mujoco_warp version?): not patched")
        return 1
    if a.check:
        print("not patched")
        return 1
    PATH.write_text(src.replace(OLD, NEW))
    print("patched now, sha256", hashlib.sha256(PATH.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
