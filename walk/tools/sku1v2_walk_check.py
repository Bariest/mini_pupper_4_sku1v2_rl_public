"""Before the long SKU1 v2 walk training (tools/train_sku1v2_walk.ps1): the firmware gait the teacher uses (minipupperesp's
stanford_gait.c / stanford_kinematics.c; on Windows the prebuilt data/libgait_win64_<hash>.dll, else MSVC), a small
clone collection on the take-like schedules with stops (the teacher must walk this body without falling), and 3 PPO
iterations from that clone. Everything goes to runs/sku1v2_walk_check/ and is deleted after.

    $env:MP3_ROBOT = "sku1v2"; & $PY -B tools\\sku1v2_walk_check.py [--device cpu]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from mp3_walk_policy import distill, firmware_gait, paths
    if paths.ROBOT != "sku1v2":
        print(f"MP3_ROBOT={paths.ROBOT}: set MP3_ROBOT=sku1v2")
        return 1
    firmware_gait.library()
    print(f"firmware gait OK (sources {firmware_gait.sources_hash()}, {firmware_gait.FW})")
    out = REPO / "runs" / "sku1v2_walk_check"
    shutil.rmtree(out, ignore_errors=True)
    data = distill.collect("bam_t5ub06", 64, action_noise=0.06, seed=0, device=a.device, omni=True, stops=True)
    falls = data["meta"]["falls"]
    print(f"teacher on SKU1 v2: {data['meta']['rows']} rows, {falls} falls of 64")
    ckpt, meta = distill.train(data, epochs=2, seed=0, device=a.device)
    meta.update({"collect": data["meta"]})
    distill.save_clone(ckpt, meta, out / "clone" / "clone.pt")
    r = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "train.py"), "--arm", "bam_t5ub06", "--stops",
                        "--clone", str(out / "clone" / "clone.pt"), "--seed", "1", "--beta", "20", "--fixed-std",
                        "--critic-warmup", "50", "--lr0", "1e-4", "--iters", "3", "--envs", "64",
                        "--device", a.device, "--out", str(out / "ppo"), "--tag", "check"], cwd=REPO)
    if r.returncode != 0:
        print("the PPO check failed")
        return 1
    shutil.rmtree(out, ignore_errors=True)
    if falls > 3:
        print(f"the firmware teacher fell {falls} times of 64 on this body: tell Claude before training")
        return 1
    print("walk check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
