# Deploying to the robot

The robot runs the policies on its ESP32-S3 with the **mpv2onlyRL** firmware. A policy is one `.mpxp` file, built into
the firmware from `main/rl/policies/`: the firmware's CMake embeds every known file name it finds there.

## The files

| File | App label (RL page) | Role |
|---|---|---|
| `default.mpxp` | Walk · SKU1 v2 · 405 g | the walk the robot boots with |
| `sku1v2_recovery.mpxp` | Walk + get-up · SKU1 v2 | walks, gets up after a fall and walks on |
| `getup_sku1.mpxp` | (none) | the get-up behind **Get up** and **Always gets back up** |
| `sku1v2_stairs.mpxp` | Stairs · SKU1 v2 | the stairs walk (forward only, heading held) |

With two get-ups built in (`getup.mpxp` for SKU2 and `getup_sku1.mpxp`), the firmware installs the one whose header
names its `APP_POLICY_ROBOT`. Every `.mpxp` from this repository is labelled `sku1`, because the firmware knows the
robots `sku1` and `sku2`, and SKU1 v2 is an `sku1`.

## Steps

```powershell
powershell -ExecutionPolicy Bypass -File deploy.ps1 -Firmware C:\esp\projects\mpv2onlyRL
cd C:\esp\projects\mpv2onlyRL
idf.py menuconfig          # App -> "This robot" = sku1 (only if it is not already)
idf.py reconfigure
idf.py build flash
```

`deploy.ps1 -Only getup_sku1.mpxp` deploys a single file. Files it replaces go to `out/replaced_<date>/`, so going back
is a copy the other way.

## On the robot

1. **Calibrate first** (Settings → Calibration). [ROBOT.md](ROBOT.md#calibration) lists the room the get-up needs.
2. RL page → **Self-test**: each policy's fixtures check the robot's forward pass, and the get-up's decoder self-test
   checks the bounds against your calibration.
3. Try each policy on a stand, then on the floor. Try the get-up on a soft surface first.
4. If a get-up fails, download the get-up log on the RL page (`getup_last.csv`: every servo's target and position).

## Memory

Every built-in policy lives in PSRAM with the firmware: the walk and walker are about 134 KB each, the stairs walk about
393 KB, and the get-up about 768 KB. The firmware runs a network's weights at 16 bits by default
(`APP_POLICY_WEIGHT_BITS`). Too many built-in policies leave too little room to run the largest one, so keep only the
ones you use. Move the others out of `main/rl/policies/` and rebuild.

## Stairs: SKU1 v2's or SKU2's

The robot currently runs SKU2's stairs policy (`sku2_stairs.mpxp`, relabelled `sku1` so it uses the robot's
calibration; `results/mpxp/sku2_stairs_on_sku1.mpxp`). Deploying `sku1v2_stairs.mpxp` adds SKU1 v2's own stairs walk
alongside it. Compare the two on the robot, then keep one. In simulation, `out/REPORT.md` lists both on SKU1 v2.
