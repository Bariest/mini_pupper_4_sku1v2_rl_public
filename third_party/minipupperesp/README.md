# minipupperesp (subset)

The two parts of MangDang's minipupperesp ESP32 firmware that training needs:

- `sim/model/`: SKU1's MuJoCo model (`pupper.xml`, built from `sku_1_urdf_fixed`; sha256 `411de14dfe18` as recorded in
  the mass files), its parts, the servo map and the stock servo profiles. `MP3_ROBOT=sku1v2` builds SKU1 v2 from it.
- `main/stanford_gait.c`, `stanford_kinematics.c` (and headers): the Stanford Pupper trot and inverse kinematics as the
  firmware runs them. The walk's teacher compiles them on the host (`walk/mp3_walk_policy/firmware_gait.py`); on
  Windows the prebuilt `walk/mp3_walk_policy/data/libgait_win64_<hash>.dll` is used when its hash matches these sources.

`MINIPUPPERESP_DIR` points `walk/` here (`scripts/common.ps1`). Edit these files only together with the dll (or with
a compiler installed), since the teacher checks the sources' hash.
