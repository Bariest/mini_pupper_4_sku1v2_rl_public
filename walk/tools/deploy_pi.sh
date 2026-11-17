#!/bin/sh
# Print (default) or run (--run) the copy of the runner, the walk command, the chain-delay tool, the measured IMU
# mounting and one or more policies to the Pi's ~/walk, and mp3-link's pi/mp3link package to ~/mp3link_calib (the
# runner imports mp3link from there; MP3_LINK_DIR, default ../mp3-link). Nothing is installed on the Pi.
#   tools/deploy_pi.sh policies/<name> [policies/<name> ...] [--run]
set -e
[ $# -ge 1 ] || { echo "usage: tools/deploy_pi.sh policies/<name> [...] [--run]"; exit 2; }
HOST=${PI_HOST:?set PI_HOST to the Pi (ssh host)}
RUN=""
POLS=""
for A in "$@"; do
  if [ "$A" = "--run" ]; then RUN=1; else POLS="$POLS ./${A%/}/policy.npz"; fi
done
CMD="rsync -a --relative --exclude __pycache__ \
  ./mp3_walk_policy/__init__.py ./mp3_walk_policy/contract.py ./mp3_walk_policy/export.py ./mp3_walk_policy/heading.py ./mp3_walk_policy/runner.py \
  ./mp3_walk_policy/demo.py ./mp3_walk_policy/firmware_gait.py ./mp3_walk_policy/paths.py ./mp3_walk_policy/gait_host.c ./tools/walk ./tools/chain_delay.py \
  ./mp3_walk_policy/data/sku2_contract.json ./mp3_walk_policy/data/imu_mount.json$POLS ./tools/tilt_mount.py \
  ./data/streams $HOST:walk/"
LINK=${MP3_LINK_DIR:-../mp3-link}/pi/mp3link
CMD="$CMD && ssh -o ConnectTimeout=10 $HOST mkdir -p mp3link_calib && rsync -a --exclude __pycache__ $LINK $HOST:mp3link_calib/"
# the firmware trot (runner gait): minipupperesp's gait and IK sources, unchanged, compiled on the Pi by firmware_gait
FW=${MINIPUPPERESP_DIR:-../minipupperesp}/main
CMD="$CMD && ssh -o ConnectTimeout=10 $HOST mkdir -p minipupperesp/main && rsync -a $FW/stanford_gait.c $FW/stanford_gait.h $FW/stanford_kinematics.c $FW/stanford_kinematics.h $HOST:minipupperesp/main/"
echo "$CMD"
echo "then on the Pi: cd ~/walk && PYTHONPATH=~/walk:~/mp3link_calib python3 -m mp3_walk_policy.runner check --policy <policy>"
echo "and once per change of the gait sources (gcc): cd ~/walk && python3 -c 'from mp3_walk_policy import firmware_gait; firmware_gait.library()'"
if [ -n "$RUN" ]; then sh -c "$CMD"; fi
