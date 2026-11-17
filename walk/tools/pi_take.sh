#!/bin/sh
# Run one logged walk (a take) on SKU2's Pi, detached from the ssh connection (a Wi-Fi drop cannot end it), wait for
# it to finish, then print a short report.
#   tools/pi_take.sh NAME runner  <runner args>     python3 -m mp3_walk_policy.runner <args> --log DIR/NAME
#   tools/pi_take.sh NAME mp3link <mp3link args>    python3 -m mp3link <args> --log DIR/NAME.npz
#   tools/pi_take.sh --soft-stop NAME               a graceful stop of a runner take (it ramps to the stand and saves)
#   tools/pi_take.sh --stop NAME                    SIGINT (the program disarms), SIGTERM 10 s later if still alive
#   tools/pi_take.sh --cmd NAME VX VY WZ            a new command for a running `runner gait` take (m/s, rad/s)
# On the Pi everything runs from ~/walk with PYTHONPATH=~/walk:~/mp3link_calib; logs, output (NAME.out), PID and boot
# id, the start time (NAME.start) and vcgencmd get_throttled before and after (NAME.thr0, NAME.thr1; its bits are
# sticky since boot, so a change between the two is this take's; only the first under-voltage of a boot shows), the
# kernel's under-voltage count since boot (NAME.uv0, NAME.uv1) and the boot id at the end (NAME.boot1) go to ~/walk/DIR (TAKE_DIR, default logs/<YYYYMMDD>;
# --stop and --soft-stop need the same). It waits up to TAKE_TIMEOUT s
# (default 150) for the take to end. It refuses an existing NAME and a held /dev/ttyAMA2.
# The PID is the program's own (written by the shell that execs it), never a launcher's or a pattern match.
set -u
HOST=${PI_HOST:?set PI_HOST to the Pi (ssh host)}
DIR=${TAKE_DIR:-logs/$(date +%Y%m%d)}
WAIT_S=${TAKE_TIMEOUT:-150}
SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=2 -o ServerAliveCountMax=3 -o ControlMaster=auto
     -o ControlPath=${XDG_RUNTIME_DIR:-/tmp}/pi_take_%C -o ControlPersist=10m $HOST"

usage() { echo "usage: tools/pi_take.sh NAME runner|mp3link ARGS... | tools/pi_take.sh --soft-stop|--stop NAME | tools/pi_take.sh --cmd NAME VX VY WZ"; exit 2; }
q() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }          # quote one argument for the remote shell
check_name() { case "$1" in ""|*[!A-Za-z0-9_-]*) echo "NAME: letters, digits, _ and - only"; exit 2;; esac; }

if [ "${1:-}" = "--soft-stop" ]; then       # the runner polls NAME.stop and ends as with Enter
  [ $# -eq 2 ] || usage
  check_name "$2"
  $SSH "cd ~/walk/$DIR 2>/dev/null && [ -s $2.pid ] || { echo 'NO PID FILE ~/walk/$DIR/$2.pid: wrong TAKE_DIR or name?'; exit 7; }
        P=\$(cat $2.pid); grep -q python3 /proc/\$P/cmdline 2>/dev/null || { echo 'not running'; exit 0; }
        touch $2.stop; echo \"soft stop requested for \$P (ramp to the stand, hold, save)\""
  exit $?
fi

if [ "${1:-}" = "--cmd" ]; then           # the gait reads NAME.cmd every 0.1 s (written whole, then renamed)
  [ $# -eq 5 ] || usage
  check_name "$2"
  for V in "$3" "$4" "$5"; do case "$V" in ""|*[!0-9.+-]*) echo "VX VY WZ: numbers only"; exit 2;; esac; done
  $SSH "cd ~/walk/$DIR 2>/dev/null && [ -s $2.pid ] || { echo 'NO PID FILE ~/walk/$DIR/$2.pid: wrong TAKE_DIR or name?'; exit 7; }
        P=\$(cat $2.pid); grep -q python3 /proc/\$P/cmdline 2>/dev/null || { echo 'not running'; exit 0; }
        echo '$3 $4 $5' > $2.cmd.tmp && mv $2.cmd.tmp $2.cmd && echo \"command for \$P: vx $3 vy $4 wz $5\""
  exit $?
fi

if [ "${1:-}" = "--stop" ]; then
  [ $# -eq 2 ] || usage
  check_name "$2"
  $SSH "cd ~/walk/$DIR 2>/dev/null && [ -s $2.pid ] || { echo 'NO PID FILE ~/walk/$DIR/$2.pid: wrong TAKE_DIR or name?'; exit 7; }
        P=\$(cat $2.pid); grep -q python3 /proc/\$P/cmdline 2>/dev/null || { echo 'not running'; exit 0; }
        kill -INT \$P
        for i in 1 2 3 4 5 6 7 8 9 10; do sleep 1; kill -0 \$P 2>/dev/null || { echo \"stopped \$P (SIGINT; disarmed, log saved)\"; exit 0; }; done
        kill -TERM \$P; echo \"still alive 10 s after SIGINT: SIGTERM sent to \$P (the ESP32 idles within 0.1 s)\""
  exit $?
fi

[ $# -ge 3 ] || usage
NAME=$1; KIND=$2; shift 2
check_name "$NAME"
case "$KIND" in
  runner) MOD=mp3_walk_policy.runner; LOG="$DIR/$NAME";;
  mp3link) MOD=mp3link; LOG="$DIR/$NAME.npz";;
  *) usage;;
esac
ARGS=""
for A in "$@"; do ARGS="$ARGS $(q "$A")"; done

OUT=$($SSH "cd ~/walk && mkdir -p $DIR || exit 1
  [ -e $DIR/$NAME.npz ] && { echo 'REFUSED: $DIR/$NAME.npz exists'; exit 3; }
  H=\$(fuser /dev/ttyAMA2 2>/dev/null) && { echo \"REFUSED: /dev/ttyAMA2 held by\$H\"; exit 4; }
  cat /proc/sys/kernel/random/boot_id > $DIR/$NAME.boot
  date -Iseconds > $DIR/$NAME.start
  vcgencmd get_throttled > $DIR/$NAME.thr0 2>&1
  journalctl -k -b --no-pager 2>/dev/null | grep -ciE 'under-?voltage' > $DIR/$NAME.uv0
  rm -f $DIR/$NAME.pid
  PYTHONPATH=\$HOME/walk:\$HOME/mp3link_calib setsid -f sh -c 'echo \$\$ > \"\$0.pid\"; shift; exec python3 -B -m \"\$@\"' \
    $DIR/$NAME _ $MOD $ARGS --log $LOG < /dev/null > $DIR/$NAME.out 2>&1
  for i in 1 2 3 4 5 6 7 8 9 10; do [ -s $DIR/$NAME.pid ] && break; sleep 0.2; done
  P=\$(cat $DIR/$NAME.pid 2>/dev/null)
  if [ -n \"\$P\" ] && grep -q python3 /proc/\$P/cmdline 2>/dev/null; then echo \"PID \$P\"; else
    echo 'FAILED to start:'; tail -5 $DIR/$NAME.out; exit 5; fi")
RC=$?
echo "$OUT"
[ $RC -eq 255 ] && { echo "ssh failed during the launch: the take may be running; check the Pi before relaunching"; exit 255; }
[ $RC -eq 0 ] || exit $RC
PID=${OUT##*PID }
echo "$(date +%H:%M:%S) launched $NAME ($KIND) as PID $PID; waiting (tools/pi_take.sh --stop $NAME to stop)"

T0=$(date +%s)
while :; do
  S=$($SSH "kill -0 $PID 2>/dev/null && grep -q python3 /proc/$PID/cmdline 2>/dev/null && echo alive || echo done" 2>/dev/null)
  case "$S" in
    alive) ;;
    done) break;;
    *) echo "$(date +%H:%M:%S) Pi unreachable; retrying (the take continues on the Pi)";;
  esac
  if [ $(( $(date +%s) - T0 )) -gt "$WAIT_S" ]; then echo "still running after ${WAIT_S} s: not stopped; check it"; exit 6; fi
  sleep 2
done

$SSH "cd ~/walk/$DIR
  echo '--- output'; tail -4 $NAME.out
  echo '--- report'
  cat /proc/sys/kernel/random/boot_id > $NAME.boot1
  [ \"\$(cat /proc/sys/kernel/random/boot_id)\" = \"\$(cat $NAME.boot)\" ] && echo 'Pi: same boot' || echo 'Pi: REBOOTED during the take (discard it)'
  journalctl -k -b --no-pager 2>/dev/null | grep -ciE 'under-?voltage' > $NAME.uv1
  [ \"\$(cat $NAME.uv0 2>/dev/null)\" = \"\$(cat $NAME.uv1)\" ] && echo \"under-voltage count \$(cat $NAME.uv1) (unchanged)\" || echo \"UNDER-VOLTAGE during the take: count \$(cat $NAME.uv0 2>/dev/null) -> \$(cat $NAME.uv1)\"
  vcgencmd get_throttled | tee $NAME.thr1
  [ \"\$(cat $NAME.thr0)\" = \"\$(cat $NAME.thr1)\" ] || echo \"THROTTLE CHANGED during the take: \$(cat $NAME.thr0) -> \$(cat $NAME.thr1)\"
  python3 -c 'import sys, numpy as np; d = np.load(sys.argv[1]); t = d[\"temp_dc\"] / 10.0
print(\"log %s: %d STATE frames, housing max %.1f C at start, %.1f C peak, %.1f C at the end\" % (sys.argv[1], len(t), t[0].max(), t.max(), t[-1].max()))' $NAME.npz 2>&1 \
    || echo 'NO LOG $NAME.npz (or unreadable)'"
