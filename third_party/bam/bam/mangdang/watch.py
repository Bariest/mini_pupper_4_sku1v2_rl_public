# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Live position monitor for hand work on the rig (centring, scribing, mounting).

Only pings: nothing is written to the AT32 and no setpoint is sent. Opening the
serial port resets the ESP32, and the bench bridge then keeps every channel
idle, so running this also stops whatever the module was doing. Stop it with
Ctrl-C before any other tool opens the port.

    python3 -m bam.mangdang.watch                  # bus ids 7, 8, 9 at 10 Hz
    python3 -m bam.mangdang.watch --ids 8 --zero 8:1586

Each line shows, per servo, the raw position (dd, 0..3100), the angle from
the pot centre (1550 dd) or from ``--zero``, the reported current and the
housing temperature decoded from the feedback word (not valid on every
channel: the first channel of a board carries other data after a config read).
"""

import argparse
import time

from bam.mangdang.spi import DEFAULT_SCALE_DD, SpiMd01IO, pick_port

parser = argparse.ArgumentParser(description="Live, read-only position monitor")
parser.add_argument("--ids", type=str, default="7,8,9", help="comma-separated bus ids")
parser.add_argument("--port", type=str, default=None)
parser.add_argument("--hz", type=float, default=10.0)
parser.add_argument(
    "--zero",
    action="append",
    default=[],
    help="ID:DD reference for the angle column (repeatable); default the pot centre",
)
parser.add_argument("--scroll", action="store_true", help="one line per refresh instead of in place")
args = parser.parse_args()


def decode_temp(res: int) -> float | None:
    """Tenths of a degree in the feedback word, as the recorder decodes it."""
    res &= 0xFFFFFFFF
    return res / 10.0 if 0 < res < 3000 else None


def main() -> None:
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    zeros = {i: DEFAULT_SCALE_DD / 2.0 for i in ids}
    for item in args.zero:
        sid, dd = item.split(":")
        zeros[int(sid)] = float(dd)
    io = SpiMd01IO(pick_port(args.port))
    period = 1.0 / args.hz
    print("read-only: pings only; every channel is idle while this runs. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.monotonic()
            cells = []
            for sid in ids:
                try:
                    r = io.ping(sid)
                except (OSError, TimeoutError) as exc:
                    cells.append(f"{sid}: no reply ({exc.__class__.__name__})")
                    continue
                temp = decode_temp(io.last_res)
                cells.append(
                    f"{sid}: {r['pos_dd']:4d} dd {(r['pos_dd'] - zeros[sid]) / 10.0:+7.1f} deg "
                    f"{r['cur_mA']:+5d} mA" + (f" {temp:4.1f} C" if temp is not None else "       ")
                )
            line = "  |  ".join(cells)
            print(line if args.scroll else "\r" + line, end="\n" if args.scroll else "", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print()
    finally:
        io.close()


if __name__ == "__main__":
    main()
