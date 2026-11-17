#!/usr/bin/env python
"""Drive an omni policy around the simulator with the keyboard.

    python -B scripts/drive.py --ckpt runs/ppo/<run>                 # the run's newest checkpoint
    python -B scripts/drive.py --ckpt policies/<name>                # an exported omni policy
    python -B scripts/drive.py --ckpt runs/clone/stock_omni/clone.pt --arm stock --omni
    python -B scripts/drive.py --ckpt runs/ppo/<run> --speed 0.12 --wz 0.6

Hold W / S to walk forward / back, A / D to strafe left / right, Q / E to turn left / right. Keys combine: a
diagonal keeps ``--speed`` (W+A walks at 45 deg), and Q / E add a yaw rate on top. Nothing held commands
(0, 0, 0). R resets the robot, P pauses. The keys are read only while the viewer window has the focus (Windows),
and they never reach the viewer: its own shortcuts on them (W wireframe, S shadow, ...) are off; its other keys work.

The frame rate. The policy steps at 50 Hz but the physics at 1 kHz: every physics substep's pose is kept, and the
window is handed a new one every frame (up to ``--fps``) instead of 50 times a second. The physics runs in its own
thread, so a slow ``env.step`` never stalls the window. With one env, mjlab's per-substep Python costs more than the
GPU's physics; the env here drops what driving does not need (rewards, the critic's observations, the feet's air
time) and three GPU syncs per substep, none of which changes the dynamics. If the physics is still slower than real
time (the BAM arm), the playback slows to match it, smoothly, and the console shows the speed ("playback x").
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                                             # noqa: E402
import mujoco.viewer                                                      # noqa: E402
import torch                                                              # noqa: E402
import warp as wp                                                         # noqa: E402

from mjlab.envs import ManagerBasedRlEnv                                  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper                                    # noqa: E402
from mjlab.sim.sim_data import TorchArray                                  # noqa: E402
from mjlab.utils.buffers.circular_buffer import CircularBuffer             # noqa: E402

from mp3_walk_policy import distill                                        # noqa: E402
from mp3_walk_policy.env_cfg import OMNI_SPEED_RANGE, OMNI_WZ_MAX, make_env_cfg, noise_profile, omni_options  # noqa: E402
from mp3_walk_policy.export import NumpyPolicy                             # noqa: E402

DEVICE = "cuda:0"


# ------------------------------------------------------------------------------------------- sync-free replacements
# The same values as mjlab's, without a host sync: each ``if torch.any(...)`` / scalar write waits for the GPU, which
# serialises the physics with Python (on Windows each wait costs ~0.1-0.5 ms, several per 1 ms substep).

def _cb_append(self, data):
    if data.shape[0] != self._batch_size:
        raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
    data = data.to(self._device)
    if self._buffer is None:
        self._pointer = -1
        self._buffer = torch.empty((self._max_len, *data.shape), dtype=data.dtype, device=self._device)
    self._pointer = (self._pointer + 1) % self._max_len
    self._buffer[self._pointer] = data
    first = (self._num_pushes == 0).view(1, -1, *([1] * (data.dim() - 1)))
    self._buffer.copy_(torch.where(first, data.unsqueeze(0), self._buffer))
    self._num_pushes += 1


def _cb_getitem(self, key):
    if self._buffer is None:
        raise RuntimeError("Buffer not initialized. Call append() first.")
    if isinstance(key, int):
        key = torch.full((self._batch_size,), key, dtype=torch.long, device=self._device)
    elif key.ndim == 0:
        key = key.expand(self._batch_size)
    key = key.to(device=self._device, dtype=torch.long)
    valid = torch.minimum(key, self._num_pushes.clamp_min(1) - 1).clamp_min(0)
    return self._buffer[torch.remainder(self._pointer - valid, self._max_len), self._all_indices]


_torch_array_setitem = TorchArray.__setitem__


def _ta_setitem(self, idx, value):
    if isinstance(value, (int, float)) and self._is_cuda:
        value = torch.full((), value, dtype=self._tensor.dtype, device=self._tensor.device)
    _torch_array_setitem(self, idx, value)


def install_speedups():
    CircularBuffer.append = _cb_append
    CircularBuffer.__getitem__ = _cb_getitem
    TorchArray.__setitem__ = _ta_setitem


def drive_cfg(opt):
    """The task's env without what only training reads (the dynamics, the actor's observations and the terminations
    unchanged)."""
    cfg = make_env_cfg(opt)
    cfg.rewards = {}
    cfg.curriculum = {}
    cfg.observations = {"actor": cfg.observations["actor"]}
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "feet_contact")
    return cfg


# ------------------------------------------------------------------------------------------------------- the input
class _KbdLLHookStruct(ctypes.Structure):
    _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD), ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


_HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wt.WPARAM, wt.LPARAM)
_WH_KEYBOARD_LL, _WM_QUIT = 13, 0x0012
_KEY_DOWN, _KEY_UP = (0x0100, 0x0104), (0x0101, 0x0105)          # WM_(SYS)KEYDOWN, WM_(SYS)KEYUP


class Keyboard:
    """The held state of ``keys``, only while a window of this process is in front. A low-level keyboard hook
    (Win32 ``WH_KEYBOARD_LL``, on its own thread) takes those keys before any window sees them, so the MuJoCo viewer's
    shortcuts on them do not fire."""

    def __init__(self, keys: str):
        if os.name != "nt":
            raise SystemExit("drive.py reads the keyboard with the Win32 API (Windows only)")
        u = self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        u.CallNextHookEx.argtypes = (wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM)
        u.CallNextHookEx.restype = ctypes.c_ssize_t
        u.SetWindowsHookExW.argtypes = (ctypes.c_int, _HOOKPROC, wt.HINSTANCE, wt.DWORD)
        u.SetWindowsHookExW.restype = wt.HHOOK
        u.GetMessageW.argtypes = (ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT)
        u.GetForegroundWindow.restype = wt.HWND
        u.GetWindowThreadProcessId.argtypes = (wt.HWND, ctypes.POINTER(wt.DWORD))
        self.keys = {ord(k): k for k in keys}
        self.pid = os.getpid()
        self.down, self.prev = set(), set()
        self._proc = _HOOKPROC(self._hook)                 # kept referenced: Windows calls it
        self._ready = threading.Event()
        self._tid = None
        threading.Thread(target=self._loop, daemon=True).start()
        self._ready.wait()
        if self._tid is None:
            raise SystemExit(f"could not install the keyboard hook (error {ctypes.get_last_error()})")

    def _loop(self):
        hook = self.user32.SetWindowsHookExW(_WH_KEYBOARD_LL, self._proc, None, 0)
        if hook:
            self._tid = ctypes.windll.kernel32.GetCurrentThreadId()
        self._ready.set()
        if not hook:
            return
        msg = wt.MSG()
        while self.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:      # the hook is called from here
            pass
        self.user32.UnhookWindowsHookEx(hook)

    def _hook(self, code, wparam, lparam):
        if code == 0:
            k = self.keys.get(ctypes.cast(lparam, ctypes.POINTER(_KbdLLHookStruct)).contents.vkCode)
            if k is not None:
                if wparam in _KEY_UP:
                    self.down.discard(k)
                if self.focused():
                    if wparam in _KEY_DOWN:
                        self.down.add(k)
                    return 1                                # swallowed: no window gets it
        return self.user32.CallNextHookEx(None, code, wparam, lparam)

    def focused(self) -> bool:
        pid = wt.DWORD()
        self.user32.GetWindowThreadProcessId(self.user32.GetForegroundWindow(), ctypes.byref(pid))
        return pid.value == self.pid

    def poll(self) -> tuple[set, set]:
        """(held, pressed since the last poll)."""
        if not self.focused():
            self.down.clear()
        now = set(self.down)
        pressed, self.prev = now - self.prev, now
        return now, pressed

    def close(self):
        if self._tid is not None:
            self.user32.PostThreadMessageW(self._tid, _WM_QUIT, 0, 0)


def command(held: set, speed: float, wz: float) -> tuple:
    fwd = ("W" in held) - ("S" in held)
    left = ("A" in held) - ("D" in held)
    turn = ("Q" in held) - ("E" in held)
    n = math.hypot(fwd, left)
    k = speed / n if n else 0.0
    return (fwd * k, left * k, turn * wz)


def load_policy(path: Path, arm: str | None, omni: bool):
    """(actor: obs -> action on the GPU, the run's config, arm, omni)."""
    if (path / "policy.npz").is_file():
        pol = NumpyPolicy(path / "policy.npz")
        cfg = path / "config.json"
        run = json.loads(cfg.read_text()) if cfg.exists() else {}
        omni = omni or bool(pol.meta.get("omni"))

        def actor(obs):
            return torch.from_numpy(pol(obs.cpu().numpy())).to(DEVICE)
    else:
        if path.is_dir():
            models = sorted(path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            if not models:
                raise SystemExit(f"no policy.npz or model_*.pt in {path}")
            path = models[-1]
        cfg = path.parent / "config.json"
        run = json.loads(cfg.read_text()) if cfg.exists() else {}
        omni = omni or bool(run.get("task", {}).get("options", {}).get("omni", False))
        actor = distill.load_actor(path, DEVICE)
    print(f"policy {path}", flush=True)
    return actor, run, arm or run.get("task", {}).get("arm", "stock"), omni


# ----------------------------------------------------------------------------------------------------- the physics
class Physics(threading.Thread):
    """Steps the env ahead of the playback clock (at most ``lead`` s of sim time) and queues every substep's qpos
    as ``(sim time, qpos)``. The viewer loop owns ``play_t``, ``cmd``, ``reset`` and ``stop``."""

    def __init__(self, env, actor, lead: float):
        super().__init__(daemon=True)
        self.env, self.actor, self.lead = env, actor, lead
        u = env.unwrapped
        self.sim, self.term = u.sim, u.command_manager.get_term("twist")
        self.dec, self.pdt, self.step_dt = u.cfg.decimation, u.physics_dt, u.step_dt
        self.buf = torch.zeros(self.dec, self.sim.mj_model.nq, device=u.device)
        self.stream = torch.cuda.ExternalStream(wp.get_stream(self.sim.wp_device).cuda_stream)
        self.k = 0
        self._sim_step = self.sim.step
        self.sim.step = self._record_substep
        self.frames = deque()
        self.sim_t = self.play_t = 0.0
        self.cmd, self._cmd = (0.0, 0.0, 0.0), None
        self.reset = self.stop = False
        self.falls = 0
        self.error = None

    def _record_substep(self):
        self._sim_step()
        with torch.cuda.stream(self.stream):
            self.buf[self.k % self.dec].copy_(self.sim.data.qpos[0])
        self.k += 1

    def _restart(self):
        self.env.reset()
        self.obs = self.env.get_observations()
        with torch.cuda.stream(self.stream):
            q = self.sim.data.qpos[0].cpu().numpy()
        self.frames.append((self.sim_t, q))

    def run(self):
        try:
            with torch.no_grad():
                self._restart()
                while not self.stop:
                    if self.reset:
                        self.reset = False
                        self._restart()
                    if self.sim_t - self.play_t > self.lead:
                        time.sleep(0.0005)
                        continue
                    if self.cmd != self._cmd:
                        self._cmd = self.cmd
                        self.term.vel_command_b[:] = torch.tensor(self._cmd, device=self.term.vel_command_b.device)
                    self.k = 0
                    out = self.env.step(self.actor(self.obs["actor"]))
                    self.obs = out[0]
                    with torch.cuda.stream(self.stream):
                        q = self.buf.cpu().numpy()
                    t0 = self.sim_t
                    for i in range(self.dec):
                        self.frames.append((t0 + (i + 1) * self.pdt, q[i]))
                    self.sim_t = t0 + self.step_dt
                    if bool(out[2][0]):
                        self.falls += 1
        except Exception as e:                                       # surfaced by the viewer loop
            self.error = e


# ------------------------------------------------------------------------------------------------------ the viewer
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a policies/<name> directory, a checkpoint (.pt) or a run directory")
    ap.add_argument("--arm", default=None, help="default: the run's arm")
    ap.add_argument("--omni", action="store_true", help="an omni policy (default: from the run's config)")
    ap.add_argument("--speed", type=float, default=0.10, help="m/s, the travel speed of W/A/S/D")
    ap.add_argument("--wz", type=float, default=0.40, help="rad/s, the yaw rate of Q/E")
    ap.add_argument("--fps", type=float, default=240.0, help="the most poses handed to the viewer per second")
    ap.add_argument("--noisy", action="store_true", help="observation noise (the run's profile at the ramp's end)")
    a = ap.parse_args()

    actor, run, arm, omni = load_policy(Path(a.ckpt), a.arm, a.omni)
    if not omni:
        raise SystemExit("drive.py needs an omni policy (a forward policy cannot strafe or turn); pass --omni to force")
    lo, hi = OMNI_SPEED_RANGE
    if not lo <= a.speed <= hi or a.wz > OMNI_WZ_MAX:
        print(f"warning: training covers speeds {lo}-{hi} m/s and |wz| <= {OMNI_WZ_MAX} rad/s", flush=True)

    install_speedups()
    opt = omni_options(fixed_command=(0.0, 0.0, 0.0), arm=arm, num_envs=1, noisy=a.noisy, randomise=False,
                       noise_ramp=False, episode_s=3600.0, visuals=True,
                       noise_end=noise_profile(run.get("noise", "n2")))
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=drive_cfg(opt), device=DEVICE), clip_actions=1.0)
    u = env.unwrapped
    step_dt = u.step_dt
    target = 1.5 * step_dt                # the sim time kept queued ahead of the screen
    phys = Physics(env, actor, lead=target + step_dt)

    mjm, mjd = u.sim.mj_model, u.sim.mj_data
    kb = Keyboard("WASDQERP")
    v = mujoco.viewer.launch_passive(mjm, mjd, show_left_ui=False, show_right_ui=False)
    vc = u.cfg.viewer
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    v.cam.trackbodyid = u.scene["robot"].indexing.root_body_id
    v.cam.distance, v.cam.elevation, v.cam.azimuth = vc.distance, vc.elevation, vc.azimuth
    print("W/S forward/back  A/D strafe left/right  Q/E turn left/right  R reset  P pause  (focus the viewer)",
          flush=True)

    sys.setswitchinterval(0.0005)         # the physics thread holds the GIL most of the time; hand it back quickly
    phys.start()
    frame_dt = 1.0 / a.fps
    paused = False
    rate, rtf, occ = 1.0, 1.0, target     # playback speed; the physics' sim time per wall time; sim time queued
    rtf_win = deque()                     # (wall, sim_t), the last second
    shown = None
    last = stats_t = time.perf_counter()
    n_frames = falls = 0
    try:
        while v.is_running():
            if phys.error is not None:
                raise phys.error
            now = time.perf_counter()
            dt, last = now - last, now
            held, pressed = kb.poll()
            if "P" in pressed:
                paused = not paused
            if "R" in pressed:
                phys.reset = True
            phys.cmd = command(held, a.speed, a.wz)

            rtf_win.append((now, phys.sim_t))
            while now - rtf_win[0][0] > 1.0:
                rtf_win.popleft()
            if now - rtf_win[0][0] > 0.2:
                rtf = (rtf_win[-1][1] - rtf_win[0][1]) / (now - rtf_win[0][0])
            occ += (phys.sim_t - phys.play_t - occ) * min(1.0, dt / 0.3)
            rate = 0.0 if paused else min(1.0, max(0.0, rtf * (1.0 + 0.5 * (occ - target) / target)))
            if rtf_win[-1][1] - rtf_win[0][1] == 0.0 and not paused:
                rate = 1.0                    # nothing produced yet (start-up) or throttled at the lead
            phys.play_t = min(phys.play_t + rate * dt, phys.sim_t)

            fr = phys.frames
            while len(fr) > 1 and fr[1][0] <= phys.play_t:
                fr.popleft()
            if fr and fr[0] is not shown:
                shown = fr[0]
                with v.lock():
                    mjd.qpos[:] = shown[1]
                    mujoco.mj_forward(mjm, mjd)
                v.sync()
                n_frames += 1

            if now - stats_t >= 0.5:
                span, stats_t = now - stats_t, now
                if phys.falls != falls:
                    falls = phys.falls
                    print(f"\nfell ({falls}): reset", flush=True)
                c = phys.cmd
                print(f"\rcmd vx {c[0]:+.3f} vy {c[1]:+.3f} m/s  wz {c[2]:+.2f} rad/s   "
                      f"{n_frames / span:5.0f} poses/s   playback x{rate:.2f}"
                      f"{'   PAUSED' if paused else ''}      ", end="", flush=True)
                n_frames = 0
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        phys.stop = True
        kb.close()
        print(flush=True)
        v.close()
    os._exit(0)


if __name__ == "__main__":
    main()
