"""Left-right mirror symmetry for PPO data augmentation (rsl_rl's ``symmetry_cfg``): every sample is trained on twice,
as observed and mirrored about the body's sagittal plane (y -> -y), so the policy cannot keep a one-sided habit such
as a crab to one side.

The mirror, derived from the SKU2 model's kinematics (a joint change on one leg moves its foot as the mirror image of
the opposite change on the other side's leg) and the policy's coordinates (model angles relative to the stand; the
knee is its tendon, ``coef_thigh * q_thigh + coef_calf * q_calf``, whose coefficients flip sign between sides):

- legs FR <-> FL, RR <-> RL; per leg, abduction -1, hip -1, knee +1;
- body angular velocity (x, y, z) -> (-x, y, -z) (a pseudovector); gravity and linear velocity (x, -y, z);
- the command (vx, vy, wz) -> (vx, -vy, -wz);
- foot-indexed terms (contacts, foot heights) swap the same legs; the critic's body scan mirrors its grid in y;
- the critic's track errors (heading, cross-track x, y) -> (-heading, x, -y); the heading loop's error -> -error.

The map is built once per env from the observation manager's term order and sizes (each term's history frames are
mirrored alike) and cached on the env."""
from __future__ import annotations

import torch

LEG_SWAP = (1, 0, 3, 2)                          # FR, FL, RR, RL -> FL, FR, RL, RR
JOINT_SIGN = (-1.0, -1.0, 1.0)                   # abduction, hip, knee (policy coordinates)
VEC_SIGN = {"ang_vel": (-1.0, 1.0, -1.0), "gravity": (1.0, -1.0, 1.0), "command": (1.0, -1.0, -1.0),
            "lin_vel": (1.0, -1.0, 1.0), "track_errors": (-1.0, 1.0, -1.0)}
JOINT_TERMS = ("pos", "vel", "prev_action")
FOOT_TERMS = ("contact", "foot_heights")
SCALAR_TERMS = ("height",)
ODD_SCALAR_TERMS = ("heading_error",)            # a yaw angle: changes sign in the mirror


def _joint_map():
    perm, sign = [], []
    for leg in range(4):
        for j in range(3):
            perm.append(3 * LEG_SWAP[leg] + j)
            sign.append(JOINT_SIGN[j])
    return perm, sign


def _scan_perm(env, sensor_name: str = "body_scan"):
    """The body scan's ray order mirrored in y: ray i -> the ray at (x_i, -y_i)."""
    cfg = env.scene[sensor_name].cfg
    off, _ = cfg.pattern.generate_rays(None, "cpu")
    xy = off[:, :2]
    perm = []
    for x, y in xy.tolist():
        d = (xy[:, 0] - x).abs() + (xy[:, 1] + y).abs()
        k = int(d.argmin())
        assert float(d[k]) < 1e-5, "the body scan's grid is not symmetric in y"
        perm.append(k)
    return perm


def _term_map(env, name: str, size: int):
    """(perm, sign) within one term of ``size`` floats (all its history frames)."""
    if name in VEC_SIGN:
        base_perm, base_sign = [0, 1, 2], list(VEC_SIGN[name])
    elif name in JOINT_TERMS:
        base_perm, base_sign = _joint_map()
    elif name in FOOT_TERMS:
        base_perm, base_sign = list(LEG_SWAP), [1.0] * 4
    elif name in SCALAR_TERMS:
        base_perm, base_sign = [0], [1.0]
    elif name in ODD_SCALAR_TERMS:
        base_perm, base_sign = [0], [-1.0]
    elif name == "body_scan":
        base_perm = _scan_perm(env)
        base_sign = [1.0] * len(base_perm)
    else:
        raise KeyError(f"no mirror for the observation term {name!r}")
    n = len(base_perm)
    assert size % n == 0, (name, size, n)
    perm, sign = [], []
    for f in range(size // n):                    # history frames are stored one after another, each of n floats
        perm += [f * n + p for p in base_perm]
        sign += base_sign
    return perm, sign


def group_map(env, group: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (perm, sign) of a whole observation group: mirrored = obs[:, perm] * sign."""
    om = env.observation_manager
    perm, sign, off = [], [], 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        size = 1
        for d in dims:
            size *= int(d)
        p, s = _term_map(env, name, size)
        perm += [off + i for i in p]
        sign += s
        off += size
    return (torch.tensor(perm, dtype=torch.long, device=env.device),
            torch.tensor(sign, dtype=torch.float32, device=env.device))


def maps(env) -> dict:
    env = getattr(env, "unwrapped", env)
    if not hasattr(env, "_mp3_mirror"):
        jp, js = _joint_map()
        m = {g: group_map(env, g) for g in ("actor", "critic")}
        m["actions"] = (torch.tensor(jp, dtype=torch.long, device=env.device),
                        torch.tensor(js, dtype=torch.float32, device=env.device))
        env._mp3_mirror = m
    return env._mp3_mirror


def mirror(x: torch.Tensor, pm: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return x[..., pm[0]] * pm[1]


def augment(env=None, obs=None, actions=None):
    """rsl_rl's ``data_augmentation_func``: the batch followed by its mirror image (twice the samples)."""
    m = maps(env)
    out_obs = None
    if obs is not None:
        out_obs = obs.clone()
        mirrored = obs.clone()
        for g in list(obs.keys()):
            if g in m:
                mirrored[g] = mirror(obs[g], m[g])
        out_obs = torch.cat([obs, mirrored], dim=0)
    out_act = None if actions is None else torch.cat([actions, mirror(actions, m["actions"])], dim=0)
    return out_obs, out_act
