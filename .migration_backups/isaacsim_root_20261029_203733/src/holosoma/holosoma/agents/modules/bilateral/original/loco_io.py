"""Original domi loco_env_cfg IO functions, copied verbatim."""
from __future__ import annotations
from typing import Sequence, Tuple, Optional
import re
import torch as th
from . import isaac_strings as string_utils

def joint_mirror_sign_cfg(
    skip_waist: bool = False,
    skip_arm: bool = False,
        skip_leg: bool = False):
    if skip_waist:
        k_waist = []
        s_waist = []
    else:
        k_waist = ['waist_yaw_joint',
                   'waist_roll_joint',
                   'waist_pitch_joint']
        s_waist = [-1., -1., +1.]
    if skip_leg:
        k_leg = []
        s_leg = []
    else:
        k_leg = ['.*_hip_pitch_joint',
                 '.*_hip_roll_joint',
                 '.*_hip_yaw_joint',
                 '.*_knee_joint',
                 '.*_ankle_pitch_joint',
                 '.*_ankle_roll_joint']
        s_leg = [1., -1., -1., 1., 1., -1.]
    if skip_arm:
        k_arm = []
        s_arm = []
    else:
        k_arm = ['.*_shoulder_pitch_joint',
                 '.*_shoulder_roll_joint',
                 '.*_shoulder_yaw_joint',
                 '.*_elbow_joint',
                 '.*_wrist_roll_joint',
                 '.*_wrist_pitch_joint',
                 '.*_wrist_yaw_joint']
        s_arm = [1., -1., -1., 1., -1., 1., -1.]
    scale_cfg = dict(zip(
        [*k_waist, *k_leg, *k_arm],
        [*s_waist, *s_leg, *s_arm]))
    return scale_cfg


def swap_lr(k: str):
    return (k
            .replace('left', 'RIGHT')
            .replace('right', 'left')
            .replace('RIGHT', 'right')
            )


def _mirror_lin_vel(x):
    x = x.clone()
    x[..., 1].neg_()
    return x


def _mirror_ang_vel(x):
    x = x.clone()
    x[..., 0].neg_()
    x[..., 2].neg_()
    return x


def _mirror_vel_cmd(x):
    # x,y, yaw
    x = x.clone()
    x[..., 1].neg_()  # flip y
    x[..., 2].neg_()  # flip yaw
    return x


def _mirror_joint_values(x: th.Tensor, joint_names: Sequence[str]) -> th.Tensor:
    """Apply the existing signs to a resolved joint subset, in tensor order.

    Isaac's resolver requires every supplied pattern to match. An observation
    may omit the waist, legs or arms, so exclude those absent patterns without
    changing the signs or adding any joints to the observation.
    """
    if joint_names is None or len(joint_names) != x.shape[-1]:
        raise ValueError("Joint reflection needs one resolved joint name per tensor column.")
    scale_cfg = {pattern: sign for pattern, sign in joint_mirror_sign_cfg().items()
                 if any(re.fullmatch(pattern, name) for name in joint_names)}
    index_list, _, value_list = string_utils.resolve_matching_names_values(
        scale_cfg, joint_names, preserve_order=False)
    if len(index_list) != len(joint_names):
        unknown = [name for i, name in enumerate(joint_names) if i not in index_list]
        raise ValueError(f"No joint mirror sign defined for: {unknown}")
    scale = x.new_ones(x.shape[-1])
    scale[index_list] = x.new_tensor(value_list)
    return x * scale


def _mirror_phase(x: th.Tensor):
    """ swap left-right phases. """
    return -x


def _mirror_act(x: th.Tensor):
    """ swap left-right actions. """
    # return x
    # -- observation --
    # old: left:left, right:right
    # new: left:mirror(right), right:mirror(left)

    # --actions --
    # old: (left, right)
    # new: mirror(right), mirror(left)

    x = x.unflatten(-1, (2, -1))[..., [1, 0], :].flatten(-2, -1)
    return x


def last_action(
        env,
        order: Tuple[str, ...],
        noise_std: float = 0.0,
        noise_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
        waist_action: Optional[str] = None,
        mirror_waist: bool = False):
    """Raw policy history in a side's canonical coordinates.

    Split-waist layout: [waist(3), own limbs(13), other limbs(13)]. Only the
    single physical waist vector needs conversion on the right: limb raw
    actions already use canonical policy coordinates (their physical signs
    are applied by the action term). Legacy callers retain their original order.
    """
    parts = []
    if waist_action is not None:
        term = env.action_manager.get_term(waist_action)
        waist = term.raw_actions
        if mirror_waist:
            # Use the domain's existing rule and actual runtime joint order.
            signs = joint_mirror_sign_cfg(skip_arm=True, skip_leg=True)
            indices, _, values = string_utils.resolve_matching_names_values(
                signs, term._joint_names, preserve_order=False)
            if len(indices) != waist.shape[-1]:
                raise ValueError("waist_action must control waist joints only.")
            scale = th.ones(waist.shape[-1], dtype=waist.dtype, device=waist.device)
            scale[indices] = th.as_tensor(values, dtype=waist.dtype, device=waist.device)
            waist = waist * scale
        parts.append(waist)
    elif mirror_waist:
        raise ValueError("mirror_waist requires waist_action.")
    parts.extend(env.action_manager.get_term(k).raw_actions for k in order)
    out = th.cat(parts, dim=-1)
    noise_std = float(noise_std)
    if noise_std > 0.0:
        noise = noise_std * th.randn_like(out)
        if noise_clip is not None:
            noise = noise.clamp(min=noise_clip[0], max=noise_clip[1])
        out = out + noise
    return out

