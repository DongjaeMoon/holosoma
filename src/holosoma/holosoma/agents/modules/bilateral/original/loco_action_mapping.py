"""Resolve locomotion action ownership before constructing Isaac managers.

The resulting config is the physical joint map. There is no private runtime
remapping, and reapplying the resolver (including after Hydra) is idempotent.
"""
from __future__ import annotations

from collections.abc import Mapping
import re


JOINT_CONTROL_REWARDS = (
    "help_qdev", "safe_energy", "safe_torque", "safe_speed",
    "safe_torque_lim", "safe_qlim", "safe_vlim",
)


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _set(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _names(selectors, names, *, unique=True):
    """Expand selectors in authored order; action selectors must not overlap."""
    if selectors is None:
        return list(names)
    if isinstance(selectors, str):
        selectors = [selectors]
    result = []
    for selector in selectors:
        matches = [name for name in names if re.fullmatch(selector, name)]
        if not matches:
            raise ValueError(f"Joint/body selector {selector!r} matches no configured G1 names.")
        for name in matches:
            if name in result:
                if unique:
                    raise ValueError(f"Overlapping action selectors for {name!r}.")
            else:
                result.append(name)
    return result


def _values(value, names, default):
    if not isinstance(value, Mapping):
        return [value] * len(names)
    result = dict.fromkeys(names, default)
    assigned = set()
    for selector, item in value.items():
        for name in _names(selector, names):
            if name in assigned:
                raise ValueError(f"Overlapping action parameter selectors for {name!r}.")
            result[name] = item
            assigned.add(name)
    return [result[name] for name in names]


def _arm_sign(name, arm_signs):
    if not name.startswith(("left_", "right_")):
        return None
    return next((sign for pattern, sign in arm_signs.items()
                 if re.fullmatch(pattern, name)), None)


def validate_action_pair(actions):
    left, right = (_get(actions, f"{side}_joint_pos") for side in ("left", "right"))
    if left is not None and right is not None:
        if _get(left, "contralateral_arms", False) != _get(right, "contralateral_arms", False):
            raise ValueError(
                "Set contralateral_arms to the same value on both "
                "env.actions.left_joint_pos and env.actions.right_joint_pos.")


def resolve_shared_joint_action(cfg, joint_names, arm_signs, *, side=None):
    """Declare destination names and signed affine maps, retaining slot order.

    An arm changes sides only when its *declared physical side* differs from
    the requested side. This avoids hidden state and also supports toggling a
    previously resolved config back to ipsilateral control.
    """
    names = _names(cfg.joint_names, joint_names)
    # Archived/Hydra dictionaries may use integer literals for floating-point
    # parameters (notably alpha=1). Keep Isaac's tensor assignments in float.
    for field in ("scale", "offset", "alpha"):
        value = getattr(cfg, field)
        setattr(cfg, field, {key: float(item) for key, item in value.items()}
                if isinstance(value, Mapping) else float(value))
    declared_side = _get(cfg, "policy_side", "")
    if side and declared_side and side != declared_side:
        raise ValueError(f"Action policy_side={declared_side!r} disagrees with term side={side!r}.")
    side = side or declared_side
    if not side:
        leg_sides = {name.split("_", 1)[0] for name in names
                     if re.fullmatch(r"(left|right)_(hip_.*|knee|ankle_.*)_joint", name)}
        if len(leg_sides) == 1:
            side = leg_sides.pop()
    if not any(_arm_sign(name, arm_signs) is not None for name in names):
        cfg.joint_names = names
        return
    if side not in ("left", "right"):
        raise ValueError("An arm action without a single leg side needs policy_side='left' or 'right'.")
    cfg.policy_side = side
    arm_side = ({"left": "right", "right": "left"}[side]
                if cfg.contralateral_arms else side)
    routed, signs = [], []
    for name in names:
        sign = _arm_sign(name, arm_signs)
        dest = arm_side + "_" + name.split("_", 1)[1] if sign is not None else name
        routed.append(dest)
        signs.append(sign if name != dest else 1.0)
    if len(set(routed)) != len(routed):
        raise ValueError("An action term cannot control both arms in the same slots.")
    if routed != names:
        # Resolve in source coordinates before renaming any selector. Alpha and
        # raw-action bounds follow slots; signed scales and explicit offsets
        # reflect into destination joint coordinates. Asset default offsets
        # are still read normally by JointPositionAction from destination IDs.
        for field, default, reflect in (("scale", 1.0, True), ("offset", 0.0, True),
                                        ("alpha", 1.0, False),
                                        ("clip", (-float("inf"), float("inf")), False)):
            value = getattr(cfg, field)
            if value is None:
                continue
            values = _values(value, names, default)
            setattr(cfg, field, {name: item * sign if reflect else item
                                for name, item, sign in zip(routed, values, signs)})
        deferred = set(_names(cfg.defer_clip_joint_names, names))
        cfg.defer_clip_joint_names = tuple(dest for name, dest in zip(names, routed) if name in deferred)
    cfg.joint_names = routed
    cfg.preserve_order = True


def reward_joint_scope(env_cfg, suffix):
    """Unified scope takes precedence; old qdev-only overrides keep their meaning."""
    scope = _get(env_cfg, "side_rewd_scope")
    field = "side_rewd_scope"
    if scope is None:
        scope = _get(env_cfg, "qdev_joint_scope", "configured") if suffix == "help_qdev" else "configured"
        field = "qdev_joint_scope"
    if scope not in ("configured", "controlled"):
        raise ValueError(f"env.{field} must be 'configured' or 'controlled'.")
    return scope


def is_joint_control_reward(suffix, asset):
    """Include additional sided joint costs with explicit joint selectors."""
    return suffix in JOINT_CONTROL_REWARDS or _get(asset, "joint_names") is not None


def resolve_action_reward_scopes(env_cfg, joint_names, arm_signs):
    """Allocate sided physical-joint costs to the action that controls them.

    Configured subsets retain their joint kinds (including legs and waist).
    Arm selectors, whether authored ipsilaterally or already resolved, follow
    the action's arm. 'controlled' covers every action joint for joint costs.
    Limb-speed costs retain their selected body types. Global and foot rewards
    do not change. A separate common action owns the extracted waist costs.
    """
    reward_joint_scope(env_cfg, "help_qdev")  # Validate even if qdev is disabled.
    rewards = _get(env_cfg, "rewards")
    if rewards is None:
        return
    terms = list(rewards.items() if isinstance(rewards, Mapping) else vars(rewards).items())
    for side in ("left", "right", "common"):
        action = _get(_get(env_cfg, "actions"), f"{side}_joint_pos")
        if action is None or not hasattr(action, "contralateral_arms"):
            continue
        controlled = list(action.joint_names)
        # Resolved physical names are authoritative, including partial actions.
        arm_side = next((n.split("_", 1)[0] for n in controlled
                         if _arm_sign(n, arm_signs) is not None), None)

        def owned_names(selectors, *, bodies=False):
            universe = [n.replace("_joint", "_link") for n in joint_names] if bodies else joint_names
            selected = _names(selectors, universe, unique=False)
            owned = set()
            for name in selected:
                joint = name.replace("_link", "_joint") if bodies else name
                if arm_side is not None and _arm_sign(joint, arm_signs) is not None:
                    joint = arm_side + "_" + joint.split("_", 1)[1]
                if joint in controlled:
                    owned.add(joint)
            result = [n for n in controlled if n in owned]
            return [n.replace("_joint", "_link") for n in result] if bodies else result

        for name, term in terms:
            if not name.startswith(side + "_"):
                continue
            suffix = name[len(side) + 1:]
            asset = _get(_get(term, "params", {}), "asset_cfg")
            if asset is None or _get(asset, "name") != action.asset_name:
                continue
            bodies = suffix == "safe_limb_speed"
            if not bodies and not is_joint_control_reward(suffix, asset):
                continue
            name_field, id_field = ("body_names", "body_ids") if bodies else ("joint_names", "joint_ids")
            selectors = _get(asset, name_field)
            all_controlled = not bodies and reward_joint_scope(env_cfg, suffix) == "controlled"
            if not all_controlled and selectors is None and _get(asset, id_field, slice(None)) != slice(None):
                raise ValueError(f"{side}_{suffix} needs explicit {name_field} for action ownership resolution.")
            selected = (controlled if all_controlled
                        else owned_names(selectors, bodies=bodies))
            if not selected:
                raise ValueError(f"{side}_{suffix} selects no joints/bodies controlled by {side}_joint_pos.")
            _set(asset, name_field, list(selected))
            _set(asset, id_field, slice(None))
            _set(asset, "preserve_order", True)
