"""Keep Holosoma's MDP and PPO distribution around the original shared-router network."""
from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from holosoma.agents.modules.ppo_modules import PPOActor
from holosoma.config_types.algo import BilateralNetworkConfig, ModuleConfig

from .original.bilateral_magnet import BilateralMAGNet
from .original.joint_layout import action_joint_names, observation_joint_names
from .original.loco_action_mapping import resolve_shared_joint_action, validate_action_pair
from .original.loco_io import (
    _mirror_act, _mirror_ang_vel, _mirror_joint_values, _mirror_lin_vel,
    _mirror_phase, _mirror_vel_cmd, joint_mirror_sign_cfg, swap_lr,
)


class ObservationView(nn.Module):
    """Bake domi's authored joint ordering and original mirror functions into gathers.

    Domi uses one [sin, cos] phase. Read Holosoma's first-leg oscillator in
    that representation; apply domi's -phase on the right, never a channel swap.
    The legacy domi history is [own raw action, other raw action]. Here the
    available history is Holosoma's 29 physical actions: the shared waist is
    duplicated into the two side views. Independent pre-merge waist histories
    cannot be recovered; this adapter deliberately adds no hidden action state.
    """

    def __init__(self, env, keys, action_names, *, context=False):
        super().__init__()
        layout = env.observation_manager.get_observation_layout(keys)
        terms, offset = {}, 0
        for term in layout:
            if term["history_length"] != 1:
                raise ValueError("Bilateral v1 requires single-frame observations.")
            if term["name"] in terms:
                raise ValueError("Bilateral v1 requires unique observation term names.")
            terms[term["name"]] = list(range(offset, offset + term["dim"]))
            offset += term["dim"]
        names = list(env.robot_config.dof_names)
        left, right, left_sign, right_sign = [], [], [], []
        for name, indices in terms.items():
            if name == "cos_phase":
                continue  # Paired with sin_phase into domi's single phase_cmd below.
            if context and name not in {"command_lin_vel", "command_ang_vel", "projected_gravity", "dof_pos", "sin_phase"}:
                continue
            li, ri = indices, indices
            ls = rs = torch.ones(len(indices))
            if name in {"dof_pos", "dof_vel"}:
                if len(indices) != len(names):
                    raise ValueError(f"{name} must have one value per named joint.")
                ordered = [observation_joint_names(side) for side in ("left", "right")]
                li, ri = ([indices[names.index(j)] for j in js] for js in ordered)
                ls = torch.ones(len(li))
                rs = _mirror_joint_values(torch.ones(len(ri)), ordered[1])
            elif name == "actions":
                raw_indices = [[indices[names.index(j)] for j in js] for js in action_names]
                li = raw_indices[0] + raw_indices[1]
                ls = torch.cat((torch.ones(len(action_names[0])),
                                _mirror_joint_values(torch.ones(len(action_names[1])), action_names[1])))
                ri = _mirror_act(torch.tensor(li)).tolist()
                rs = _mirror_act(ls)
            elif name == "base_ang_vel":
                rs = _mirror_ang_vel(rs)
            elif name in {"base_lin_vel", "projected_gravity"}:
                rs = _mirror_lin_vel(rs)
            elif name in {"command_lin_vel", "command_ang_vel"}:
                command_sign = _mirror_vel_cmd(torch.ones(3))
                rs = command_sign[:2] if name == "command_lin_vel" else command_sign[2:]
                if len(indices) != len(rs):
                    raise ValueError("Expected planar velocity command [vx, vy] and one yaw command.")
            elif name == "sin_phase":
                li = ri = [indices[0], terms["cos_phase"][0]]
                ls = torch.ones(2)
                rs = _mirror_phase(ls)
            else:
                raise ValueError(f"No domi IO mapping for observation {name!r}.")
            left.extend(li)
            right.extend(ri)
            left_sign.extend(ls.tolist())
            right_sign.extend(rs.tolist())
        if not left or len(left) != len(right):
            raise ValueError("Invalid bilateral observation layout.")
        self.register_buffer("left_indices", torch.tensor(left, dtype=torch.long))
        self.register_buffer("right_indices", torch.tensor(right, dtype=torch.long))
        self.register_buffer("left_signs", torch.tensor(left_sign))
        self.register_buffer("right_signs", torch.tensor(right_sign))
        self.input_dim, self.side_dim = offset, len(left)

    def forward(self, obs):
        return obs[..., self.left_indices] * self.left_signs, obs[..., self.right_indices] * self.right_signs


class BilateralPolicy(nn.Module):
    """One original model owns both banks and their single context router.

    Central joints are read from both sides and averaged in physical coordinates.
    The two critic readouts are averaged into Holosoma's scalar value target.
    """

    def __init__(self, env, actor_keys, critic_keys, cfg: BilateralNetworkConfig):
        super().__init__()
        robot = env.robot_config
        names = list(robot.dof_names)
        if robot.actions_dim != len(names):
            raise ValueError("Bilateral v1 requires one action per named joint.")
        # Original pair resolver, with the user's contralateral_arms=false setting.
        actions = SimpleNamespace()
        for side in ("left", "right"):
            action = SimpleNamespace(
                joint_names=action_joint_names(side), policy_side=side,
                contralateral_arms=False, scale=1.0, offset=0.0, alpha=1.0,
                clip=None, defer_clip_joint_names=(),
            )
            resolve_shared_joint_action(action, names, joint_mirror_sign_cfg(skip_waist=True, skip_leg=True), side=side)
            setattr(actions, f"{side}_joint_pos", action)
        validate_action_pair(actions)
        self.action_names = [actions.left_joint_pos.joint_names, actions.right_joint_pos.joint_names]
        canonical = self.action_names[0]
        mapped = [swap_lr(name) if name.startswith("right_") else name for name in names]
        if set().union(*map(set, self.action_names)) != set(names) or any(n not in canonical for n in mapped):
            raise ValueError("The original domi action pair must cover all Holosoma joints.")
        self.actor_view = ObservationView(env, actor_keys, self.action_names)
        self.critic_view = ObservationView(env, critic_keys, self.action_names)
        self.context_view = ObservationView(env, actor_keys, self.action_names, context=True)
        self.network = BilateralMAGNet(
            [self.actor_view.side_dim, *cfg.hidden_dims, len(canonical)],
            [self.critic_view.side_dim, *cfg.hidden_dims, 1],
            [2 * self.context_view.side_dim, *cfg.router_dims],
            num_module=cfg.num_module, gate_sizes=cfg.gate_sizes,
            share_ab_modules=cfg.share_ab_modules, use_sym_param=cfg.use_sym_param,
            compose_mode="feature", feature_bias_scale=cfg.feature_bias_scale,
            checkpoint_layers=cfg.checkpoint_layers, use_fast=cfg.use_fast,
            weight_recenter=False,
        )
        self.register_buffer("readout", torch.tensor([canonical.index(name) for name in mapped]))
        signs = _mirror_joint_values(torch.ones(len(names)), names).tolist()
        self.register_buffer("left_weight", torch.tensor([
            0.0 if name.startswith("right_") else (1.0 if name.startswith("left_") else 0.5)
            for name in names
        ]))
        self.register_buffer("right_weight", torch.tensor([
            sign * (0.0 if name.startswith("left_") else (1.0 if name.startswith("right_") else 0.5))
            for name, sign in zip(names, signs)
        ]))

    def forward(self, actor_obs, critic_obs=None):
        left, right = self.actor_view(actor_obs)
        context = torch.cat(self.context_view(actor_obs), dim=-1)
        # Actor inference has no privileged observations. Unused critic computations
        # receive zeros; neither the router nor action output depends on this branch.
        if critic_obs is None:
            critic_obs = actor_obs.new_zeros(*actor_obs.shape[:-1], self.critic_view.input_dim)
        left_value, right_value = self.critic_view(critic_obs)
        al, vl, ar, vr = self.network(context, [left, left_value, right, right_value])
        mean = al[..., self.readout] * self.left_weight + ar[..., self.readout] * self.right_weight
        return mean, 0.5 * (vl + vr)


class ActorMean(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy(obs)[0]


class BilateralActor(PPOActor):
    """Inherit sampling, log_prob, entropy and inference from ordinary PPOActor."""

    def __init__(self, policy, module_config: ModuleConfig, num_actions, init_noise_std):
        nn.Module.__init__(self)
        self.actor_module = ActorMean(policy)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config.min_noise_std
        self.min_mean_noise_std = module_config.min_mean_noise_std
        self.distribution = None


class BilateralCritic(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def evaluate(self, state):
        return self.policy(state["actor_obs"], state["critic_obs"])[1]

    def reset(self, dones=None):
        pass
