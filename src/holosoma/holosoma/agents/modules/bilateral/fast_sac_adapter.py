"""Domi's bilateral banks around Holosoma's unchanged SAC distributions."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from holosoma.agents.fast_sac.fast_sac import Actor, Critic, DistributionalQNetwork
from .adapter import BilateralPolicy
from .original.bilateral_magnet import BilateralMAGNet
from .original.loco_io import _mirror_joint_values


class BilateralBank(nn.Module):
    """Evaluate one bank with the original routing slots and original layers."""

    def __init__(self, network, *, actor, router=None):
        super().__init__()
        self.layers = network.actor if actor else network.critic
        self.router = network.modulator if router is None else router
        self.bank_count = len(network.banks)
        self.gate_count = 2 * (len(network.actor) + len(network.critic))
        self.bank_offset = 0 if actor else 3 * len(network.actor)
        self.gate_offset = 0 if actor else 2 * len(network.actor)
        self.checkpoint_layers = network.checkpoint_layers

    def forward(self, context, left, right):
        probs, gates = self.router(context)
        pi, gi = self.bank_offset, self.gate_offset
        for layer in self.layers:
            p = (probs[pi:pi + 3], probs[self.bank_count + pi:self.bank_count + pi + 3])
            g = None if gates is None else (
                gates[gi:gi + 2], gates[self.gate_count + gi:self.gate_count + gi + 2])
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                left, right = checkpoint(layer, left, right, p, g, use_reentrant=False)
            else:
                left, right = layer(left, right, p, g)
            pi, gi = pi + 3, gi + 2
        return left, right


class ActorFeatures(nn.Module):
    def __init__(self, layout, network):
        super().__init__()
        self.view, self.context = layout.actor_view, layout.context_view
        self.bank = BilateralBank(network, actor=True)
        self.register_buffer('readout', layout.readout.clone())
        self.register_buffer('left_weight', layout.left_weight.clone())
        self.register_buffer('right_weight', layout.right_weight.clone())

    def forward(self, obs):
        left, right = self.view(obs)
        left, right = self.bank(torch.cat(self.context(obs), -1), left, right)
        lm, ls = left.chunk(2, -1)
        rm, rs = right.chunk(2, -1)
        mean = lm[..., self.readout] * self.left_weight + rm[..., self.readout] * self.right_weight
        logstd = ls[..., self.readout] * self.left_weight.abs() + rs[..., self.readout] * self.right_weight.abs()
        return torch.cat((mean, logstd), -1)


class Slice(nn.Module):
    def __init__(self, start, stop):
        super().__init__()
        self.start, self.stop = start, stop

    def forward(self, value):
        return value[..., self.start:self.stop]


class BilateralSACActor(Actor):
    """Reuse native tanh, log-std bounds, sampling, scaling and log probabilities."""

    def __init__(self, layout, network, original):
        nn.Module.__init__(self)
        for name in ('obs_indices', 'obs_keys', 'n_act', 'log_std_max', 'log_std_min',
                     'use_tanh', 'n_envs', 'device', 'hidden_dim', 'use_layer_norm',
                     'encoder_obs_key', 'encoder_obs_shape'):
            setattr(self, name, getattr(original, name))
        self.net = ActorFeatures(layout, network)
        self.fc_mu, self.fc_logstd = Slice(0, self.n_act), Slice(self.n_act, 2 * self.n_act)
        self.register_buffer('action_scale', original.action_scale.clone())
        self.register_buffer('action_bias', original.action_bias.clone())
        # The stock SAC actor starts both heads at zero.
        for parameter in network.actor[-1].parameters():
            nn.init.zeros_(parameter)


class BilateralQ(DistributionalQNetwork):
    def __init__(self, layout, network, router, original):
        nn.Module.__init__(self)
        self.view = layout.critic_view
        self.context = layout.context_view
        self.bank = BilateralBank(network, actor=False, router=router)
        self.v_min, self.v_max, self.num_atoms = original.v_min, original.v_max, original.num_atoms
        names = list(layout._joint_names)
        for side, joint_names in zip(('left', 'right'), layout.action_names):
            self.register_buffer(side + '_action_indices', torch.tensor([names.index(n) for n in joint_names]))
            signs = torch.ones(len(joint_names))
            if side == 'right':
                signs = _mirror_joint_values(signs, joint_names)
            self.register_buffer(side + '_action_signs', signs)

    def forward(self, obs, actions):
        critic_obs, actor_obs = obs[..., :self.view.input_dim], obs[..., self.view.input_dim:]
        left, right = self.view(critic_obs)
        left = torch.cat((left, actions[..., self.left_action_indices] * self.left_action_signs), -1)
        right = torch.cat((right, actions[..., self.right_action_indices] * self.right_action_signs), -1)
        left, right = self.bank(torch.cat(self.context(actor_obs), -1), left, right)
        return 0.5 * (left + right)


class BilateralSACCritic(Critic):
    """Keep independent distributional Q banks and native categorical projection."""

    def __init__(self, heads, original, input_dim):
        nn.Module.__init__(self)
        self.obs_keys = ['full']
        self.obs_indices = {'full': dict(start=0, end=input_dim, size=input_dim)}
        for name in ('n_act', 'num_atoms', 'v_min', 'v_max', 'num_q_networks'):
            setattr(self, name, getattr(original, name))
        self.qnets = nn.ModuleList(heads)
        self.register_buffer('q_support', original.q_support.clone())


def build_bilateral_sac(env, config, actor, critic):
    cfg = config.bilateral
    layout = BilateralPolicy(env, config.actor_obs_keys, config.critic_obs_keys, cfg)
    layout._joint_names = list(env.robot_config.dof_names)
    n_side = len(layout.action_names[0])

    def network():
        return BilateralMAGNet(
            [layout.actor_view.side_dim, *cfg.hidden_dims, 2 * n_side],
            [layout.critic_view.side_dim + n_side, *cfg.hidden_dims, critic.num_atoms],
            [2 * layout.context_view.side_dim, *cfg.router_dims],
            num_module=cfg.num_module, gate_sizes=cfg.gate_sizes,
            share_ab_modules=cfg.share_ab_modules, use_sym_param=cfg.use_sym_param,
            compose_mode='feature', feature_bias_scale=cfg.feature_bias_scale,
            checkpoint_layers=cfg.checkpoint_layers, use_fast=cfg.use_fast,
            weight_recenter=False,
        )

    first = network()
    result_actor = BilateralSACActor(layout, first, actor)
    router = first.modulator
    heads = [BilateralQ(layout, first if i == 0 else network(), router, critic)
             for i in range(critic.num_q_networks)]
    result_critic = BilateralSACCritic(heads, critic, layout.actor_view.input_dim + layout.critic_view.input_dim)
    return result_actor, result_critic, router
