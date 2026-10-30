"""Domi HAMNet (MAGNetMA/MultiRange) with native FastSAC distributions.

The YAML's actor consumes concatenated left/right observations; unlike biHAM,
its full-body expert banks and router have no left/right parameter tying.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from .adapter import BilateralPolicy
from .fast_sac_adapter import BilateralQ, BilateralSACCritic, Slice
from .original.ham_range_layers import MultiRange


class RouterSlice(nn.Module):
    def __init__(self, router, start, stop):
        super().__init__()
        self.router, self.start, self.stop = router, start, stop

    def forward(self, context):
        probs, gates = self.router(context)
        section = slice(self.start, self.stop)
        return probs[section], None if gates is None else gates[section]


class HAMBank(nn.Module):
    """Select one original MultiRange subnet; retain its original forward intact."""
    def __init__(self, model, index, router):
        super().__init__()
        start = sum(model.num_params[:index])
        size = model.num_params[index]
        model.params = nn.ParameterList(list(model.params)[start:start + size])
        model.executors = nn.ModuleList([model.executors[index]])
        model.modulator = RouterSlice(router, start, start + size)
        model.num_params, model.num_net = [size], 1
        self.model = model
        self.checkpoint_layers = False  # Interface used by shared SAC router updates.

    def forward(self, context, left, right):
        return self.model(context, [torch.cat((left, right), -1)])[0]


class HAMFeatures(nn.Module):
    def __init__(self, layout, bank):
        super().__init__()
        self.view, self.context, self.bank = layout.actor_view, layout.context_view, bank
        for key in ('readout', 'left_weight', 'right_weight'):
            self.register_buffer(key, getattr(layout, key).clone())

    def forward(self, obs):
        left, right = self.view(obs)
        lm, ls, rm, rs = self.bank(torch.cat(self.context(obs), -1), left, right).chunk(4, -1)
        mean = lm[..., self.readout] * self.left_weight + rm[..., self.readout] * self.right_weight
        logstd = ls[..., self.readout] * self.left_weight.abs() + rs[..., self.readout] * self.right_weight.abs()
        return torch.cat((mean, logstd), -1)


class HAMQ(BilateralQ):
    def __init__(self, layout, bank, original):
        # Reuse the audited domi action indices/signs without constructing a bilateral bank.
        nn.Module.__init__(self)
        self.view, self.context, self.bank = layout.critic_view, layout.context_view, bank
        self.v_min, self.v_max, self.num_atoms = original.v_min, original.v_max, original.num_atoms
        from .original.loco_io import _mirror_joint_values
        names = layout._joint_names
        for side, joints in zip(('left', 'right'), layout.action_names):
            self.register_buffer(side + '_action_indices', torch.tensor([names.index(n) for n in joints]))
            signs = torch.ones(len(joints))
            self.register_buffer(side + '_action_signs', _mirror_joint_values(signs, joints) if side == 'right' else signs)

    def forward(self, obs, actions):
        critic_obs, actor_obs = obs[..., :self.view.input_dim], obs[..., self.view.input_dim:]
        left, right = self.view(critic_obs)
        left = torch.cat((left, actions[..., self.left_action_indices] * self.left_action_signs), -1)
        right = torch.cat((right, actions[..., self.right_action_indices] * self.right_action_signs), -1)
        # Categorical projection accumulates probability masses in float32.
        return self.bank(torch.cat(self.context(actor_obs), -1), left, right).float()


def build_ham_sac(env, config, actor, critic):
    cfg = config.bilateral
    layout = BilateralPolicy(env, config.actor_obs_keys, config.critic_obs_keys, cfg)
    layout._joint_names = list(env.robot_config.dof_names)
    side = len(layout.action_names[0])

    def network():
        # Exact YAML architecture/defaults from MAGNetMA; only SAC input/output dims differ.
        return MultiRange(cfg.num_module, True, False, [2 * layout.context_view.side_dim, *cfg.router_dims],
            mlp_kwds=[dict(dims=[2 * layout.actor_view.side_dim, *cfg.hidden_dims, 4 * side], norm='ln'),
                      dict(dims=[2 * (layout.critic_view.side_dim + side), *cfg.hidden_dims, critic.num_atoms], norm='ln')],
            gate_sizes=cfg.gate_sizes, ortho_init=True, use_sym_param=False, compose_mode='feature', preact=False)

    model = network()
    router = model.modulator
    actor_bank = HAMBank(model, 0, router)
    # Match native SAC zero-initialized mean/log-std readouts.
    with torch.no_grad():
        for parameter in actor_bank.model.params[-2:]:
            parameter.zero_()
    result_actor = copy.deepcopy(actor)
    result_actor.net = HAMFeatures(layout, actor_bank)
    result_actor.fc_mu, result_actor.fc_logstd = Slice(0, actor.n_act), Slice(actor.n_act, 2 * actor.n_act)
    heads = [HAMQ(layout, HAMBank(network(), 1, router), critic) for _ in range(critic.num_q_networks)]
    return result_actor, BilateralSACCritic(heads, critic, layout.actor_view.input_dim + layout.critic_view.input_dim), router
