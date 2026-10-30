#!/usr/bin/env python3

from typing import Tuple, Dict, Iterable, Union, Optional, Mapping
import numpy as np
import torch as th
import torch.nn as nn
from tensordict import TensorDict
from gym import spaces
from domi.net.td.generic import WrapActorCritic
from domi.rl.stable_normal import StableNormalFromLogStd
from tensordict.nn.distributions import CompositeDistribution
from tensordict.utils import NestedKey, unravel_key, unravel_keys
from domi.nn.range_layers import MultiRange


class MAGNetMA(nn.Module):
    def __init__(self,
                 obs_space: spaces.Space,
                 act_space: spaces.Space,
                 rew_space: spaces.Space,
                 aux_space: Optional[spaces.Space],

                 router: Dict[str, Dict[str, Union[NestedKey, Iterable[NestedKey]]]],
                 subnet: Dict[str, Dict[str, Union[NestedKey, Iterable[NestedKey]]]],

                 dim_mod: Tuple[int, ...] = (256, 256, 128),

                 num_module: int = 4,
                 gate: bool = True,
                 fuse: bool = False,
                 gate_sizes: Optional[Union[int, Tuple[int, ...]]] = None,
                 mod_norm: str = 'ln',
                 ortho_init: bool = False,
                 mod_gate: str = 'softmax',
                 mod_noop: bool = False,
                 use_default_module: bool = False,

                 rew_alias: Optional[Dict[str, str]] = None,
                 sub_alias: Optional[Dict[str, str]] = None,
                 slice_action: Optional[Dict[str, Tuple[int, int]]] = None,

                 subnet_std: float = 0.1,
                 layerwise_z: bool = False,
                 use_sym_param: bool = False,
                 penultimate_only: bool = False,
                 preact: bool = False,
                 compose_mode: str = 'parameter',
                 route_scope: str = 'parameter',
                 weight_recenter: bool = False,
                 weight_recenter_momentum: float = 0.1,
                 tie_actor_critic_probs: bool = False,
                 mod_prob_alias: Optional[Dict[str, str]] = None,
                 sync_bilateral_outputs: bool = False,
                 sync_bilateral_actor_outputs: Optional[bool] = None,
                 sync_bilateral_value_outputs: Optional[bool] = None,
                 sync_actor_keys: Tuple[str, str] = ('left', 'right'),
                 sync_actor_dim: int = 3,
                 sync_actor_right_sign: Tuple[float, ...] = (-1.0, -1.0, +1.0),
                 ):
        super().__init__()

        net_spec = {
            # _always_ start with `ctx` part
            'ctx': dict(obs=router['ctx'],
                        act=None, rew=None),
            **subnet
        }
        self.wrap = WrapActorCritic(obs_space,
                                    act_space,
                                    rew_space,
                                    aux_space,
                                    net_spec,
                                    rew_alias,
                                    sub_alias,
                                    slice_action)
        self.sync_bilateral_actor_outputs = (
            sync_bilateral_outputs
            if sync_bilateral_actor_outputs is None
            else sync_bilateral_actor_outputs
        )
        self.sync_bilateral_value_outputs = (
            sync_bilateral_outputs
            if sync_bilateral_value_outputs is None
            else sync_bilateral_value_outputs
        )
        self.sync_actor_keys = tuple(sync_actor_keys)
        self.sync_actor_dim = int(sync_actor_dim)
        self.sync_actor_right_sign = tuple(float(x) for x in sync_actor_right_sign)
        if len(self.sync_actor_keys) != 2:
            raise ValueError("sync_actor_keys must contain exactly two action keys.")
        if self.sync_actor_dim < 0:
            raise ValueError("sync_actor_dim must be non-negative.")
        if len(self.sync_actor_right_sign) < self.sync_actor_dim:
            raise ValueError(
                "sync_actor_right_sign must provide at least sync_actor_dim entries.")

        subnet_keys = list(subnet.keys())
        tie_prob_indices = self._build_tie_prob_indices(
            subnet_keys,
            tie_actor_critic_probs,
            mod_prob_alias)

        mod_dims = [self.wrap.dim_in['ctx'], *dim_mod]
        mlp_kwds = []
        for k, v in subnet.items():
            mlp_kwds.append(dict(
                dims=(self.wrap.dim_in[k],
                      # self.wrap.dim_in['ctx'],
                      *v['dim'],
                      self.wrap.dim_out[k]),
                norm='ln',
                lora_rank=v.get('lora_rank', 0)
                )
            )
        self.log_x_tau = nn.Identity()
        self.model = MultiRange(num_module,
                                gate,
                                fuse,
                                mod_dims,
                                mlp_kwds=mlp_kwds,
                                gate_sizes=gate_sizes,
                                mod_norm=mod_norm,
                                ortho_init=ortho_init,
                                mod_gate=mod_gate,
                                mod_noop=mod_noop,
                                use_default_module=use_default_module,
                                layerwise_z=layerwise_z,
                                subnet_std=subnet_std,
                                use_sym_param=use_sym_param,
                                penultimate_only=penultimate_only,
                                preact=preact,
                                compose_mode=compose_mode,
                                route_scope=route_scope,
                                weight_recenter=weight_recenter,
                                weight_recenter_momentum=weight_recenter_momentum,
                                tie_prob_indices=tie_prob_indices)
        if sub_alias is not None:
            j1s = np.cumsum(self.model.num_params)
            j0s = j1s - self.model.num_params
            for k_dst, k_src in sub_alias.items():
                i_dst = subnet_keys.index(k_dst)
                i_src = subnet_keys.index(k_src)
                j0_src = j0s[i_src]
                j0_dst = j0s[i_dst]
                n_src = self.model.num_params[i_src]
                n_dst = self.model.num_params[i_dst]
                assert (n_src == n_dst)
                for j in range(n_src):
                    # ic(F'SUB_ALIAS {j0_src + j} -> {j0_dst + j}')
                    if self.model.params[j0_dst + j].shape != self.model.params[j0_src + j].shape:
                        print(
                            F'param shape mismatch @ { k_dst }, { k_src }, { j0_src+j } -> {j0_dst+j}'
                        )
                    self.model.params[j0_dst +
                                      j] = self.model.params[j0_src +
                                                             j]
                # NOTE(ycho): handled inside WrapActorCritic
                # if k_dst in self.logstd:
                #     self.logstd[k_dst] = self.logstd[k_src]

    @staticmethod
    def _build_tie_prob_indices(subnet_keys,
                                tie_actor_critic_probs: bool,
                                mod_prob_alias: Optional[Dict[str, str]]):
        prob_alias = {}
        subnet_key_set = set(subnet_keys)
        if tie_actor_critic_probs:
            for key in subnet_keys:
                src_key = None
                if key.endswith('_critic'):
                    src_key = key[:-len('_critic')]
                elif key == 'critic':
                    src_key = 'actor'
                if src_key in subnet_key_set:
                    prob_alias[key] = src_key
        if mod_prob_alias is not None:
            prob_alias.update(mod_prob_alias)

        if not prob_alias:
            return None

        out = {}
        for dst_key, src_key in prob_alias.items():
            if dst_key not in subnet_key_set:
                raise ValueError(f"Unknown probability alias target: {dst_key}")
            if src_key not in subnet_key_set:
                raise ValueError(f"Unknown probability alias source: {src_key}")
            out[subnet_keys.index(dst_key)] = subnet_keys.index(src_key)
        return out

    def _forward(self, s):
        # NOTE(ycho):
        # this unpacking logic only works
        # because we made "net_spec" above start with `ctx`
        z, *ss = s
        ss = self.log_x_tau(ss)
        ys = self.model(z, ss)
        return [None, *ys]

    def _sync_actor_outputs(self, act: Dict[str, th.Tensor]) -> None:
        left_key, right_key = self.sync_actor_keys
        if left_key not in act or right_key not in act:
            return

        left = act[left_key]
        right = act[right_key]
        n = min(self.sync_actor_dim, left.shape[-1], right.shape[-1])
        if n <= 0:
            return

        sign = th.as_tensor(
            self.sync_actor_right_sign[:n],
            dtype=left.dtype,
            device=left.device,
        ).view(*([1] * (left.ndim - 1)), n)
        shared = 0.5 * (left[..., :n] + sign * right[..., :n])
        act[left_key] = th.cat([shared, left[..., n:]], dim=-1)
        act[right_key] = th.cat([sign * shared, right[..., n:]], dim=-1)

    def _forward_synced(self, s):
        act = {}
        val = {}
        val_count = {}

        xs = [
            th.cat([s[k_o] for k_o in self.wrap.obss[k]], dim=-1)
            for k in self.wrap.net_keys
        ]
        ys = self._forward(xs)

        logstd = {}
        for k, y in zip(self.wrap.net_keys, ys):
            if k in self.wrap.acts:
                act_names, act_dims = self.wrap.acts[k]

                a = y
                l = self.wrap.logstd[k]

                if (self.wrap.slice_action is not None) and (
                        k in self.wrap.slice_action):
                    slices = [
                        [int(x) for x in v.split(':')]
                        for v in self.wrap.slice_action[k]['slice']
                    ]

                    a = th.cat([
                        a[..., i0:i1]
                        for (i0, i1) in slices
                    ], dim=-1)

                    l = th.cat([
                        l[..., i0:i1]
                        for (i0, i1) in slices
                    ], dim=-1)

                for k_a, d_a in zip(act_names, act_dims):
                    act[k_a] = a[..., :d_a]
                    logstd[k_a] = l[..., :d_a]

                    a = a[..., d_a:]
                    l = l[..., d_a:]

            if k in self.wrap.rews:
                rew_names, rew_dims = self.wrap.rews[k]
                v = y
                for k_v, d_v in zip(rew_names, rew_dims):
                    v_k = v[..., :d_v]
                    if self.sync_bilateral_value_outputs and k_v in val:
                        val[k_v] = val[k_v] + v_k
                        val_count[k_v] += 1
                    else:
                        val[k_v] = v_k
                        val_count[k_v] = 1
                    v = v[..., d_v:]

        if self.sync_bilateral_actor_outputs:
            self._sync_actor_outputs(act)

        if self.sync_bilateral_value_outputs:
            for k, count in val_count.items():
                if count > 1:
                    val[k] = val[k] / count

        batch_size = xs[0].shape[:-1]
        params = TensorDict(act, batch_size=batch_size)
        act_dist = CompositeDistribution.from_distributions(
            params,
            {
                k: th.distributions.Independent(
                    StableNormalFromLogStd(
                        th.nan_to_num(v),
                        logstd[k],
                        validate_args=False,
                    ),
                    1,
                )
                for (k, v) in act.items()
            },
        )

        if self.wrap.rew_alias is not None:
            val.update({k_dst: val[k_src]
                       for (k_dst, k_src) in self.wrap.rew_alias.items()})
        val = {k: th.nan_to_num(v) for (k, v) in val.items()}
        val = TensorDict(val, batch_size=batch_size)
        return (act_dist, val)

    def forward(self, mem1, z, s):
        if (self.sync_bilateral_actor_outputs
                or self.sync_bilateral_value_outputs):
            return self._forward_synced(s)
        return self.wrap(s, self._forward)
