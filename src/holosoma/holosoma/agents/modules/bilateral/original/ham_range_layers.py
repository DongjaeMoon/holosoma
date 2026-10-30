from __future__ import annotations
#!/usr/bin/env python3

from typing import (
    Tuple,
    Iterable,
    Optional,
    List,
    Union,
    Mapping
)
from itertools import islice

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
from opt_einsum import contract_expression

from .layers import (
    MLP,
    LayerNorm1d,
    SplitDim,
    get_activation_function,
    get_normalization_function,
    symmetric
)

from icecream import ic
from functools import partial
import string

COMP_SUM: bool = True


def batch_outer_einsum(*tensors):
    n = len(tensors)
    if n > 25:
        raise ValueError("Too many tensors for einsum alphabet.")

    letters = string.ascii_lowercase[:n]
    # Creates string like "...a,...b,...c->...abc"
    input_subs = ",".join([f"...{l}" for l in letters])
    output_sub = f"...{''.join(letters)}"
    equation = f"{input_subs}->{output_sub}"

    # Compute the high-dimensional outer product
    res = th.einsum(equation, *tensors)

    # Flatten all dimensions created by the outer product
    return res.flatten(-n)


def split_by_sizes(lst, sizes):
    it = iter(lst)
    return [list(islice(it, size)) for size in sizes]


def f_Wx(bi, bm, mij, bj):
    # bj, bm -> bjm
    bjm = bj.unsqueeze(2) * bm.unsqueeze(1)

    # bjm, mij -> bi
    # mji = mij.permute(0, 2, 1)  # [M, J, I]
    intermediate_bi = th.tensordot(
        bjm, mij,
        dims=([1, 2], [2, 0])
    )

    # bi, bi -> bi
    result = bi * intermediate_bi  # [B, I]

    return result


def f_b(bj, bm, mj):
    # mj,bm->jb                             bj,jb->bj
    # jb,bj->bj                                bj->bj
    return th.matmul(bm, mj) * bj


def find_closest_factors(X, Y, M):
    """Given X, Y, and M, find integers A and B such that:

    1. A * B = M
    2. (X * A) / (Y * B) is closest to 1.
    """

    # --- 1. Calculate the Ideal A ---
    # Handle edge case to avoid division by zero
    if X == 0 or Y == 0:
        return None, None  # Or raise an error

    # --- 2. & 3. Find the Closest Factor ---
    best_A = 0
    best_B = 0
    min_cost = float('inf')

    # Iterate up to the square root of M
    for i in range(1, int(math.sqrt(M)) + 1):
        if M % i != 0:
            continue

        # --- Check the first factor: A = i ---
        A1 = i
        B1 = M // i
        # distance1 = abs(A1 - target_A)
        cost1 = abs(math.log((A1 * X) / (Y * B1)))

        if cost1 < min_cost:
            min_cost = cost1
            best_A = A1
            best_B = B1

        A2, B2 = B1, A1
        cost2 = abs(math.log((A2 * X) / (Y * B2)))
        if cost2 < min_cost:
            min_cost = cost2
            best_A = A2
            best_B = B2

    # --- 4. Return the Best Pair ---
    return best_A, best_B


def ortho_init_module(weight: th.Tensor,
                      std: float = 1.0,
                      act: str = 'leaky_relu'):
    w = weight
    num_mod, dim_out, dim_in = w.shape
    rep_in, rep_out = find_closest_factors(dim_in, dim_out, num_mod)
    with th.no_grad():
        w = einops.rearrange(w, '(ry rx) y x -> (ry y) (rx x)',
                             rx=rep_in, ry=rep_out)
        iso_gain: float = max(
            1, math.sqrt(
                w.shape[0] / w.shape[1])) * math.sqrt(rep_in)
        act_gain: float = nn.init.calculate_gain(act)
        # act_gain: float = 1.0
        gain: float = iso_gain * act_gain * std
        w = nn.init.orthogonal_(w, gain=gain)
        w = einops.rearrange(w, '(ry y) (rx x) -> (ry rx) y x',
                             rx=rep_in, ry=rep_out)
        weight.copy_(w.clone())

        # -- validate --
        if False:
            batch_size: int = 512
            x = th.randn((batch_size, w.shape[-1]),
                         dtype=th.float32,
                         device=weight.device)
            y = th.einsum('...x, myx -> ...my', x, weight)
            y = F.leaky_relu(y)
            c = th.einsum('...ad,...bd -> ab', y, y) / batch_size / y.shape[-1]
            print(c)

def _act_gain(act: str = "linear", param: float | None = None) -> float:
    """Return PyTorch's activation-specific variance-preserving gain."""
    if act in (None, "identity", "none"):
        act = "linear"
    if param is None:
        return float(nn.init.calculate_gain(act))
    return float(nn.init.calculate_gain(act, param))


def _mscale(
    num_module: int,
    dim_out: int,
    dim_in: int,
) -> float:
    stack_dim_out = num_module * dim_out
    if stack_dim_out <= dim_in:
        return math.sqrt(num_module)
    return math.sqrt(min(dim_out, dim_in) / dim_in) * num_module


def ortho_init_module_v2(
    weight: th.Tensor,
    std: float = 1.0,
    act: str = "linear",
    act_param: float | None = None,
    generator: th.Generator | None = None,
):
    num_module, dim_out, dim_in = weight.shape
    base_gain = _act_gain(act, act_param) * float(std)
    gain = base_gain * _mscale(num_module, dim_out, dim_in)
    with th.no_grad():
        stack = weight.view(num_module * dim_out, dim_in)
        nn.init.orthogonal_(stack, gain=gain, generator=generator)
    return weight

class GateModLinear(nn.Module):
    def __init__(self,
                 d_i: int,
                 d_o: int,
                 actv='tanh',
                 norm='layernorm',
                 gate: bool = True,
                 fuse: bool = False,
                 native: bool = False,
                 preact: bool = False,
                 compose_mode: str = 'parameter',
                 lora_rank:int=0):
        super().__init__()
        if compose_mode not in ('parameter', 'feature'):
            raise ValueError(f"Unknown compose_mode={compose_mode}")
        self.d_i = d_i
        self.d_o = d_o
        self.act = get_activation_function(actv)()
        self.norm = get_normalization_function(norm)(
            num_features=d_o,
            # NOTE(ycho): GateModLinear is a parameter-free layer
            # Also, affine=False is generally a good idea
            # since we're already applying `gate` anyway.
            affine=False
        )
        self.gate = gate
        self.fuse = fuse
        self.compose_mode = compose_mode
        self.preact = preact or (compose_mode == 'feature')

        # NOTE(ycho): rough guesses here are fine.
        b = 1024 * 10  # batch size
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        if gate:
            self.f_Wx = contract_expression(
                'bi,bm,mij,bj->bi',
                (b, i),
                (b, m),
                (m, i, j),
                (b, j),
                #    (b,i),
                #    backend='torch'
            )
            if native:
                self.f_Wx = f_Wx
            # print(d_i, d_o)
            # print(self.f_Wx)
            self.f_b = contract_expression(
                'bj,bm,mj->bj',
                (b, j),
                (b, m),
                (m, j),
                #    (b,j),
                #    backend='torch'
            )
            if native:
                self.f_b = f_b
            # ic(self.f_Wx)
            # ic(self.f_b)
        else:
            self.f_Wx = contract_expression(
                'bm,mij,bj->bi',
                (b, m),
                (m, i, j),
                (b, j),
            )
            self.f_b = contract_expression(
                'bm,mj->bj',
                (b, m),
                (m, j)
            )

        self.lora_rank=lora_rank
        if lora_rank > 0:
            assert(not self.fuse)
            self.register_parameter('Wl_i',
                                    nn.Parameter(
                                        th.zeros((lora_rank, d_i), dtype=th.float32),
                                        requires_grad=True
                                    ))
            self.register_parameter('Wl_o',
                                    nn.Parameter(
                                        th.zeros((d_o, lora_rank), dtype=th.float32),
                                        requires_grad=True
                                    ))
            with th.no_grad():
                nn.init.kaiming_uniform_(self.Wl_i, 1.0 / lora_rank)
                nn.init.zeros_(self.Wl_o)


    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def _lora(self, x: th.Tensor) -> th.Tensor:
        return th.einsum('yk,kx,...x->...y',
                         self.Wl_o,
                         self.Wl_i,
                         x)

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: Optional[th.Tensor] = None,
                gb: Optional[th.Tensor] = None):

        if self.preact:
            if False:
                # old impl
                if self.gate:
                    wx = th.einsum('mij, ...j -> ...mi', Ws, x)
                    anwx = self.act(self.norm(wx))
                    Wx = th.einsum('...i, ...m, ...mi-> ...i', gW, pW, anwx)
                    b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
                    return Wx + b
                else:
                    anwx = self.norm(self.act(
                        th.einsum(
                            'mij, ...j -> ...mi',
                            Ws,
                            x)))
                    Wx = th.einsum('...m, ...mi-> ...i', pW, anwx)
                    b = th.einsum('...m, mj -> ...j', pb, bs)
                    return Wx + b
            else:
                # new impl (*requires #W == #b)
                p = 0.5 * (pW + pb)
                wx = th.einsum('mij, ...j -> ...mi', Ws, x)
                if self.lora_rank > 0:
                    wx = wx + self._lora(x).unsqueeze(dim=-2)
                wxpb = self.norm(self.act(wx + bs))
                if self.gate:
                    g = 0.5 * (gW + gb)
                    return th.einsum('...m, ...mi, ...i -> ...i', p, wxpb, g)
                else:
                    return th.einsum('...m, ...mi -> ...i', p, wxpb)

        if True:
            if self.gate:
                # ic(gW.shape,
                #    pW.shape,
                #    Ws.shape,
                #    x.shape)
                # gW.shape: torch.Size([1, 4, 64])
                # pW.shape: torch.Size([1, 4, 8])
                # Ws.shape: torch.Size([8, 64, 462])
                # x.shape: torch.Size([1, 4, 462])
                if self.fuse:
                    y = th.empty(
                        (*x.shape[:-1], self.d_o + 1),
                        dtype=x.dtype,
                        device=x.device,
                        requires_grad=True)
                    Wxpb = self.f_Wx(gW, pW,
                                     th.cat([Ws, bs[..., None]], dim=-1),
                                     F.pad(x, (0, 1), 'constant', 1.0),
                                     out=y[..., :-1]
                                     )
                    # print(y[..., :-1] - Wxpb)
                    out1 = self.act(self.norm(Wxpb))
                    # comparison with "standard" operation
                    # Wx = self.f_Wx(gW, pW, Ws, x)
                    # b = self.f_b(gW, pW, bs)
                    # out2 = self.act(self.norm(Wx + b))
                    # print('minus', out1 - out2)
                    return out1
                else:
                    Wx = self.f_Wx(gW, pW, Ws, x)
                    if self.lora_rank > 0:
                        Wx = Wx + self._lora(x)
                    b = self.f_b(gb, pb, bs)
            else:
                Wx = self.f_Wx(pW, Ws, x)
                if self.lora_rank > 0:
                    Wx = Wx + self._lora(x)
                b = self.f_b(pb, bs)
        else:
            Wx = th.einsum('...i, ...m, mij, ...j -> ...i', gW, pW, Ws, x)
            if self.lora_rank > 0:
                Wx = Wx + self._lora(x)
            b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
        return self.act(self.norm(Wx + b))

class GateModMLP(nn.Module):
    """
    Specialized executor for MLPs without internal buffers.
    """

    def __init__(self,
                 # Same args as MLP
                 dims: Tuple[int, ...],
                 actv: nn.Module = nn.LeakyReLU,
                 activate_output: bool = False,
                 norm: str = 'layernorm',
                 bias: bool = True,

                 # Extra `gate` arg
                 gate: bool = True,
                 fuse: bool = False,
                 native: bool = False,
                 preact: bool = False,
                 compose_mode: str = 'parameter',
                 lora_rank:int=0
                 ):
        super().__init__()
        num_layer: int = len(dims) - 1
        last_idx: int = num_layer - 1
        self.gate = gate
        self.layers = nn.ModuleList([
            GateModLinear(d_i, d_o,
                          actv='none' if (
                              (i_l == last_idx) and (not activate_output)
                          ) else actv,
                          norm='none' if (
                              (i_l == last_idx) and (not activate_output)
                          ) else norm,
                          gate=gate,
                          fuse=fuse,
                          native=native,
                          preact=preact,
                          compose_mode=compose_mode,
                          lora_rank=lora_rank)
            for i_l, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:]))
        ])

    def forward(self, x: th.Tensor,
                # weight/bias (modules)
                # shape: Lx(M,Di,Do)
                Ws: Iterable[th.Tensor],
                # shape: Lx(M,Do)
                bs: Iterable[th.Tensor],

                # module selection probabilities
                # shape: (...,L,M)
                pWs: th.Tensor,
                # shape: (...,L,M)
                pbs: th.Tensor,

                # output gates
                # shape: Lx(..., Do)
                gWs: Optional[th.Tensor] = None,
                # shape: Lx(..., Do)
                gbs: Optional[th.Tensor] = None):
        if self.gate:
            for layer, W, b, pW, pb, gW, gb in zip(self.layers,
                                                   Ws, bs, pWs, pbs, gWs, gbs):
                x = layer(x, W, b, pW, pb, gW.squeeze(dim=-1), gb)
        else:
            for layer, W, b, pW, pb in zip(self.layers, Ws, bs, pWs, pbs):
                x = layer(x, W, b, pW, pb)
        return x


class TempSoftmax(nn.Softmax):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.register_buffer('temperature',
                             nn.Parameter(th.ones((), dtype=th.float32),
                                          requires_grad=False))

    def forward(self, x: th.Tensor):
        return super().forward(x / self.temperature)


class MAGNet(nn.Module):
    """
    Modulation-And-Gating Network.

    Args:
        dims : Backbone dims (dims[0]=input, dims[-1]=latent task embedding dim)
        num_param : number of independent parameters (#weights+#biases)
        num_module: number of compatible modules per parameter
        out_dims: (for gate only) feature dimensions at each layer
        init_std: initial standard deviations of scale headers
    """

    def __init__(self,
                 dims: Tuple[int, ...],
                 num_param: int,
                 num_module: int,
                 out_dims: Tuple[int, ...],

                 actv: str = 'gelu',
                 norm: str = 'layernorm',
                 affine: bool = True,
                 init_std: float = 0.008,
                 gate_sizes: Optional[Tuple[int, ...]] = None,
                 ortho_init: bool = False,
                 mod_gate: str = 'softmax',
                 gate: bool = True,
                 mod_noop: bool = False,
                 use_default_module: bool = False,
                 layerwise_z: bool = False,
                 use_sym_param: bool = False,
                 route_scope: str = 'parameter',
                 ):
        super().__init__()
        self.num_param = num_param
        self.num_module = num_module
        route_scope = {'param': 'parameter'}.get(route_scope, route_scope)
        if route_scope not in ('parameter', 'layer', 'network'):
            raise ValueError(f"Unknown route_scope={route_scope}")
        if layerwise_z and route_scope != 'parameter':
            raise ValueError("layerwise_z only supports route_scope='parameter'")
        if route_scope == 'layer' and num_param % 2 != 0:
            raise ValueError("route_scope='layer' expects interleaved weight/bias pairs")
        self.route_scope = route_scope
        self.num_route = {
            'parameter': num_param,
            'layer': num_param // 2,
            'network': 1,
        }[route_scope]

        if isinstance(num_module, Mapping):
            if COMP_SUM:
                total_num_module = np.sum(list(num_module.values()))
            else:
                total_num_module = np.prod(list(num_module.values()))
        else:
            total_num_module = num_module
        self.total_num_module = total_num_module

        self.gate = gate
        self.use_default_module = use_default_module
        self.layerwise_z = layerwise_z

        # predict task embeddings
        if isinstance(dims, Mapping):
            # multiple backbones
            self.backbone = nn.ModuleDict({
                k: MLP(v, actv=actv,
                       norm=norm,
                       affine=affine,
                       activate_output=True)
                for (k, v) in dims.items()
            })

            gate_dim: int = 0
            for k, v in dims.items():
                gate_dim += v[0]

            # FIXME(ycho): hardcoded gate backbone dimensions
            self.backbone['gate'] = MLP([gate_dim, 256, 128],
                                        actv=actv,
                                        norm=norm,
                                        affine=affine,
                                        activate_output=True)
        else:
            # single backbone
            if len(dims) == 1:
                self.backbone = nn.Identity()
            else:
                back_dims = list(dims)
                if layerwise_z:
                    back_dims[-1] *= num_param
                self.backbone = MLP(back_dims,
                                    actv=actv,
                                    norm=norm,
                                    affine=affine,
                                    activate_output=True)

        # linearly project to logit outputs
        self.logits = None
        if mod_noop:
            assert (total_num_module == dims[-1]
                    or total_num_module * self.num_route == dims[-1])
            self.logits = nn.Identity()
        else:
            num_logits = total_num_module
            if isinstance(num_module, Mapping):
                # assert(not self.use_default_module)
                self.logits = nn.ModuleDict({
                    k: nn.Linear(dims[k][-1], self.num_route * v)
                    for k, v in num_module.items()
                })
            else:
                if self.use_default_module:
                    num_logits -= 1
                if num_logits > 1:
                    self.logits = nn.Linear(dims[-1], self.num_route * num_logits)
        self.on_logits = nn.Identity()

        # also output gates (scales)
        self.out_dims = out_dims
        h_out = sum(out_dims)

        if mod_gate == 'softmax':
            self.mod_gate = nn.Softmax(dim=-1)
        elif mod_gate == 't_softmax':
            self.mod_gate = TempSoftmax(dim=-1)
        elif mod_gate == 'sigmoid':
            self.mod_gate = nn.Sigmoid()
        elif mod_gate == 'identity':
            self.mod_gate = nn.Identity()
        else:
            raise ValueError(F'Unknown mod_gate={mod_gate}')

        # Orthogonal initialization for the logit projectors
        # that belong to the same layer.
        if (self.logits is not None) and ortho_init:
            def _ortho_init(layer, m: int):
                if not hasattr(layer, 'weight'):
                    return
                with th.no_grad():
                    for i0 in range(0, self.num_route * m, m):
                        i1 = i0 + m
                        nn.init.orthogonal_(layer.weight[i0:i1, :])

            if isinstance(num_module, Mapping):
                layers = list(self.logits.values())
                for k, l in self.logits.items():
                    _ortho_init(l, m=num_module[k])
            else:
                _ortho_init(self.logits, m=num_module)

        if self.gate:
            self.gate_sizes = gate_sizes
            if isinstance(dims, Mapping):
                scale_input_dim = self.backbone['gate'].dims[-1]
            else:
                scale_input_dim = dims[-1]

            if gate_sizes is None:
                # disable grouped gating
                self.scale_header = nn.Linear(
                    scale_input_dim, h_out)
                self.split_header = SplitDim(self.out_dims)
            else:
                # Enable grouped gating (except last layer)
                assert (len(out_dims) == len(gate_sizes))
                grouped_out_dims = []
                for (d, g) in zip(out_dims, gate_sizes):
                    assert ((d % g) == 0)
                    grouped_out_dims.append(d // g)
                h_out = sum(grouped_out_dims)
                self.scale_header = nn.Linear(
                    scale_input_dim, h_out)
                self.split_header = SplitDim(grouped_out_dims)

            # init gate with small std
            with th.no_grad():
                layer = self.scale_header
                nn.init.uniform_(layer.weight, -init_std, init_std)
                nn.init.zeros_(layer.bias)

        if use_sym_param:
            # NOTE(ycho): hardcoded assumption
            # that everything should be
            # organized in (left.{actor,critic}, right.{actor,critic} fashion)
            for k, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    print('apply sym @', k)
                    symmetric(m)
                if isinstance(m, LayerNorm1d):
                    print('apply sym @', k)
                    symmetric(m)
        self.z_tau_np = None

    def _expand_route_probs(self, p: th.Tensor) -> th.Tensor:
        if self.route_scope == 'parameter':
            return p
        if self.route_scope == 'layer':
            p = p.repeat_interleave(2, dim=-2)
            return p[..., :self.num_param, :]
        if self.route_scope == 'network':
            return p.expand(*p.shape[:-2], self.num_param, p.shape[-1])
        raise RuntimeError(f"Unknown route_scope={self.route_scope}")

    def forward(self, x: th.Tensor) -> Tuple[List[th.Tensor],
                                             List[th.Tensor]]:
        if isinstance(x, th.Tensor):
            z = self.backbone(x)
        else:
            x = dict(x)
            x['gate'] = th.cat(list(x.values()), dim=-1)
            z = {k: v(x[k]) for (k, v) in self.backbone.items()}

        # NOTE(ycho): deprecated `z_tau` override from file-cache for debugging
        # if self.z_tau_np is not None:
        #     z[...] = th.as_tensor(self.z_tau_np,
        #                           dtype=z.dtype,
        #                           device=z.device)

        # -> module activations
        if self.logits is not None:
            num_output_module = self.total_num_module - self.use_default_module
            if isinstance(self.logits, nn.Identity):
                # FIXME(ycho): somewhat fragile check
                if z.shape[-1] == self.num_module:
                    p = einops.repeat(z, '... m -> ... p m',
                                      p=self.num_route)
                else:
                    p = z.unflatten(-1, (self.num_route, -1))
            else:
                if isinstance(self.logits, nn.ModuleDict):
                    y = {
                        k: th.softmax(self.logits[k](z[k]).unflatten(-1, (self.num_route, self.num_module[k])), dim=-1)
                        for k in self.logits.keys()
                    }
                    if COMP_SUM:
                        p = th.cat(list(y.values()), dim=-1)
                    else:
                        p = batch_outer_einsum(*list(y.values()))
                else:
                    if self.layerwise_z:
                        z_l = z.unflatten(-1, (self.num_param, -1))
                        w = self.logits.weight.unflatten(
                            0, (self.num_param, -1))
                        b = self.logits.bias.unflatten(-1,
                                                       (self.num_param, -1))
                        y = th.einsum('...lx, lyx -> ...ly', z_l, w) + b
                    else:
                        y = self.logits(z)  # 16,64
                    y = self.on_logits(y)
                    y = y.reshape(*x.shape[:-1],
                                  self.num_route,
                                  num_output_module)
                    p = self.mod_gate(y)
            p = self._expand_route_probs(p)
        else:
            # module routing is sometimes disabled (ex: when num_module<=1)
            p = th.ones(*x.shape[:-1], self.num_param, self.num_module,
                        dtype=x.dtype,
                        device=x.device)
        if self.use_default_module:
            # the first module is used "by default" in this code-path.
            p = F.pad(p, (1, 0), mode='constant', value=1.0)
        p = th.unbind(p, dim=-2)

        # -> gates
        scale = None
        if self.gate:
            if self.layerwise_z:
                z_l = z.unflatten(-1, (self.num_param, -1))
                zs = th.unbind(z_l, dim=-2)
                ws = th.tensor_split(self.scale_header.weight,
                                     self.split_header.splits,
                                     dim=0)
                bs = th.tensor_split(self.scale_header.bias,
                                     self.split_header.splits,
                                     dim=0)
                scale = [F.linear(z_, w, b) for (z_, w, b)
                         in zip(zs, ws, bs)]
                # w = self.scale_header.weight.unflatten(0, (self.num_param, -1))
                # b = self.scale_header.bias.unflatten(-1, (self.num_param, -1))
                # scale = 1.0 + th.einsum('...lx, lyx -> ...ly', z, w) + b
            else:
                if isinstance(z, Mapping):
                    z_gate = z['gate']
                else:
                    z_gate = z
                scale = 1 + self.scale_header(z_gate)
                scale = self.split_header(scale)

            if self.gate_sizes is not None:
                scale = [
                    einops.repeat(s, '... s -> ... (s g)', g=g)
                    for (s, g) in
                    zip(scale, self.gate_sizes)
                ]

        return p, scale


class Range(nn.Module):
    """
    Reconfigurable & Modular Architecture
    """

    def __init__(self,
                 num_module: int,
                 gate: bool,

                 # --magnet args--
                 mod_dims: Tuple[int, ...],
                 mod_actv: str = 'gelu',
                 mod_norm: str = 'ln',
                 mod_affine: bool = True,
                 init_gate_std: float = 0.008,

                 # --subnet args--
                 # -> MLP/GateModMLP
                 subnet_args=[],
                 subnet_kwds={},
                 fuse: bool = False,
                 ):
        super().__init__()
        self.gate = gate

        # == parameters ==
        subnet_affine = subnet_kwds.pop('affine',
                                        False)
        assert (not subnet_affine)
        paramss = []
        for i in range(num_module):
            subnet = MLP(*subnet_args,
                         **subnet_kwds,
                         affine=subnet_affine)
            params = subnet.parameters()
            # NOTE(ycho): for now, we don't allow buffers.
            bufs = subnet.buffers()
            assert (len(list(bufs)) == 0)
            params = nn.ParameterList(params)
            paramss.append(params)
        stacked_params = [th.stack(p, dim=0) for p in zip(*paramss)]
        self.params = nn.ParameterList(stacked_params)
        # == modulator ==
        self.modulator = MAGNet(
            mod_dims,
            len(self.params),
            num_module,
            # NOTE(ycho):
            # "p.shape[1]" logic is valid because:
            # for weight-type: (M, d_o, d_i)[1] = d_o
            # for bias-type: (M, d_o)[1] = d_o
            [p.shape[1] for p in self.params],

            actv=mod_actv,
            norm=mod_norm,
            affine=mod_affine,
            init_std=init_gate_std,

        )
        # == executor ==
        self.executor = GateModMLP(*subnet_args,
                                   **subnet_kwds,
                                   gate=gate,
                                   fuse=fuse)

    def forward(self,
                z: th.Tensor,
                x: th.Tensor):
        N: int = len(self.params)

        weights = self.params[:N:2]
        biases = self.params[1:N:2]

        probs, scales = self.modulator(z)
        pWs = probs[:N:2]
        pbs = probs[1:N:2]
        gWs = (scales[:N:2] if self.gate else None)
        if gWs is not None:
            # Append extra dim to account for d_i
            # since W shape = (d_o, d_i)
            gWs = [s[..., None] for s in gWs]
        gbs = (scales[1:N:2] if self.gate else None)

        return self.executor(x,
                             weights,
                             biases,
                             pWs, pbs,
                             gWs, gbs)


class RangeActorCritic(nn.Module):
    """
    Range Network (Actor-Critic Variant)
    TODO(ycho): support arbitrary number of subnets?
    """

    def __init__(self,
                 num_module: int,
                 gate: bool,
                 fuse: bool,

                 # --magnet args--
                 mod_dims: Tuple[int, ...],
                 mod_actv: str = 'gelu',
                 mod_norm: str = 'ln',
                 mod_affine: bool = True,
                 init_gate_std: float = 0.008,

                 # --subnet args--
                 # -> MLP/GateModMLP
                 actor_args=[],
                 actor_kwds={},
                 critic_args=[],
                 critic_kwds={},

                 gate_sizes: Optional[Union[int, Tuple[int, ...]]] = None
                 ):
        super().__init__()
        self.gate = gate
        self.gate_sizes = gate_sizes

        # == parameters ==
        paramss = []

        self.num_params = []
        for i in range(num_module):
            ps = []
            for sa, sk in [(actor_args, actor_kwds),
                           (critic_args, critic_kwds)]:
                subnet_affine = sk.pop('affine',
                                       False)
                assert (not subnet_affine)
                subnet = MLP(*sa, **sk,
                             affine=subnet_affine)
                params = list(subnet.parameters())
                # NOTE(ycho): for now, we don't allow buffers.
                bufs = subnet.buffers()
                assert (len(list(bufs)) == 0)
                ps.extend(params)

                # FIXME(ycho): this is a hacky workaround
                # that exploits the fact that
                # actor_args/actor_kwds iterates
                # before `critic_args/critic_kwds`.
                if i == 0:
                    self.num_params.append(len(params))
            params = nn.ParameterList(ps)
            paramss.append(params)
        stacked_params = [th.stack(p, dim=0) for p in zip(*paramss)]
        self.params = nn.ParameterList(stacked_params)
        # == modulator ==
        self.modulator = MAGNet(
            mod_dims,
            len(self.params),
            num_module,
            # NOTE(ycho):
            # "p.shape[1]" logic is valid because:
            # for weight-type: (M, d_o, d_i)[1] = d_o
            # for bias-type: (M, d_o)[1] = d_o
            [p.shape[1] for p in self.params],

            actv=mod_actv,
            norm=mod_norm,
            affine=mod_affine,
            init_std=init_gate_std,
            gate_sizes=self.gate_sizes

        )
        # == executor ==
        # TODO(ycho): consider parallelizing
        self.actor_executor = GateModMLP(*actor_args,
                                         **actor_kwds,
                                         gate=gate,
                                         fuse=fuse)
        self.critic_executor = GateModMLP(*critic_args,
                                          **critic_kwds,
                                          gate=gate,
                                          fuse=fuse)

    def forward(self,
                z: th.Tensor,
                x: th.Tensor):
        batch_shape = z.shape[:-1]

        # flatten batch dims
        z = z.view(-1, z.shape[-1])
        x = x.view(-1, x.shape[-1])

        probs, scales = self.modulator(z)

        S = self.num_params[0]
        q_a, q_c = list(self.params[:S]), list(self.params[S:])
        p_a, p_c = probs[:S], probs[S:]
        g_a, g_c = scales[:S], scales[S:]

        out = []
        for (q, p, g, f) in [
                (q_a, p_a, g_a, self.actor_executor),
                (q_c, p_c, g_c, self.critic_executor)]:

            weights = q[::2]
            biases = q[1::2]

            pWs = p[::2]
            pbs = p[1::2]
            gWs = (g[::2] if self.gate else None)
            if gWs is not None:
                # Append extra dim to account for d_i
                # since W shape = (d_o, d_i)
                gWs = [s[..., None] for s in gWs]
            gbs = (g[1::2] if self.gate else None)

            y = f(x,
                  weights, biases,
                  pWs, pbs,
                  gWs, gbs)

            # restore batch dims
            y = y.view(*batch_shape, y.shape[-1])
            out.append(y)
        return out


class MultiRange(nn.Module):
    """
    Range Network (Multiple subnetwork variant)
    """

    def __init__(self,
                 num_module: int,
                 gate: bool,
                 fuse: bool,

                 # --magnet args--
                 mod_dims: Tuple[int, ...],
                 mod_actv: str = 'gelu',
                 mod_norm: str = 'ln',
                 mod_affine: bool = True,
                 init_gate_std: float = 0.008,
                 mod_prob: str = 'softmax',

                 # --subnet args--
                 # -> MLP/GateModMLP
                 # No longer allows vanilla args, for simplicity.
                 mlp_kwds=[],

                 # --optional args--
                 native: bool = False,
                 gate_sizes: Optional[Union[int, Tuple[int, ...]]] = None,
                 ortho_init: bool = False,
                 mod_gate: str = 'softmax',
                 mod_noop: bool = False,
                 use_default_module: bool = False,
                 layerwise_z: bool = False,
                 subnet_std: float = 0.1,
                 use_sym_param: bool = False,
                 penultimate_only: bool = False,
                 preact: bool = False,
                 compose_mode: str = 'parameter',
                 route_scope: str = 'parameter',
                 weight_recenter: bool = False,
                 weight_recenter_momentum: float = 0.1,
                 tie_prob_indices: Optional[Mapping[int, int]] = None):
        super().__init__()
        self.gate = gate
        # ALWAYS enabled, for now
        # assert (self.gate)
        self.gate_sizes = gate_sizes

        # == parameters ==
        paramss = []

        self.num_params = []

        if isinstance(num_module, Mapping):
            if COMP_SUM:
                total_num_module = np.sum(list(num_module.values()))
            else:
                total_num_module = np.prod(list(num_module.values()))
        else:
            total_num_module = num_module
        self.total_num_module = int(total_num_module)
        self.use_default_module = bool(use_default_module)
        self.compose_mode = compose_mode
        self.weight_recenter = bool(weight_recenter)
        self.weight_recenter_momentum = float(weight_recenter_momentum)
        self.tie_prob_indices = {
            int(k): int(v)
            for k, v in (tie_prob_indices or {}).items()
        }
        if self.weight_recenter:
            if not self.use_default_module:
                raise ValueError(
                    "weight_recenter requires use_default_module=True.")
            if isinstance(num_module, Mapping):
                raise ValueError(
                    "weight_recenter does not support Mapping num_module yet.")
            if mod_gate not in ('softmax', 't_softmax'):
                raise ValueError(
                    "weight_recenter expects normalized non-default routes.")
            if self.total_num_module < 2:
                raise ValueError(
                    "weight_recenter needs at least one default and one expert module.")
            if compose_mode != 'parameter':
                print(
                    "WARNING: weight_recenter is exactly function-preserving "
                    "only for compose_mode='parameter'.")

        for i in range(total_num_module):
            ps = []
            for kwds in mlp_kwds:
                subnet_affine = kwds.pop('affine', False)
                assert (not subnet_affine)
                # FIXME(ycho): temporary hack
                lora_rank = kwds.pop('lora_rank', 0)
                subnet = MLP(**kwds, affine=subnet_affine)
                kwds['lora_rank'] = lora_rank

                params = list(subnet.parameters())
                # NOTE(ycho): for now, we don't allow buffers.
                bufs = subnet.buffers()
                assert (len(list(bufs)) == 0)
                ps.extend(params)

                # Track parameter vector sizes (@ module#0)
                if i == 0:
                    self.num_params.append(len(params))
            params = nn.ParameterList(ps)
            paramss.append(params)
        stacked_params = [th.stack(p, dim=0) for p in zip(*paramss)]
        self.params = nn.ParameterList(stacked_params)

        for dst, src in self.tie_prob_indices.items():
            if dst == src:
                continue
            if not (0 <= dst < len(self.num_params)):
                raise ValueError(f"Invalid tied probability target index={dst}")
            if not (0 <= src < len(self.num_params)):
                raise ValueError(f"Invalid tied probability source index={src}")
            if self.num_params[dst] != self.num_params[src]:
                raise ValueError(
                    "Tied probability subnets must have matching layer layouts: "
                    f"{dst} has {self.num_params[dst]} params, "
                    f"{src} has {self.num_params[src]}")

        if self.weight_recenter:
            num_expert = self.total_num_module - 1
            running_alpha = th.full(
                (len(self.params), num_expert),
                1.0 / float(num_expert),
                dtype=th.float32)
            self.register_buffer('running_alpha', running_alpha)
        else:
            self.register_buffer(
                'running_alpha', th.empty(0, dtype=th.float32),
                persistent=False)

        if ortho_init:
            with th.no_grad():
                for p in self.params:
                    # weight matrix of shape (num_module, dim_out, dim_in)
                    if len(p.shape) != 3:
                        continue

                    if False:
                        pr = p.reshape(p.shape[0] * p.shape[1], p.shape[2])
                        gain = math.sqrt(2.0) * math.sqrt(2.0 /
                                                          (p.shape[1] + p.shape[2]))
                        nn.init.orthogonal_(pr, gain=gain)
                    else:
                        # ??
                        ortho_init_module_v2(p, std=subnet_std)

        # In case `gate_sizes` was given as an int,
        # apply all interim features with the given gate size
        # _except_ the last layer.
        if gate and isinstance(self.gate_sizes, int):
            g = self.gate_sizes
            self.gate_sizes = [g] * len(self.params)
            gate_sizes_per_net = split_by_sizes(
                self.gate_sizes,
                self.num_params
            )
            for gs in gate_sizes_per_net:
                if self.gate:
                    # weight & bias
                    gs[-2:] = [1, 1]
                else:
                    # weight only
                    gs[-2:] = [1, 1]
            self.gate_sizes = np.concatenate(gate_sizes_per_net)

        # == modulator ==
        self.modulator = MAGNet(
            mod_dims,
            len(self.params),
            num_module,
            # NOTE(ycho):
            # "p.shape[1]" logic is valid because:
            # for weight-type: (M, d_o, d_i)[1] = d_o
            # for bias-type: (M, d_o)[1] = d_o
            [p.shape[1] for p in self.params],

            actv=mod_actv,
            norm=mod_norm,
            affine=mod_affine,
            init_std=init_gate_std,
            gate_sizes=self.gate_sizes,
            ortho_init=ortho_init,
            mod_gate=mod_gate,
            gate=gate,
            mod_noop=mod_noop,
            use_default_module=use_default_module,
            layerwise_z=layerwise_z,
            use_sym_param=use_sym_param,
            route_scope=route_scope
        )
        self.penultimate_only = penultimate_only
        # == executor ==
        # TODO(ycho): consider parallelizing
        self.executors = nn.ModuleList([
            GateModMLP(**kwds,
                       gate=gate,
                       fuse=fuse,
                       preact=preact,
                       compose_mode=compose_mode)
            for kwds in mlp_kwds
        ])
        self.num_net = len(self.executors)

    def _apply_tied_prob_indices(self, probs: Iterable[th.Tensor]):
        if not self.tie_prob_indices:
            return probs
        ps = split_by_sizes(probs, self.num_params)
        for dst, src in self.tie_prob_indices.items():
            if dst == src:
                continue
            ps[dst] = ps[src]
        return [p for group in ps for p in group]

    def _update_recenter_statistics(self, probs: Iterable[th.Tensor]):
        if (not self.weight_recenter) or (not self.training):
            return
        momentum = self.weight_recenter_momentum
        if momentum <= 0.0:
            return
        with th.no_grad():
            for i, prob in enumerate(probs):
                alpha = prob[..., 1:]
                alpha = alpha.reshape(-1, alpha.shape[-1]).mean(dim=0)
                alpha = alpha.to(
                    device=self.running_alpha.device,
                    dtype=self.running_alpha.dtype)
                self.running_alpha[i].mul_(1.0 - momentum).add_(
                    alpha, alpha=momentum)
                self.running_alpha[i].clamp_(min=0.0)
                self.running_alpha[i].div_(
                    self.running_alpha[i].sum().clamp_min(1.0e-8))

    @th.no_grad()
    def recenter_weights(self, gamma: float = 1.0):
        if not self.weight_recenter:
            return {}
        gamma = float(gamma)
        if gamma <= 0.0:
            return {}

        num_expert = self.total_num_module - 1
        groups = {}
        for i, param in enumerate(self.params):
            entry = groups.setdefault(id(param), [param, []])
            entry[1].append(i)

        shift_sq = 0.0
        max_shift = 0.0
        num_banks = 0
        for param, indices in groups.values():
            alpha = self.running_alpha[indices].mean(dim=0)
            alpha = alpha.to(device=param.device, dtype=param.dtype)
            alpha = alpha.clamp_min(0.0)
            alpha = alpha / alpha.sum().clamp_min(1.0e-8)
            alpha = alpha.view(num_expert, *([1] * (param.ndim - 1)))

            expected = (alpha * param[1:]).sum(dim=0)
            shift = expected.mul(gamma)
            param[0].add_(shift)
            param[1:].sub_(shift.unsqueeze(0))

            shift_norm = float(shift.norm().detach().cpu())
            shift_sq += shift_norm * shift_norm
            max_shift = max(max_shift, shift_norm)
            num_banks += 1

        return {
            'num_banks': num_banks,
            'shift_norm': math.sqrt(shift_sq),
            'max_shift_norm': max_shift,
        }

    def forward(self,
                z: th.Tensor,
                xs: Iterable[th.Tensor]):
        if isinstance(z, Mapping):
            # FIXME(ycho): assumes single level nesting only
            batch_shape = next(iter(z.values())).shape[:-1]
            z = {k: v.view(-1, v.shape[-1]) for (k, v) in z.items()}
        else:
            batch_shape = z.shape[:-1]
            z = z.view(-1, z.shape[-1])

        # flatten batch dims
        xs = [x.view(-1, x.shape[-1]) for x in xs]

        probs, scales = self.modulator(z)
        probs = self._apply_tied_prob_indices(probs)
        self._update_recenter_statistics(probs)
        # ic([p.shape for p in probs],
        #    [s.shape for s in scales])

        S = self.num_params[0]

        qs = split_by_sizes(self.params, self.num_params)
        ps = split_by_sizes(probs, self.num_params)

        gs = None
        if scales is not None:
            gs = split_by_sizes(scales, self.num_params)

        out = []
        for i in range(self.num_net):
            # Unpack inputs.
            q = qs[i]
            p = ps[i]
            g = None
            if gs is not None:
                g = gs[i]
            f = self.executors[i]
            x = xs[i]

            if self.penultimate_only:
                # Disable all meaningful modulations _except_
                # the penultimate layer
                num_wb = len(p)  # weights & biases, interleaved order
                indices = np.arange(num_wb)
                for j in indices[:-4]:
                    p[j] = th.full_like(p[j], 1.0 / p[j].shape[-1])
                for j in indices[-2:]:
                    p[j] = th.full_like(p[j], 1.0 / p[j].shape[-1])

            # params are interleaved (w,b)
            weights = q[0::2]
            biases = q[1::2]

            # probs are also interleaved (w,b)
            pWs = p[0::2]
            pbs = p[1::2]

            # gates are also interleaved (w,b)
            if g is None:
                y = f(x, weights, biases, pWs, pbs)
            else:
                gWs = (g[0::2] if self.gate else None)
                if gWs is not None:
                    # Append extra dim to account for d_i
                    # since W shape = (d_o, d_i)
                    gWs = [s[..., None] for s in gWs]
                gbs = (g[1::2] if self.gate else None)

                y = f(x,
                      weights, biases,
                      pWs, pbs,
                      gWs, gbs)

            # restore batch dims
            y = y.view(*batch_shape, y.shape[-1])
            out.append(y)
        return out


def test_range():

    dim_b: int = 7
    dim_z: int = 8
    dim_x: int = 2
    dim_y: int = 3

    net = Range(4, True,
                # MAGNet args
                mod_dims=[dim_z, 4],
                # subnet MLP args
                subnet_kwds=dict(
                    dims=[dim_x, 8, 2, dim_y],
                    norm='ln'
                )
                )
    ic(net)
    z = th.randn((dim_b, dim_z))
    x = th.randn((dim_b, dim_x))
    y = net(z, x)
    ic('y', y.shape)


def test_range_ac():
    dim_b: int = 7
    dim_z: int = 8
    dim_x: int = 2
    dim_a: int = 3
    dim_v: int = 4

    net = RangeActorCritic(4, True, True,
                           # MAGNet args
                           mod_dims=[dim_z, 4],
                           # subnet MLP args
                           actor_kwds=dict(
                               dims=[dim_x, 3, 5, 3, dim_a],
                               norm='ln'
                           ),
                           critic_kwds=dict(
                               dims=[dim_x, 7, 4, dim_v],
                               norm='ln'
                           ))
    ic(net)
    z = th.randn((dim_b, dim_z))
    x = th.randn((dim_b, dim_x))
    a, v = net(z, x)
    ic('a', a.shape)
    ic('v', v.shape)
    v.sum().backward()


def test_multi_range():
    dim_b: int = 7
    dim_z: int = 8
    dim_x: int = 2

    dim_a: int = 3
    dim_v: int = 4
    dim_a2: int = 5

    for native in [False, True]:
        th.random.manual_seed(0)
        th.cuda.manual_seed_all(0)

        net = MultiRange(4, True, False,
                         # MAGNet args
                         mod_dims=[dim_z, 4],
                         # subnet MLP args
                         mlp_kwds=[
                             dict(
                                 dims=[dim_x, 3, 5, 3, dim_a],
                                 norm='ln'
                             ),
                             dict(
                                 dims=[dim_x, 7, 4, dim_v],
                                 norm='ln'
                             ),
                             dict(
                                 dims=[dim_x, 7, 2, dim_a2],
                                 norm='ln'
                             ),
                         ],
                         native=native)
        ic(net)
        z = th.randn((dim_b, dim_z))
        x = th.randn((dim_b, dim_x))
        a, v, a2 = net(z, [x, x, x])
        if native:
            a0, v0, a20 = a, v, a2
        if native:
            print(a - a0, v - v0, a20 - a2)
        ic('a', a.shape)
        ic('v', v.shape)
        ic('a2', a2.shape)
        v.sum().backward()


def test_multi_grouped_gating_range():
    dim_b: int = 7
    dim_z: int = 8
    dim_x: int = 2

    dim_a: int = 3
    dim_v: int = 4
    dim_a2: int = 5

    for native in [False, True]:
        th.random.manual_seed(0)
        th.cuda.manual_seed_all(0)

        net = MultiRange(4, True, False,
                         # MAGNet args
                         mod_dims=[dim_z, 4],
                         # subnet MLP args
                         mlp_kwds=[
                             dict(
                                 dims=[dim_x, 8, 8, 8, dim_a],
                                 norm='ln'
                             ),
                             dict(
                                 dims=[dim_x, 16, 16, dim_v],
                                 norm='ln'
                             ),
                             dict(
                                 dims=[dim_x, 32, 32, dim_a2],
                                 norm='ln'
                             ),
                         ],
                         native=native,
                         # NOTE(ycho): we need to repeat twice
                         # (once for weight // once for bias dim)
                         # gate_sizes=[
                         #     3, 3, 5, 5, 3, 3, 1, 1,
                         #     7, 7, 4, 4, 1, 1,
                         #     7, 7, 2, 2, 1, 1,
                         # ]
                         gate_sizes=4
                         )
        ic(net)
        z = th.randn((dim_b, dim_z))
        x = th.randn((dim_b, dim_x))
        a, v, a2 = net(z, [x, x, x])
        if native:
            a0, v0, a20 = a, v, a2
        if native:
            print(a - a0, v - v0, a20 - a2)
        ic('a', a.shape)
        ic('v', v.shape)
        ic('a2', a2.shape)
        v.sum().backward()


def test_ortho_init_module():
    m: int = 8
    y: int = 2
    x: int = 16
    w = th.randn((m, y, x))
    ortho_init_module(w, std=0.1)

def test_ortho_init_module_v2():
    w = th.zeros((4, 64, 64))
    m:int = w.shape[0]
    d:int=w.shape[-2]//2
    ortho_init_module_v2(w, std=0.1)

    w2 = w.clone()
    with th.no_grad():
        w2[:d, :d] = ortho_init_module_v2(w2[:d, :d].clone(), std=0.1)
        w2[d:, d:] = ortho_init_module_v2(w2[d:, d:].clone(), std=0.1)
        w2[d:, :d] = ortho_init_module_v2(w2[d:, :d].clone(), std=0.1)
        w2[:d, d:] = ortho_init_module_v2(w2[:d, d:].clone(), std=0.1)

    x = th.randn((128,w.shape[-2]))
    alpha = th.randn((m,))
    print('orig')
    print(th.einsum('...m, myx, ...x -> ...y', alpha, w, x).std(dim=-1).mean())
    print('block')
    print(th.einsum('...m, myx, ...x -> ...y', alpha, w2, x).std(dim=-1).mean())

def main():
    # test_multi_grouped_gating_range()
    test_ortho_init_module_v2()


if __name__ == '__main__':
    main()
