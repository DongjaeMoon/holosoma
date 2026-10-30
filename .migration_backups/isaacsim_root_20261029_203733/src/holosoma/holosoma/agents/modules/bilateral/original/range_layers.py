"""Original router and its helpers; function/class bodies copied verbatim."""
from typing import Tuple, Iterable, Optional, List, Union, Mapping
import math
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import einops
from .layers import MLP, LayerNorm1d, SplitDim, symmetric
COMP_SUM = True

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

