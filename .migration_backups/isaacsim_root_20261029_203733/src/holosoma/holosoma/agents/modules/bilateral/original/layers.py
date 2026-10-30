"""Original layer helpers; function/class bodies copied verbatim."""
from typing import Tuple, Optional, Union, Callable
import numpy as np
import torch as th
from torch import nn
import einops
from .export_compat import tensorrt_compat_enabled
LN_NAMES = ["ln", "layer_norm", "layernorm"]
BN_NAMES = ["bn", "batch_norm", "batchnorm"]
GN_NAMES = ["gn", "group_norm", "groupnorm"]

def get_activation_function(actv: str) -> nn.Module:
    if not isinstance(actv, str):
        return actv
    actv = actv.lower()
    if actv == 'tanh':
        out = nn.Tanh
    elif actv == 'relu':
        out = nn.ReLU
    elif actv == 'lrelu':
        out = nn.LeakyReLU
    elif actv == 'leaky_relu':
        out = nn.LeakyReLU
    elif actv == 'elu':
        out = nn.ELU
    elif actv == 'relu6':
        out = nn.ReLU6
    elif actv == 'gelu':
        out = nn.GELU
    elif actv == 'selu':
        out = nn.SELU
    elif actv == 'silu':
        out = nn.SiLU
    elif actv == 'none':
        out = nn.Identity
    else:
        raise KeyError(F'Unknown actv={actv}')
    return out


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 *args, **kwds):
        kwds.pop('num_groups', None)
        super().__init__(num_features, eps, affine=affine)

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


class LayerNorm1d(nn.LayerNorm):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 *args, **kwds):
        kwds.pop('num_groups', None)
        super().__init__(num_features, eps, affine, *args, **kwds)

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


class GroupNorm1d(nn.GroupNorm):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 *args, **kwds):
        # assert num_features % num_groups == 0
        num_groups = kwds.pop('num_groups')
        super().__init__(num_groups, num_features, eps, affine=affine, *args, **kwds)
        # for k, v in self.named_parameters():
        #     print('par', k, v.shape)

        # for k, v in self.named_buffers():
        #     print('buf', k, v.shape)

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_shape = input.shape[:-1]
        out = super().forward(input.reshape(-1, input.shape[-1]))
        return out.view(*batch_shape, out.shape[-1])


def get_normalization_function(norm_cls: str) -> nn.Module:
    if isinstance(norm_cls, Callable):
        return norm_cls

    if norm_cls is None:
        return nn.Identity

    norm_cls = norm_cls.lower()
    if norm_cls in LN_NAMES:
        # NOTE(ycho): LayerNorm1d wraps nn.LayerNorm
        # to provide a consistent interface w.r.t. BatchNorm1d
        out = LayerNorm1d
    elif norm_cls in BN_NAMES:
        out = BatchNorm1d
    elif norm_cls in GN_NAMES:
        out = GroupNorm1d
    elif norm_cls in ['none']:
        out = nn.Identity
    else:
        raise KeyError(F'Unknown norm_cls={norm_cls}')
    return out


class LinearNorm(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 norm: Optional[str] = None,
                 **kwds):
        super().__init__()

        affine = kwds.pop('affine', False)
        bias = kwds.pop('bias', True)
        self.dim_out = dim_out
        self.norm = get_normalization_function(norm)(
            num_features=dim_out,
            affine=affine)
        if isinstance(self.norm, BatchNorm1d):
            bias = False
        self.linear = nn.Linear(
            dim_in, dim_out, bias=bias,
            **kwds)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        if (self.dim_out <= 0):
            # -- empty output --
            return th.empty((*s[:-1], self.dim_out),
                            dtype=x.dtype,
                            device=x.device)
        else:
            x = x.reshape(-1, s[-1])
            x = self.norm(x)
            x = x.reshape(s)
        return x


class MLP(nn.Sequential):
    """ Generic multilayer perceptron. """

    def __init__(self,
                 dims: Tuple[int, ...],
                 bias: bool = True,
                 actv: nn.Module = nn.LeakyReLU,
                 norm: Optional[str] = 'layernorm',
                 affine: bool = False,
                 activate_output: bool = False,
                 **kwds
                 ):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(actv, str):
            actv = get_activation_function(actv)

        layers = []
        # hidden layers
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                (LinearNorm(
                    d0,
                    d1,
                    bias=bias,
                    norm=norm,
                    affine=affine,
                    **kwds
                ),
                    actv(),
                ))
        # last layer
        if activate_output:
            layers.extend((
                LinearNorm(
                    dims[-2],
                    dims[-1],
                    bias=bias,
                    norm=norm,
                    affine=affine,
                    **kwds
                ),
                actv()))
        else:
            # FIXME(ycho): not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        super().__init__(*layers)
        self.dims = dims


class _Symmetric(nn.Module):
    def __init__(self, weight: th.Tensor,) -> None:
        super().__init__()

    def forward(self, weight: th.Tensor) -> th.Tensor:
        if len(weight.shape) == 1:
            # bias
            o, = weight.shape
            w = weight
            return einops.repeat(w[:], '... d -> ... (2 d)')

        if len(weight.shape) == 2:
            # weight
            o, i = weight.shape
            w = weight
            w_self = w[:, :i // 2]
            w_cross = w[:, i // 2:]
            w0 = th.cat([w_self, w_cross], dim=1)
            w1 = th.cat([w_cross, w_self], dim=1)
            w = th.cat([w0, w1], dim=0)
            return w

    def right_inverse(self, value: th.Tensor) -> th.Tensor:
        if len(value.shape) == 1:
            o, = value.shape
            w = value
            return w[:o // 2]
        o, i = value.shape
        w = value
        return w[:o // 2]


def symmetric(module: nn.Linear,) -> nn.Module:
    weight = getattr(module, 'weight', None)
    bias = getattr(module, 'bias', None)

    symw = _Symmetric(weight)
    nn.utils.parametrize.register_parametrization(
        module, 'weight', symw, unsafe=True)
    symb = _Symmetric(bias)
    nn.utils.parametrize.register_parametrization(
        module, 'bias', symb, unsafe=True)

    return module


class SplitDim(nn.Module):
    def __init__(self,
                 sizes: Tuple[int, ...],
                 dim: int = -1):
        super().__init__()
        self.dim = dim
        self.sizes = sizes
        self.splits = np.cumsum(sizes)[:-1].tolist()
        # assert(dim==-1)
        i1 = np.cumsum(sizes).tolist()
        i0 = i1 - np.asarray(sizes)
        self.i0=i0
        self.i1=i1

    def extra_repr(self):
        src = sum(self.sizes)
        return F'{src}->{self.sizes}'

    def forward(self, x: th.Tensor):
        if not tensorrt_compat_enabled():
            return th.tensor_split(x, self.splits, dim=self.dim)
        out = [x[..., a:b] for (a,b) in zip(self.i0, self.i1)]
        # alt = th.tensor_split(x, self.splits, dim=self.dim)
        #print(len(out), len(alt))
        #assert(len(alt) == len(out))
        #for aa, bb in zip(out, alt):
        #    assert( (aa==bb).all())
        return out

