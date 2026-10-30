"""MAGNet with A/B module banks for bilateral self/cross inference.

Inputs are already in the domain's canonical coordinates. The existing
use_sym_param router supplies paired module activations and feature gates.
"""
import math

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import get_activation_function, get_normalization_function
from .range_layers import MAGNet, ortho_init_module_v2


class BilateralModularLinear(nn.Module):
    """Shared A/B banks, with MAGNet's context-dependent routes on each side."""

    def __init__(self, d_i, d_o, num_module, *, actv='lrelu', norm='ln',
                 ortho_init=False, subnet_std=0.1, preact=False,
                 compose_mode='parameter',
                 share_ab_modules=False,
                 share_bias:bool=False,
                 use_fast=False,
                 feature_bias_scale=1.0,
                 ):
        super().__init__()
        if compose_mode not in ('parameter', 'feature'):
            raise ValueError(f'Unknown compose_mode={compose_mode}')
        self.compose_mode = compose_mode
        self.preact = preact or (compose_mode == 'feature')
        self.share_ab_modules = share_ab_modules
        self.use_fast = use_fast
        self.feature_bias_scale = float(feature_bias_scale)
        self.A = nn.Parameter(th.empty(num_module, d_o, d_i))
        self.B = self.A if share_ab_modules else nn.Parameter(th.empty(num_module, d_o, d_i))
        self.bias = nn.Parameter(th.empty(num_module, d_o))
        bound = 1 / math.sqrt(2 * d_i)
        nn.init.uniform_(self.bias, -bound, bound)
        for weight in ((self.A,) if share_ab_modules else (self.A, self.B)):
            if ortho_init:
                ortho_init_module_v2(weight, std=subnet_std)
            else:
                nn.init.uniform_(weight, -bound, bound)
        self.act = get_activation_function(actv)()
        self.norm = get_normalization_function(norm)(num_features=d_o, affine=False)

    @staticmethod
    def linear(x, bank, alpha=None):
        # (sum_m alpha_m W_m) x, without allocating per-env weight matrices.
        # Take leading dimensions from x so ONNX preserves a dynamic batch.
        y = F.linear(x, bank.flatten(0, 1)).reshape(*x.shape[:-1], *bank.shape[:2])
        return y if alpha is None else BilateralModularLinear.mix(y, alpha)

    @staticmethod
    def mix(features, alpha):
        # Contract the module axis directly; broadcasting multiply/add would
        # allocate extra [batch, modules, channels] tensors. Preserve the old
        # elementwise expression's dtype promotion under mixed precision.
        dtype = th.promote_types(features.dtype, alpha.dtype)
        with th.autocast(device_type=features.device.type, enabled=False):
            return th.matmul(alpha.to(dtype).unsqueeze(-2), features.to(dtype)).squeeze(-2)

    def features(self, x, bank, bias):
        y = self.linear(x, bank)
        # d34e157 split the shared bias equally between the incoming A/B paths.
        if self.feature_bias_scale != 1.0:
            bias = bias * self.feature_bias_scale
        return self.norm(self.act(y + bias)) if self.preact else y

    def _forward_magnet(self, left, right, probs, gates=None):
        """
        Direct copy of current "MAGNet" operation, assuming `preact`.
        order of operation:

        * F.linear (matmul + bias)
        * normalize
        * activate
        * combine module outputs
        * feature-wise gating
        """
        assert(gates is not None)
        gl, gr = gates

        # module-wise predictions
        # requires module-level alignment b/w weight and bias; e.g., A[0] and bias[0]
        Al = self.act(self.norm(th.einsum('myx, ...x -> ...my', self.A, left) + self.bias))
        Ar = self.act(self.norm(th.einsum('myx, ...x -> ...my', self.A, right) + self.bias))
        Bl = self.act(self.norm(th.einsum('myx, ...x -> ...my', self.B, left) + self.bias))
        Br = self.act(self.norm(th.einsum('myx, ...x -> ...my', self.B, right) + self.bias))

        # apply convex combination
        Al = th.einsum('...m, ...my -> ...y', pA, Al)
        Ar = th.einsum('...m, ...my -> ...y', pA, Ar)
        Bl = th.einsum('...m, ...my -> ...y', pB, Bl)
        Br = th.einsum('...m, ...my -> ...y', pB, Br)

        # apply gating at the very end
        y_l = (Al+Br) * gl
        y_r = (Ar+Bl) * gr

        return (y_l, y_r)

    def _forward_fast(self, left, right, probs, gates=None):
        """Pack both sides per bank, preserving the ordinary forward's math.

        Feature mode retains affine -> activation -> norm -> mixing -> gating.
        The final linear layer can mix its bias after the expert reduction.
        No per-sample weight matrices or concatenated A/B weight copies are made.
        """
        inputs = th.stack((left, right))
        pA, pB, pb = (th.stack(p) for p in zip(*probs))
        if self.preact:
            pA, pB = 0.5 * (pA + pb), 0.5 * (pB + pb)
        nonlinear = self.preact and not (
            isinstance(self.act, nn.Identity) and isinstance(self.norm, nn.Identity))
        feature_bias = (self.bias if self.feature_bias_scale == 1.0
                        else self.bias * self.feature_bias_scale)

        def project(bank):
            if nonlinear and not th.is_autocast_enabled(inputs.device.type) and bank.dtype == self.bias.dtype:
                # A 2-D affine lets addmm fuse the bias without a separate
                # [side, batch, module, channel] addition/allocation.
                y = F.linear(inputs.reshape(-1, inputs.shape[-1]),
                             bank.flatten(0, 1), feature_bias.flatten())
                y = y.reshape(*inputs.shape[:-1], *bank.shape[:2])
            else:
                y = self.linear(inputs, bank)
                if self.preact:
                    # Preserve the reference's dtype promotion under autocast.
                    y = y + feature_bias if nonlinear else y.to(th.promote_types(y.dtype, self.bias.dtype))
            if nonlinear:
                # y is a fresh affine result. Reusing its storage makes the
                # activation and LayerNorm backward save the same tensor.
                y = (F.leaky_relu(y, self.act.negative_slope, inplace=True)
                     if isinstance(self.act, nn.LeakyReLU) and self.act.negative_slope >= 0
                     else self.act(y))
                y = self.norm(y)
            return y

        features = project(self.A)
        own = self.mix(features, pA)
        if not self.share_ab_modules:
            del features
            features = project(self.B)
        # Reverse routing, not the large expert-feature tensor: B(left) goes
        # to the right output; B(right) goes to the left output.
        cross = self.mix(features, th.stack((pB[1], pB[0])))
        wx = th.stack((own[0] + cross[1], own[1] + cross[0]))
        if self.preact:
            if not nonlinear:
                wx = wx + self.mix(feature_bias, pA + pB)
            if gates is not None:
                gW, gb = (th.stack(g) for g in zip(*gates))
                wx = wx * (0.5 * (gW + gb))
            return tuple(wx.unbind(0))

        bias = pb @ self.bias
        if gates is None:
            y = wx + bias
        else:
            gW, gb = (th.stack(g) for g in zip(*gates))
            y = gW * wx + gb * bias
        return tuple(self.act(self.norm(y)).unbind(0))

    def forward(self, left, right, probs, gates=None):
        if self.use_fast:
            return self._forward_fast(left, right, probs, gates)
        # With tied banks, self/cross paths differ only in their routes. The
        # expert features for a given input are identical and can be reused.
        shared = ((self.features(left, self.A, self.bias), self.features(right, self.A, self.bias))
                  if self.share_ab_modules else None)
        outputs = []
        for side, (own, other, p, g) in enumerate(zip(
                (left, right), (right, left), probs,
                (None, None) if gates is None else gates)):
            pA, pB, pb = p
            if self.preact:
                # Apply GateModLinear's feature operation independently to A/B.
                # Both branches use the shared bias bank.
                a, b = ((self.features(own, self.A, self.bias), self.features(other, self.B, self.bias))
                        if shared is None else (shared[side], shared[1 - side]))
                alpha_a, alpha_b = 0.5 * (pA + pb), 0.5 * (pB + pb)
                y = self.mix(a, alpha_a) + self.mix(b, alpha_b)
                if g is not None:
                    y = y * (0.5 * (g[0] + g[1]))
                outputs.append(y)
                continue
            wx = (self.linear(own, self.A, pA) + self.linear(other, self.B, pB)
                  if shared is None else
                  self.mix(shared[side], pA) + self.mix(shared[1 - side], pB))
            bias = pb @ self.bias
            # Same gate/norm/activation order as parameter-mode GateModLinear.
            y = wx + bias if g is None else g[0] * wx + g[1] * bias
            outputs.append(self.act(self.norm(y)))
        return tuple(outputs)


class BilateralMAGNet(nn.Module):
    """Independent actor/critic banks and one existing symmetry-tied MAGNet.

    actor_dims/critic_dims specify one side's input, hidden and output widths.
    Context is [z_left, z_right]. Inputs/outputs use the original MAGNet order:
    left actor, left critic, right actor, right critic.
    """

    def __init__(self, actor_dims, critic_dims, mod_dims, *, num_module=4,
                 gate=True, gate_sizes=8, ortho_init=True, subnet_std=0.1,
                 use_sym_param=True, mod_gate='softmax', use_default_module=True,
                 weight_recenter=True, weight_recenter_momentum=0.01,
                 preact=False, compose_mode='parameter', share_ab_modules=False,
                 checkpoint_layers=True, use_fast=False, feature_bias_scale=1.0):
        super().__init__()
        self.compose_mode = compose_mode
        self.preact = preact or (compose_mode == 'feature')
        self.share_ab_modules = share_ab_modules
        # Recompute deterministic inference layers during backward instead of
        # retaining every A/B expert activation. The stateful router/usage EMA
        # stays outside the checkpoint and is evaluated only once.
        self.checkpoint_layers = checkpoint_layers
        self.weight_recenter = weight_recenter
        self.weight_recenter_momentum = weight_recenter_momentum
        if weight_recenter and (not use_default_module or num_module < 2 or mod_gate != 'softmax'):
            raise ValueError('weight_recenter requires a default module and softmax-routed experts.')
        if weight_recenter and self.preact:
            print("WARNING: weight_recenter is exactly function-preserving only "
                  "for compose_mode='parameter' with preact=False.")

        def layers(dims):
            return nn.ModuleList([
                BilateralModularLinear(a, b, num_module,
                    actv='none' if i == len(dims) - 2 else 'lrelu',
                    norm='none' if i == len(dims) - 2 else 'ln',
                    ortho_init=ortho_init, subnet_std=subnet_std,
                    preact=preact, compose_mode=compose_mode,
                    share_ab_modules=share_ab_modules, use_fast=use_fast,
                    feature_bias_scale=feature_bias_scale)
                for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))])

        self.actor, self.critic = layers(actor_dims), layers(critic_dims)
        widths, groups = [], []
        for dims in (actor_dims, critic_dims):
            for i, width in enumerate(dims[1:]):
                widths.extend((width, width))  # weight and bias gates
                group = gate_sizes if i < len(dims) - 2 and gate_sizes is not None else 1
                groups.extend((group, group))
        self.modulator = MAGNet(mod_dims, 2 * len(self.banks), num_module, widths * 2,
            norm='ln', gate=gate, gate_sizes=groups * 2, ortho_init=ortho_init,
            mod_gate=mod_gate, use_default_module=use_default_module,
            use_sym_param=use_sym_param)
        self.register_buffer('running_alpha', th.full(
            (len(self.banks), max(1, num_module - 1)), 1 / max(1, num_module - 1)),
            persistent=weight_recenter)

    @property
    def banks(self):
        # Logical routing slots: keep A/B routes distinct even when weights alias.
        return tuple(p for layer in (*self.actor, *self.critic)
                     for p in (layer.A, layer.B, layer.bias))

    def forward(self, context, inputs):
        probs, gates = self.modulator(context)
        n = len(self.banks)
        if self.training and self.weight_recenter:
            with th.no_grad():
                for average, pl, pr in zip(self.running_alpha, probs[:n], probs[n:]):
                    # Running usage of each shared bank, for the existing recenter hook.
                    alpha = 0.5 * (pl[..., 1:] + pr[..., 1:])
                    average.lerp_(alpha.reshape(-1, alpha.shape[-1]).mean(0),
                                  self.weight_recenter_momentum)
        ng = 2 * (len(self.actor) + len(self.critic))
        pi, gi, outputs = 0, 0, []
        for layers, left, right in ((self.actor, inputs[0], inputs[2]),
                                    (self.critic, inputs[1], inputs[3])):
            for layer in layers:
                p = (probs[pi:pi + 3], probs[n + pi:n + pi + 3])
                g = None if gates is None else (gates[gi:gi + 2], gates[ng + gi:ng + gi + 2])
                if self.checkpoint_layers and self.training and th.is_grad_enabled():
                    left, right = checkpoint(layer, left, right, p, g, use_reentrant=False)
                else:
                    left, right = layer(left, right, p, g)
                pi, gi = pi + 3, gi + 2
            outputs.append((left, right))
        return [outputs[0][0], outputs[1][0], outputs[0][1], outputs[1][1]]

    @th.no_grad()
    def recenter_weights(self, gamma=1.0):
        if not self.weight_recenter or gamma <= 0:
            return {}
        groups = {}
        for i, bank in enumerate(self.banks):
            groups.setdefault(id(bank), (bank, []))[1].append(i)
        norms = []
        for bank, indices in groups.values():
            # Recenter a tied A/B bank once, averaging usage of both branches.
            alpha = self.running_alpha[indices].mean(0)
            alpha = alpha / alpha.sum().clamp_min(1e-8)
            shift = (bank[1:] * alpha.reshape(-1, *([1] * (bank.ndim - 1)))).sum(0) * gamma
            bank[0].add_(shift)
            bank[1:].sub_(shift)
            norms.append(float(shift.norm()))
        return dict(num_banks=len(norms), shift_norm=math.sqrt(sum(x * x for x in norms)),
                    max_shift_norm=max(norms, default=0.0))
