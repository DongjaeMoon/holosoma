"""Reflection-compatible preprocessing; independent of replay augmentation."""
from __future__ import annotations

import torch

from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.augmentation_utils import SymmetryUtils


class DomiSymmetry(SymmetryUtils):
    """Domi's -[sin, cos] reflection, for both Holosoma leg channels."""

    def mirror_obs_sin_phase(self, phase):
        return -phase

    def mirror_obs_cos_phase(self, phase):
        return -phase


class ReflectionNormalization(EmpiricalNormalization):
    """Tie reflected means/variances without adding transitions to replay.

    Projection includes the between-mean variance. Buffers retain the native
    checkpoint format; the signed permutation is reconstructed from the config.
    """

    def __init__(self, shape, device, symmetry, keys):
        super().__init__(shape, device)
        labels = torch.arange(1, shape + 1, device=device).float().unsqueeze(0)
        mirrored = symmetry.mirror_xz_plane(labels, symmetry.env, keys).squeeze(0)
        indices, signs = mirrored.abs().long() - 1, mirrored.sign()
        if not torch.equal(indices[indices], torch.arange(shape, device=device)):
            raise ValueError("Observation reflection must be an involution.")
        if not torch.equal(signs * signs[indices], torch.ones_like(signs)):
            raise ValueError("Observation reflection signs must be an involution.")
        self.register_buffer("mirror_indices", indices, persistent=False)
        self.register_buffer("mirror_signs", signs, persistent=False)

    @torch.no_grad()
    def update(self, x):
        super().update(x)
        reflected_mean = self._mean[:, self.mirror_indices] * self.mirror_signs
        variance = .5 * (self._var + self._var[:, self.mirror_indices])
        variance = variance + .25 * (self._mean - reflected_mean).square()
        self._mean.copy_(.5 * (self._mean + reflected_mean))
        self._var.copy_(variance.clamp_min(0))
        self._std.copy_(self._var.sqrt())
