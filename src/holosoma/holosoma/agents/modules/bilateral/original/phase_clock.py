"""Clock methods copied from domi PhaseCommand; IsaacLab lifecycle replaced by a small bridge.
Source: /home/user/domi-alps/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_art/mdp/phase_command.py
SHA256: c8f6ed6341fce471a80e323aca2299883e85700a9dae29c4156795464af6cfff
Only contact_phase_reset=False is supported, matching the requested experiment.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Sequence
import numpy as np
import torch as th


class DomiPhaseClock:
    def __init__(self, env, period_range=(.9, 1.1)):
        self._env, self.dt = env, env.dt
        self.num_envs, self.device = env.num_envs, env.device
        self.cfg = SimpleNamespace(period_range=period_range, contact_phase_reset=False)
        self.phase = th.zeros(self.num_envs, device=self.device)
        self.phase_offset = th.zeros_like(self.phase)
        self.phase_period = th.ones_like(self.phase)

    def _resample_command(self, env_ids: Sequence[int]):
        p0, p1 = self.cfg.period_range
        self.phase_period[env_ids] = p0 + (p1-p0) * th.rand(len(env_ids),
                                                               dtype=self.phase_period.dtype,
                                                               device=self.phase_period.device)
        self.phase_offset[env_ids] = (self.phase_period[env_ids]) * th.rand(len(env_ids),
                                                               dtype=self.phase_offset.dtype,
                                                               device=self.phase_offset.device)

    def _episode_time_s(self) -> th.Tensor:
        episode_length = getattr(self._env, "episode_length_buf", None)
        if episode_length is None:
            return th.zeros(self.num_envs, dtype=self.phase.dtype, device=self.device)
        return episode_length.to(dtype=self.phase.dtype) * self.dt

    @property
    def command(self) -> th.Tensor:
        """Return the sin/cos encoding of one normalized gait phase.

        Shape: ``(num_envs, 2)`` holding ``[sin(2π phase), cos(2π phase)]``.
        It is not a left/right phase pair: gait consumers derive the two leg
        phases with their required half-cycle offset.
        """
        command = th.cat(
            (th.sin(2 * np.pi * self.phase).unsqueeze(-1),
             th.cos(2 * np.pi * self.phase).unsqueeze(-1)), dim=-1
        )
        return command

    def _update_command(self):
        """Update the phase command based on the current episode length buffer.

        The phase is normalized to [0, 1) using the configured period.  It is
        a single oscillator with a randomized episode-start offset; consumers
        define left/right leg conventions by applying a half-cycle offset.
        """
        # Compute the normalized phase (0 <= phase < 1)
        elapsed_s = self._episode_time_s()
        self.phase = th.remainder(
            elapsed_s + self.phase_offset, self.phase_period) / self.phase_period
        if self.cfg.contact_phase_reset:
            self._update_phase_from_contact(elapsed_s)
            # Recompute after potentially modifying the physical-time offset.
            self.phase = th.remainder(
                elapsed_s + self.phase_offset, self.phase_period) / self.phase_period

