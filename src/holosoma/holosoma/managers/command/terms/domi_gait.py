"""Bridge the original domi oscillator to Holosoma's two-leg gait interface."""
from __future__ import annotations

import torch

from holosoma.agents.modules.bilateral.original.phase_clock import DomiPhaseClock
from holosoma.managers.command.terms.locomotion import LocomotionGait


class DomiGait(LocomotionGait):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.clock = DomiPhaseClock(env, tuple(cfg.params.get("period_range", (.9, 1.1))))

    def _initialize_indices(self, env_ids, *, evaluating):
        self.resample_frequency(self._ensure_index_tensor(env_ids))
        self.step()

    def resample_frequency(self, env_ids):
        ids = self._ensure_index_tensor(env_ids)
        self.clock._resample_command(ids)
        self.gait_freq[ids, 0] = 1 / self.clock.phase_period[ids]
        self.phase_dt[ids, 0] = 2 * torch.pi * self.env.dt * self.gait_freq[ids, 0]

    def standing(self):
        command = getattr(self.manager, "commands", None)
        if command is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return (torch.linalg.vector_norm(command[:, :2], dim=1) < .01) & (command[:, 2].abs() < .01)

    def step(self):
        self.clock._update_command()
        angle = 2 * torch.pi * self.clock.phase
        # Domi phase zero: left swing peak, right stance. Same convention as
        # Holosoma's unchanged foot-height reward at [0, -pi].
        self.phase[:, 0] = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi
        self.phase[:, 1] = torch.remainder(angle, 2 * torch.pi) - torch.pi
        self.phase[self.standing()] = self.stand_phase_value

    def encoding(self):
        # Original loco_env_cfg.phase_zero_stand, using Holosoma's stand mask.
        return self.clock.command * (1.0 - self.standing()[:, None].float())


def sin_phase(env):
    value = env.command_manager.get_state("locomotion_gait").encoding()[:, :1]
    return torch.cat((value, -value), -1)


def cos_phase(env):
    value = env.command_manager.get_state("locomotion_gait").encoding()[:, 1:]
    return torch.cat((value, -value), -1)
