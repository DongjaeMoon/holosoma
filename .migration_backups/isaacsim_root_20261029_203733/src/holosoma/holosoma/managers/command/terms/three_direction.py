"""Hold out body-frame rightward commands (vy < 0) during training."""
import torch

from .locomotion import LocomotionCommand


class ThreeDirectionCommand(LocomotionCommand):
    def _resample(self, env_ids):
        super()._resample(env_ids)
        if self.commands is None or env_ids.numel() == 0:
            return
        # Keep the original speed distribution, stand probability and yaw sampling.
        speeds = self.commands[env_ids, :2].abs()
        direction = torch.randint(3, (env_ids.numel(),), device=self.env.device)
        self.commands[env_ids, 0] = torch.where(
            direction == 0, speeds[:, 0], torch.where(direction == 1, -speeds[:, 0], 0.0),
        )
        self.commands[env_ids, 1] = torch.where(direction == 2, speeds[:, 1], 0.0)
