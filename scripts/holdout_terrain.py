"""Separate parallel evaluation robots on the same flat ground."""
import math

from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion
from holosoma.utils.safe_torch_import import torch


class EvaluationPlane(TerrainLocomotion):
    def _get_env_origins(self):
        # Native eval spawns every robot on tile (0, 0). A grid avoids broadphase
        # buffer exhaustion when evaluating hundreds of independent episodes.
        self._custom_origins = True
        ids = torch.arange(self.num_envs, device=self.device)
        columns = math.ceil(math.sqrt(self.num_envs))
        self._env_origins[:, 0] = 24 + (ids // columns) * 8
        self._env_origins[:, 1] = 24 + (ids % columns) * 8
