"""PPO with the original bilateral actor/critic and a single shared router."""
import torch

from holosoma.agents.modules.bilateral.adapter import BilateralActor, BilateralCritic, BilateralPolicy
from holosoma.agents.ppo.ppo import PPO


class BilateralPPO(PPO):
    def _build_models(self):
        if self.config.bilateral is None:
            raise ValueError("BilateralPPO requires a bilateral network configuration.")
        policy = BilateralPolicy(self.env, self.actor_obs_keys, self.critic_obs_keys, self.config.bilateral).to(self.device)
        self.actor = BilateralActor(
            policy, self.config.module_dict.actor, self.num_act, self.config.init_noise_std,
        ).to(self.device)
        self.critic = BilateralCritic(policy)

    def _optimizer_parameters(self):
        network = self.critic.policy.network
        # A joint backward sends both losses to the router, which is stepped once
        # at the actor LR. The critic optimizer owns only its private banks.
        actor = [self.actor.std, *network.actor.parameters(), *network.modulator.parameters()]
        return actor, list(network.critic.parameters())

    def _evaluate_actor_critic(self, actor_obs, critic_obs):
        mean, value = self.critic.policy(actor_obs, critic_obs)
        self.actor.update_distribution_from_mean(mean)
        return self.actor.distribution.sample(), value

    def _bootstrap_value(self, obs_dict):
        actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
        actor_obs = self._normalize_actor_obs(actor_obs, update=False)
        critic_obs = self._normalize_critic_obs(critic_obs, update=False)
        return self.critic.policy(actor_obs, critic_obs)[1]
