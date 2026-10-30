from __future__ import annotations
import copy
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.agents.modules.bilateral.fast_sac_adapter import build_bilateral_sac
from holosoma.agents.modules.bilateral.original.fast_sac_updates import BilateralUpdates
from holosoma.utils.safe_torch_import import torch


class BilateralFastSAC(BilateralUpdates, FastSACAgent):
    network_builder = staticmethod(build_bilateral_sac)

    def setup(self):
        super().setup()
        self.actor, self.qnet, router = self.network_builder(self.unwrapped_env, self.config, self.actor, self.qnet)
        self.actor.to(self.device)
        self.qnet.to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet)
        self._router_parameters = list(router.parameters())
        ids = {id(p) for p in self._router_parameters}
        def optimizer(params, lr):
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.config.weight_decay,
                                    fused=str(self.device).startswith('cuda'), betas=(0.9, 0.95))
        self.actor_optimizer = optimizer([p for p in self.actor.parameters() if id(p) not in ids], self.config.actor_learning_rate)
        self.q_optimizer = optimizer([p for p in self.qnet.parameters() if id(p) not in ids], self.config.critic_learning_rate)
        self.router_optimizer = optimizer(self._router_parameters, self.config.actor_learning_rate)
        self._router_grads = [torch.zeros_like(p) for p in self._router_parameters]
        self.policy = self.actor.explore
        print(f'{type(self).__name__}: actor and independent Q banks share one router; target router is separate.', flush=True)

    def _sample_and_prepare_batches(self, *args, **kwargs):
        batches = super()._sample_and_prepare_batches(*args, **kwargs)
        for data in batches:
            for block in (data, data['next']):
                block['critic_observations'] = torch.cat((block['critic_observations'], block['observations']), -1)
        return batches

    def _reset_router_grads(self):
        self.router_optimizer.zero_grad(set_to_none=True)
        for gradient in self._router_grads:
            gradient.zero_()

    def _collect_router_grads(self):
        scale = self.scaler.get_scale()
        for parameter, gradient in zip(self._router_parameters, self._router_grads):
            if parameter.grad is not None:
                gradient.add_(parameter.grad.detach() / scale)
                parameter.grad = None

    def _policy_q(self, obs, actions):
        # functional_call on aliased shared routers can leave detached tensors
        # registered in PyTorch 2.4. Freeze Q weights without replacing objects.
        parameters = list(self.qnet.parameters())
        flags = [p.requires_grad for p in parameters]
        previous = [head.bank.checkpoint_layers for head in self.qnet.qnets]
        for parameter in parameters:
            parameter.requires_grad_(False)
        for head in self.qnet.qnets:
            head.bank.checkpoint_layers = False
        try:
            return self.qnet(obs, actions)
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)
            for head, value in zip(self.qnet.qnets, previous):
                head.bank.checkpoint_layers = value

    def _finish_update(self):
        for parameter, gradient in zip(self._router_parameters, self._router_grads):
            parameter.grad = gradient
        limit = self.config.max_grad_norm if self.config.max_grad_norm > 0 else float('inf')
        norm = torch.nn.utils.clip_grad_norm_(self._router_parameters, limit)
        if torch.isfinite(norm):
            self.router_optimizer.step()

    def save(self, path):
        super().save(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        state['router_optimizer_state_dict'] = self.router_optimizer.state_dict()
        torch.save(state, path)

    def load(self, path):
        super().load(path)
        if path:
            state = torch.load(path, map_location=self.device, weights_only=False)
            if 'router_optimizer_state_dict' in state:
                self.router_optimizer.load_state_dict(state['router_optimizer_state_dict'])
