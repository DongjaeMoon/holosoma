"""Simulator-free checks for the bilateral integration; run with CUDA_VISIBLE_DEVICES=."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

from holosoma.agents.ppo.bilateral_ppo import BilateralPPO
from holosoma.config_values.loco.g1.experiment import g1_29dof_bilateral_magnet, g1_29dof_bilateral_no_right
from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.robot import g1_29dof
from holosoma.managers.command.manager import CommandManager
from holosoma.managers.command.terms.three_direction import ThreeDirectionCommand


def main():
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 0, 'Run this check with CUDA_VISIBLE_DEVICES= (CPU only).'
    dims = dict(actions=29, dof_pos=29, dof_vel=29, base_ang_vel=3, base_lin_vel=3,
                command_lin_vel=2, command_ang_vel=1, projected_gravity=3, sin_phase=2, cos_phase=2)
    groups = g1_29dof_loco_single_wolinvel.groups
    class Manager:
        cfg = SimpleNamespace(groups=groups)
        def term_shape(self, group, name):
            return (dims[name],)
        def get_obs_dims(self):
            return {key: sum(dims[name] for name in group.terms) for key,group in groups.items()}
        def get_observation_layout(self, keys):
            return [dict(name=name, dim=dims[name], history_length=1) for key in keys for name in sorted(groups[key].terms)]
    env = SimpleNamespace(robot_config=g1_29dof, observation_manager=Manager(), device='cpu',
        history_length={'actor_obs':1,'critic_obs':1}, num_envs=256, dt=0.02, is_evaluating=False)
    algo = BilateralPPO.__new__(BilateralPPO)
    algo.env, algo.device, algo.is_multi_gpu = env, 'cpu', False
    algo.config = g1_29dof_bilateral_magnet.algo.config
    algo._init_config()
    algo._setup_models_and_optimizer()
    policy = algo.critic.policy
    a, c = torch.randn(8, policy.actor_view.input_dim), torch.randn(8, policy.critic_view.input_dim)
    action, value = algo._evaluate_actor_critic(a,c)
    mean, std, logp = algo.actor.action_mean.detach(), algo.actor.action_std.detach(), algo.actor.get_actions_log_prob(action).detach()
    assert action.shape == (8,29) and value.shape == (8,1)
    torch.testing.assert_close(policy(a,c+10)[0], mean)
    algo._bootstrap_value({'actor_obs':a+1,'critic_obs':c+1})
    torch.testing.assert_close(algo.actor.get_actions_log_prob(action),logp)
    actor, critic = algo._optimizer_parameters()
    assert len({id(p) for p in actor+critic}) == len(actor+critic)
    batch = dict(actor_obs=a,critic_obs=c,actions=action.detach(),values=value.detach(),returns=torch.randn(8,1),
        advantages=torch.randn(8,1),actions_log_prob=logp[:,None],action_mean=mean,action_sigma=std)
    losses = algo._update_algo_step(batch,dict(Value=0.0,Surrogate=0.0,Entropy=0.0,KL=0.0))
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    for parameter in policy.network.modulator.parameters():
        assert algo.actor_optimizer.state[parameter]['step'] == 1
        assert parameter not in algo.critic_optimizer.state
    print('PASS: native PPO setup, joint update, single router step, privileged-input isolation, bootstrap distribution')
    manager=CommandManager(g1_29dof_bilateral_no_right.command,env,'cpu')
    env.command_manager=manager
    manager.setup(); manager.reset(torch.arange(env.num_envs))
    assert isinstance(manager.get_state('locomotion_command'),ThreeDirectionCommand)
    commands=manager.commands
    assert (commands[:,1]>=0).all() and ((commands[:,0]==0)|(commands[:,1]==0)).all()
    assert (commands[:,0]>0).any() and (commands[:,0]<0).any() and (commands[:,1]>0).any()
    print('PASS: real command manager samples forward/backward/left, no rightward commands')
    algo._eval_mode()
    with TemporaryDirectory() as temporary:
        path=Path(temporary)/'bilateral.onnx'
        torch.onnx.export(algo.actor_onnx_wrapper,(a[:1],),str(path),input_names=['actor_obs'],output_names=['actions'],
            dynamic_axes={'actor_obs':{0:'batch'},'actions':{0:'batch'}},opset_version=17,dynamo=False)
        import onnx
        onnx.checker.check_model(onnx.load(str(path)))
    print('PASS: ONNX export and graph validation')
    print('Network parameters:',sum(p.numel() for p in algo.actor.parameters()))


if __name__=='__main__':
    main()
