"""No simulator/GPU: check HAM's original executor, SAC updates and comparison configs."""
from holosoma.utils.safe_torch_import import torch, TensorDict
from types import SimpleNamespace
from dataclasses import replace
from contextlib import nullcontext
import copy
import sys
from holosoma.config_values.loco.g1 import experiment as presets
from holosoma.config_values.robot import g1_29dof
from holosoma.agents.fast_sac.fast_sac import Actor, Critic
from holosoma.agents.fast_sac.ham_fast_sac import HAMFastSAC
from holosoma.agents.modules.bilateral.ham_adapter import build_ham_sac, HAMBank
from holosoma.agents.modules.bilateral.original.ham_range_layers import MultiRange
from holosoma.managers.command.manager import CommandManager


def main():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    assert not torch.cuda.is_available()
    bilateral = '--bilateral' in sys.argv
    cfg = presets.g1_29dof_bilateral_no_right_fast_sac_v2 if bilateral else presets.g1_29dof_ham_no_right_fast_sac_v2
    builder, agent_class = build_ham_sac, HAMFastSAC
    if bilateral:
        from holosoma.agents.modules.bilateral.fast_sac_adapter import build_bilateral_sac
        from holosoma.agents.fast_sac.bilateral_fast_sac import BilateralFastSAC
        builder, agent_class = build_bilateral_sac, BilateralFastSAC
    groups = cfg.observation.groups
    dims = dict(actions=29, dof_pos=29, dof_vel=29, base_ang_vel=3, base_lin_vel=3,
                command_lin_vel=2, command_ang_vel=1, projected_gravity=3, sin_phase=2, cos_phase=2)
    class Manager:
        def get_observation_layout(self, keys):
            return [dict(name=n, dim=dims[n], history_length=1) for k in keys for n in sorted(groups[k].terms)]
    env = SimpleNamespace(robot_config=g1_29dof, observation_manager=Manager(), device='cpu',
                          num_envs=256, dt=.02, is_evaluating=False)
    ac = cfg.algo.config
    size = lambda keys: sum(t['dim'] for t in env.observation_manager.get_observation_layout(keys))
    na, nc = size(ac.actor_obs_keys), size(ac.critic_obs_keys)
    indices = lambda key,n: {key:dict(start=0,end=n,size=n)}
    actor = Actor(indices('actor_obs',na), ['actor_obs'],29,4,512,ac.log_std_max,ac.log_std_min,device='cpu')
    critic = Critic(indices('critic_obs',nc),['critic_obs'],29,ac.num_atoms,ac.v_min,ac.v_max,768,device='cpu')
    a,q,r = builder(env,ac,actor,critic)
    rid = {id(p) for p in r.parameters()}
    assert rid == ({id(p) for p in a.parameters()} & {id(p) for p in q.parameters()})
    assert not (({id(p) for p in q.qnets[0].parameters()} - rid) & ({id(p) for p in q.qnets[1].parameters()} - rid))
    # Verify bank selection agrees with the untouched original MultiRange forward.
    m = MultiRange(4,True,False,[6,16], mlp_kwds=[dict(dims=[8,16,4],norm='ln'),dict(dims=[10,16,3],norm='ln')], gate_sizes=8,compose_mode='feature')
    z,x,y=torch.randn(4,6),torch.randn(4,8),torch.randn(4,10)
    expected=m(z,[x,y])[1]
    bank=HAMBank(m,1,m.modulator)
    torch.testing.assert_close(bank(z,y[:,:5],y[:,5:]),expected)
    agent=agent_class.__new__(agent_class)
    agent.config=replace(ac,amp=False)
    agent.device,agent.is_multi_gpu='cpu',False
    agent.actor,agent.qnet,agent.qnet_target=a,q,copy.deepcopy(q)
    agent.scaler=torch.amp.GradScaler('cuda',enabled=False)
    agent._maybe_amp=lambda:torch.autocast("cpu", dtype=torch.bfloat16) if "--amp" in sys.argv else nullcontext()
    agent._router_parameters=list(r.parameters())
    agent._router_grads=[torch.zeros_like(p) for p in agent._router_parameters]
    agent.actor_optimizer=torch.optim.AdamW([p for p in a.parameters() if id(p) not in rid],lr=ac.actor_learning_rate)
    agent.q_optimizer=torch.optim.AdamW([p for p in q.parameters() if id(p) not in rid],lr=ac.critic_learning_rate)
    agent.router_optimizer=torch.optim.AdamW(r.parameters(),lr=ac.actor_learning_rate)
    agent.log_alpha=torch.tensor([-2.],requires_grad=True)
    agent.alpha_optimizer=torch.optim.Adam([agent.log_alpha],lr=.001)
    agent.target_entropy=-29.
    data=TensorDict(dict(observations=torch.randn(4,na),critic_observations=torch.randn(4,nc+na),actions=torch.randn(4,29),next=dict(observations=torch.randn(4,na),critic_observations=torch.randn(4,nc+na),rewards=torch.randn(4),dones=torch.zeros(4),truncations=torch.zeros(4),effective_n_steps=torch.ones(4))),[4])
    before=[p.detach().clone() for p in r.parameters()]
    for loss in (*agent._update_main(data),*agent._update_pol(data)):
        assert torch.isfinite(loss).all()
    assert all(torch.isfinite(g).all() for g in agent._router_grads), [(n, bool(torch.isfinite(g).all())) for (n,p),g in zip(r.named_parameters(),agent._router_grads)]
    agent._finish_update()
    assert any(not torch.equal(b,p) for b,p in zip(before,r.parameters()))
    assert rid=={id(p) for p in r.parameters()}
    assert all(p.requires_grad for p in r.parameters())
    assert all(agent.router_optimizer.state[p]['step']==1 for p in agent._router_parameters)
    assert not ({id(p) for p in agent.qnet_target.parameters()} & rid)
    for loss in (*agent._update_main(data),*agent._update_pol(data)):
        assert torch.isfinite(loss).all()
    agent._finish_update()
    assert all(agent.router_optimizer.state[p]['step']==2 for p in r.parameters())
    assert rid=={id(p) for p in r.parameters()}
    clone=copy.deepcopy(a); clone.load_state_dict(a.state_dict())
    torch.testing.assert_close(clone(data['observations'])[0],a(data['observations'])[0])
    configs=[presets.g1_29dof_bilateral_no_right_fast_sac_v2,presets.g1_29dof_ham_sym_aug_no_right_fast_sac_v2,presets.g1_29dof_ham_no_right_fast_sac_v2,presets.g1_29dof_mlp_no_right_fast_sac_v2]
    assert [c.algo.config.use_symmetry for c in configs]==[False,True,False,False]
    for c in configs:
        assert c.command==cfg.command and c.training.checkpoint is None
        manager=CommandManager(c.command,env,'cpu'); env.command_manager=manager
        manager.setup(); manager.reset(torch.arange(env.num_envs))
        commands=manager.commands
        assert (commands[:,1]>=0).all() and ((commands[:,0]==0)|(commands[:,1]==0)).all()
    print('PASS: original HAM executor equivalence; SAC Q+actor updates; router shared/stepped once; Q bank independence; target isolation; checkpoint roundtrip; four augmentation flags; no-right command sampling.')

if __name__=='__main__':
    main()
