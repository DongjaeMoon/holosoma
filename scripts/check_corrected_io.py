"""CPU regression: biased statistics, physical reflection, phase and saved IO."""
from holosoma.utils.safe_torch_import import torch
from types import SimpleNamespace as NS
import copy

from holosoma.config_values.loco.g1 import experiment as presets
from holosoma.agents.fast_sac.fast_sac import Actor, Critic
from holosoma.agents.modules.bilateral.fast_sac_adapter import build_bilateral_sac
from holosoma.agents.modules.bilateral.normalization import DomiSymmetry, ReflectionNormalization
from holosoma.managers.command.manager import CommandManager


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    cfg = presets.g1_29dof_bilateral_no_right_fast_sac_v2
    dims = dict(actions=29,dof_pos=29,dof_vel=29,base_ang_vel=3,base_lin_vel=3,
                command_lin_vel=2,command_ang_vel=1,projected_gravity=3,sin_phase=2,cos_phase=2)
    class Manager:
        def __init__(self):
            self.cfg=cfg.observation
        def _compute_term(self,group,name,term):
            return torch.zeros(1,dims[name])
        def term_shape(self,group,name):
            return (dims[name],)
        def get_observation_layout(self,keys):
            return [dict(name=n,dim=dims[n],history_length=1) for k in keys for n in sorted(self.cfg.groups[k].terms)]
    env=NS(robot_config=cfg.robot,observation_manager=Manager(),device='cpu',num_envs=128,dt=.02,
           is_evaluating=False,episode_length_buf=torch.zeros(128),history_length={'actor_obs':1,'critic_obs':1})
    symmetry=DomiSymmetry(env)
    size=lambda keys:sum(t['dim'] for t in env.observation_manager.get_observation_layout(keys))
    ac=cfg.algo.config
    na,nc=size(ac.actor_obs_keys),size(ac.critic_obs_keys)
    indices=lambda key,n:{key:dict(start=0,end=n,size=n)}
    actor=Actor(indices('actor_obs',na),['actor_obs'],29,1,ac.actor_hidden_dim,ac.log_std_max,ac.log_std_min,device='cpu')
    critic=Critic(indices('critic_obs',nc),['critic_obs'],29,ac.num_atoms,ac.v_min,ac.v_max,ac.critic_hidden_dim,device='cpu')
    actor,q,_=build_bilateral_sac(env,ac,actor,critic)
    # Nonzero heads: zero initialization would hide defects in the equivariance test.
    with torch.no_grad():
        for p in actor.net.bank.parameters():
            if p.numel() and p.detach().abs().max()==0:
                p.normal_(std=.03)
    obs=[]
    normalizers=[]
    for n,keys in ((na,ac.actor_obs_keys),(nc,ac.critic_obs_keys)):
        norm=ReflectionNormalization(n,'cpu',symmetry,keys)
        x=torch.randn(128,n)+torch.linspace(-2,3,n)
        for _ in range(3): norm.update(x)
        norm.eval()
        reflect=lambda value:symmetry.mirror_xz_plane(value,env,keys)
        torch.testing.assert_close(norm(reflect(x)),reflect(norm(x)),rtol=1e-6,atol=1e-6)
        clone=ReflectionNormalization(n,'cpu',symmetry,keys); clone.load_state_dict(norm.state_dict()); clone.eval()
        torch.testing.assert_close(clone(x),norm(x))
        obs.append((norm(x),norm(reflect(x))))
        normalizers.append(norm)
    a,ma=obs[0]; c,mc=obs[1]
    with torch.no_grad():
        physical,mean,std=actor(a)
        reflected,mmean,mstd=actor(ma)
        torch.testing.assert_close(reflected,symmetry.mirror_action_xz_plane(physical),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(mstd,std[:,symmetry.joint_index_map],rtol=2e-5,atol=2e-6)
        action=torch.randn(128,29)
        qa=q(torch.cat((c,a),-1),action)
        qm=q(torch.cat((mc,ma),-1),symmetry.mirror_action_xz_plane(action))
        for value,mirrored in zip(qa,qm):
            torch.testing.assert_close(value,mirrored,rtol=2e-5,atol=2e-6)
    env.command_manager=CommandManager(cfg.command,env,'cpu'); env.command_manager.setup()
    env.command_manager.reset(torch.arange(128))
    gait=env.command_manager.get_state('locomotion_gait')
    assert ((gait.clock.phase_period>=.9)&(gait.clock.phase_period<=1.1)).all()
    env.command_manager.commands[:]=torch.tensor([0.,.6,0.])
    env.episode_length_buf[:]=37; gait.step()
    expected=torch.remainder(env.episode_length_buf*env.dt+gait.clock.phase_offset,gait.clock.phase_period)/gait.clock.phase_period
    torch.testing.assert_close(gait.clock.phase,expected)
    torch.testing.assert_close(gait.encoding(),torch.stack((torch.sin(2*torch.pi*expected),torch.cos(2*torch.pi*expected)),-1))
    env.command_manager.commands[:]=0; gait.step()
    assert gait.encoding().count_nonzero()==0
    from holosoma.managers.command.terms.domi_gait import sin_phase,cos_phase
    for func,mirror in ((sin_phase,symmetry.mirror_obs_sin_phase),(cos_phase,symmetry.mirror_obs_cos_phase)):
        for moving in (False,True):
            env.command_manager.commands[:,1]=.6 if moving else 0
            value=func(env)
            torch.testing.assert_close(mirror(value),value.flip(-1))
    old=presets.g1_29dof_bilateral_no_right_fast_sac
    assert not old.algo.config.symmetry_compatible_io
    print('PASS: actor/critic normalization commutes with physical reflection; nonzero actor mean/std and Q equivariance; normalizer checkpoint; original domi clock; standing zero; both phase channels; old presets unchanged.')


if __name__=='__main__': main()
