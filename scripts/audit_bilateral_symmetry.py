"""CPU-only audit of saved biHAM normalization and reflection equivariance."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

from holosoma.utils.safe_torch_import import torch
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.agents.fast_sac.fast_sac import Actor, Critic
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.bilateral.fast_sac_adapter import build_bilateral_sac
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.bilateral.original.loco_io import _mirror_joint_values


def audit(checkpoint, evaluation):
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES=.'
    torch.set_num_threads(1)
    torch.manual_seed(43)
    cfg,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    ac=cfg.algo.config
    dims=dict(actions=29,dof_pos=29,dof_vel=29,base_ang_vel=3,base_lin_vel=3,
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
    env=NS(robot_config=cfg.robot,observation_manager=Manager(),device='cpu',history_length={'actor_obs':1,'critic_obs':1})
    size=lambda keys:sum(t['dim'] for t in env.observation_manager.get_observation_layout(keys))
    na,nc=size(ac.actor_obs_keys),size(ac.critic_obs_keys)
    indices=lambda key,n:{key:dict(start=0,end=n,size=n)}
    native=Actor(indices('actor_obs',na),['actor_obs'],29,1,ac.actor_hidden_dim,ac.log_std_max,ac.log_std_min,device='cpu')
    critic=Critic(indices('critic_obs',nc),['critic_obs'],29,ac.num_atoms,ac.v_min,ac.v_max,ac.critic_hidden_dim,device='cpu')
    actor,_,_=build_bilateral_sac(env,ac,native,critic)
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    actor.load_state_dict(state['actor_state_dict'])
    actor.eval()
    norm=EmpiricalNormalization(na,'cpu')
    norm.load_state_dict(state['obs_normalizer_state'])
    norm.eval()
    if getattr(ac, 'symmetry_compatible_io', False):
        from holosoma.agents.modules.bilateral.normalization import DomiSymmetry
        symmetry = DomiSymmetry(env)
    else:
        symmetry=SymmetryUtils(env)
    reflect=lambda x:symmetry.mirror_xz_plane(x,env,ac.actor_obs_keys)
    reflect_action=symmetry.mirror_action_xz_plane
    samples=[]
    for path in sorted((evaluation/'videos/biHAM'/checkpoint.stem).glob('*/metrics.json')):
        samples.extend(v['initial_state']['actor_obs'] for v in json.loads(path.read_text())['per_trial'])
    assert samples, 'Need saved evaluation observations to audit the actual policy inputs.'
    x=torch.tensor(samples)
    stats=lambda z:dict(rms=float(z.square().mean().sqrt()),max_abs=float(z.abs().max()))
    normalizer_error=norm(reflect(x),update=False)-reflect(norm(x,update=False))
    terms=symmetry.sub_observation_indices_single_frame['actor_obs']
    term_errors={k:stats(normalizer_error[:,ids]) for k,ids in terms.items()}
    # Controlled diagnostic only: project mean/variance onto the reflection-
    # compatible subspace. This does NOT edit the saved normalizer or checkpoint.
    mean=norm._mean.detach()
    var=norm._var.detach()
    sym_mean=.5*(mean+reflect(mean))
    sym_var=.5*(var+reflect(var).abs())+.25*(mean-reflect(mean)).square()
    sym_normalize=lambda value:(value-sym_mean)/(sym_var.sqrt()+norm.eps)
    def policy_error(normalize):
        with torch.no_grad():
            a=actor(normalize(x))[0]
            mirrored=actor(normalize(reflect(x)))[0]
            return stats(mirrored-reflect_action(a))
    with torch.no_grad():
        features=actor.net
        left,right=features.view(x)
        ml,mr=features.view(reflect(x))
        context=torch.cat(features.context(x),-1)
        swapped=torch.cat(context.chunk(2,-1)[::-1],-1)
        a,b=features.bank(context,left,right)
        ma,mb=features.bank(swapped,right,left)
        # Mean flips physical signs; log-std swaps joints without sign reversal.
        unscaled=actor(x)[1]
        reflected_unscaled=actor(reflect(x))[1]
        output=dict(
            checkpoint=str(checkpoint),global_step=state['global_step'],samples=len(samples),
            policy='deterministic mean',units='Holosoma actor action units before environment action_scale',
            normalizer_commutation=stats(normalizer_error),normalizer_by_term=term_errors,
            native_view_swap_error=stats(torch.cat((ml-right,mr-left),-1)),
            core_bank_swap_error=stats(torch.cat((ma-b,mb-a),-1)),
            raw_mean_reflection_error=stats(reflected_unscaled-reflect_action(unscaled)),
            full_policy_current_normalizer=policy_error(lambda value:norm(value,update=False)),
            full_policy_without_normalizer=policy_error(lambda value:value),
            full_policy_projected_normalizer=policy_error(sym_normalize),
            action_scale_pair_error=float((actor.action_scale-actor.action_scale[symmetry.joint_index_map]).abs().max()),
            domi_vs_native_joint_sign_error=float((_mirror_joint_values(torch.ones(29),cfg.robot.dof_names)-symmetry.sign_flip_mask).abs().max()),
            router_optimizer_steps=sorted(set(float(v['step']) for v in state['router_optimizer_state_dict']['state'].values())),
            normalization_command_y=dict(mean=float(mean[0,terms['command_lin_vel'][1]]),std=float(norm._std[0,terms['command_lin_vel'][1]])),
            caution='CPU algebraic diagnostic; projected normalization is not a locomotion evaluation or a performance fix.')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.checkpoint,args.evaluation)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
