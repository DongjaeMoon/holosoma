# Architecture, reflection and autograd diagnostics. No optimizer steps.
from types import SimpleNamespace as NS
import copy
from holosoma.agents.fast_sac.fast_sac import Actor,Critic
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.bilateral.fast_sac_adapter import build_bilateral_sac
from holosoma.agents.modules.bilateral.ham_adapter import build_ham_sac
from holosoma.agents.modules.bilateral.normalization import DomiSymmetry
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.utils.eval_utils import _experiment_config_from_saved
from holosoma.agents.fast_sac.bilateral_fast_sac import BilateralFastSAC
dims=dict(actions=29,dof_pos=29,dof_vel=29,base_ang_vel=3,base_lin_vel=3,command_lin_vel=2,command_ang_vel=1,projected_gravity=3,sin_phase=2,cos_phase=2)
class AuditObservationManager:
 def __init__(self,cfg):self.cfg=cfg
 def _compute_term(self,group,name,term):return torch.zeros(1,dims[name])
 def term_shape(self,group,name):return (dims[name],)
 def get_observation_layout(self,keys):
  return [dict(group=k,name=n,dim=dims[n],history_length=1,scale=self.cfg.groups[k].terms[n].scale) for k in keys for n in sorted(self.cfg.groups[k].terms)]
def build_saved(cohort,model,iteration=10000):
 folder=cohorts[cohort];manifest=json.loads((folder/'manifest.json').read_text())
 if cohort=='native':checkpoint=next((Q/'vanilla_mlp_30k_native_io_gpu2/training').rglob(f'model_{iteration:07d}.pt'))
 else:
  location=Path(manifest['models'][model])
  if not location.is_absolute():location=Q/location
  checkpoint=next(location.rglob(f'model_{iteration:07d}.pt'))
 state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
 cfg=_experiment_config_from_saved(state['experiment_config']);ac=cfg.algo.config
 env=NS(robot_config=cfg.robot,observation_manager=AuditObservationManager(cfg.observation),device='cpu',history_length={'actor_obs':1,'critic_obs':1})
 na=state['obs_normalizer_state']['_mean'].numel();nc=state['critic_obs_normalizer_state']['_mean'].numel()
 index=lambda key,n:{key:dict(start=0,end=n,size=n)}
 actor=Actor(index('actor_obs',na),['actor_obs'],29,1,ac.actor_hidden_dim,ac.log_std_max,ac.log_std_min,use_tanh=ac.use_tanh,use_layer_norm=ac.use_layer_norm,device='cpu')
 critic=Critic(index('critic_obs',nc),['critic_obs'],29,ac.num_atoms,ac.v_min,ac.v_max,ac.critic_hidden_dim,device='cpu')
 router=None
 if model=='biHAM':actor,critic,router=build_bilateral_sac(env,ac,actor,critic)
 elif model in ('HAM','sym_aug_HAM'):actor,critic,router=build_ham_sac(env,ac,actor,critic)
 actor.load_state_dict(state['actor_state_dict'],strict=True)
 critic.load_state_dict(state['qnet_state_dict'],strict=True)
 actor.eval();critic.eval()
 norm=EmpiricalNormalization(na,'cpu');norm.load_state_dict(state['obs_normalizer_state']);norm.eval()
 phase_function=cfg.observation.groups['actor_obs'].terms['cos_phase'].func
 symmetry=DomiSymmetry(env) if 'domi_gait:' in phase_function else SymmetryUtils(env)
 return checkpoint,state,cfg,env,actor,critic,router,norm,symmetry
torch.manual_seed(43)
architecture_reports=[]
selections=[('v2',m,10000) for m in ('biHAM','sym_aug_HAM','HAM','MLP')]+[('native','MLP',10000),('native','MLP',30000),('legacy_v1','biHAM',10000),('legacy_v1','MLP',10000)]
for cohort,model,iteration in selections:
 try:
  checkpoint,state,cfg,env,actor,critic,router,norm,symmetry=build_saved(cohort,model,iteration)
  pool=[v for key,v in initial_records.items() if key[:3]==(cohort,model,iteration)]
  x=torch.tensor([obs for v in pool for obs in v['actor_obs']])
  x=x[::max(1,len(x)//96)]
  reflect=lambda a:symmetry.mirror_xz_plane(a,env,cfg.algo.config.actor_obs_keys)
  reflect_action=symmetry.mirror_action_xz_plane
  stats=lambda a:dict(rms=float(a.square().mean().sqrt()),max_abs=float(a.abs().max()))
  norm_before={k:v.clone() for k,v in norm.state_dict().items()}
  with torch.no_grad():
   rx=reflect(x)
   involution=stats(reflect(rx)-x)
   commutation=stats(norm(rx,update=False)-reflect(norm(x,update=False)))
   raw_eq=stats(actor(rx)[0]-reflect_action(actor(x)[0]))
   policy_eq=stats(actor(norm(rx,update=False))[0]-reflect_action(actor(norm(x,update=False))[0]))
   mean,var=norm._mean,norm._var
   projected_mean=.5*(mean+reflect(mean))
   projected_var=.5*(var+reflect(var).abs())+.25*(mean-reflect(mean)).square()
   project=lambda a:(a-projected_mean)/(projected_var.sqrt()+norm.eps)
   projected_eq=stats(actor(project(rx))[0]-reflect_action(actor(project(x))[0]))
   sweep=[];actions={}
   for vy in (-.9,-.6,-.3,0.,.3,.6,.9):
    value=x.clone();value[:,[33,34,32]]=torch.tensor([0.,vy,0.])
    normalized=norm(value,update=False);actions[vy]=actor(normalized)[0]
    sweep.append(dict(raw_vy=vy,z_vy=float(normalized[0,34]),action_rms=float(actions[vy].square().mean().sqrt())))
   for row in sweep:
    delta=actions[row['raw_vy']]-actions[0.]
    row['action_change_from_zero_rms']=float(delta.square().mean().sqrt())
  actor_ids={id(p) for p in actor.parameters()};critic_ids={id(p) for p in critic.parameters()}
  router_ids={id(p) for p in router.parameters()} if router is not None else set()
  shared_ok=(actor_ids&critic_ids)==router_ids
  assert shared_ok
  target=copy.deepcopy(critic)
  assert not ({id(p) for p in target.parameters()}&(actor_ids|critic_ids))
  del target
  actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
  batch=norm(x[:4],update=False)
  actor(batch)[0].square().mean().backward()
  actor_active=sum(p.numel() for p in actor.parameters() if p.grad is not None)
  actor_router_grad=float(sum(p.grad.square().sum() for p in router.parameters() if p.grad is not None).sqrt()) if router is not None else None
  critic_router_grad=None;policy_router_grad=None;restored=True
  if router is not None:
   actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
   nc=state['critic_obs_normalizer_state']['_mean'].numel()
   critic_input=torch.cat((torch.randn(4,nc),batch.detach()),-1)
   critic(critic_input,torch.randn(4,29)).square().mean().backward()
   critic_router_grad=float(sum(p.grad.square().sum() for p in router.parameters() if p.grad is not None).sqrt())
   actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
   candidate=actor(batch)[0]
   flags=[p.requires_grad for p in critic.parameters()]
   frozen_logits=BilateralFastSAC._policy_q(NS(qnet=critic),critic_input,candidate)
   frozen_logits.square().mean().backward()
   policy_router_grad=float(sum(p.grad.square().sum() for p in router.parameters() if p.grad is not None).sqrt())
   restored=all(p.requires_grad==f for p,f in zip(critic.parameters(),flags))
   assert restored
  rec=dict(cohort=cohort,model=model,iteration=iteration,checkpoint=str(checkpoint.relative_to(R)),
    physical_reflection_type=type(symmetry).__name__,phase_function=cfg.observation.groups['actor_obs'].terms['cos_phase'].func,
    actor_type=type(actor).__name__,actor_feature_type=type(actor.net).__name__,
    actor_parameters=sum(p.numel() for p in actor.parameters()),
    actor_active_gradient_parameters=actor_active,critic_parameters=sum(p.numel() for p in critic.parameters()),
    router_parameters=sum(p.numel() for p in router.parameters()) if router is not None else 0,
    parameter_sharing_exact=shared_ok,target_router_separate=True,
    actor_router_gradient_norm=actor_router_grad,critic_router_gradient_norm=critic_router_grad,
    actor_loss_through_frozen_q_router_gradient_norm=policy_router_grad,requires_grad_restored=restored,
    mirror_involution=involution,normalizer_commutation=commutation,raw_actor_reflection_error=raw_eq,
    full_policy_reflection_error=policy_eq,projected_normalizer_policy_reflection_error=projected_eq,
    normalized_stats_frozen=all(torch.equal(norm_before[k],v) for k,v in norm.state_dict().items()),
    phase_pair_sum_max=float(torch.cat((x[:,35:37].sum(-1),x[:,98:100].sum(-1))).abs().max()),
    command_sweep=sweep,actor_base_linear_velocity_observation=False)
  architecture_reports.append(rec)
  print('ARCHITECTURE',cohort,model,iteration,'params',rec['actor_parameters'],'active',actor_active,'eq_rms',policy_eq['rms'],'router_grad',policy_router_grad,flush=True)
  write_json('architecture_validation.json',architecture_reports)
  del state,actor,critic,router,norm,x;gc.collect()
 except Exception as e:
  architecture_reports.append(dict(cohort=cohort,model=model,iteration=iteration,error=repr(e)))
  write_json('architecture_validation.json',architecture_reports)
  print('ARCHITECTURE_ERROR',cohort,model,iteration,repr(e),flush=True)
