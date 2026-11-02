# Repository-wide source scan and targeted mathematical/lifecycle reproductions.
from holosoma.agents.modules.bilateral.normalization import ReflectionNormalization
from holosoma.utils.eval_utils import load_saved_experiment_config,CheckpointConfig
from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
from scipy.spatial.transform import Rotation
syntax_errors=[];source_references=collections.defaultdict(list);normalizer_locations=[]
patterns=('use_symmetry','symmetry_compatible_io','augment_observations','mirror_obs_command','command_lin_vel','command_ang_vel','torch.load(')
for relative in baseline_hashes:
 p=R/relative
 if p.suffix!='.py':continue
 try:
  text=p.read_text();tree=ast.parse(text,filename=relative)
  for node in ast.walk(tree):
   if isinstance(node,ast.ClassDef) and 'Normalization' in node.name:normalizer_locations.append(dict(file=relative,name=node.name,line=node.lineno))
  for i,line in enumerate(text.splitlines(),1):
   for pattern in patterns:
    if pattern in line:source_references[pattern].append(dict(file=relative,line=i,text=line.strip()[:250]))
 except Exception as e:syntax_errors.append(dict(file=relative,error=repr(e)))
write_json('source_scan.json',dict(syntax_errors=syntax_errors,normalizers=normalizer_locations,references=dict(source_references)))

# Verify quaternion convention numerically against an independent library.
rng=np.random.default_rng(43)
qs=rng.normal(size=(10000,4));qs/=np.linalg.norm(qs,axis=1)[:,None]
vs=rng.normal(size=(10000,3))
from holosoma.utils.rotations import quat_rotate_inverse
rotated=quat_rotate_inverse(torch.tensor(qs),torch.tensor(vs),w_last=True).numpy()
expected=Rotation.from_quat(qs).inv().apply(vs)
rotation_error=float(np.abs(rotated-expected).max())
assert rotation_error<1e-12

# Known native empirical variance-merge error.
emp=EmpiricalNormalization(1,'cpu')
emp(torch.tensor([[0.],[0.]]));emp(torch.tensor([[2.],[2.]]))
native_var=dict(samples=[0.,0.,2.,2.],expected_mean=1.,stored_mean=float(emp._mean[0,0]),expected_variance=1.,stored_variance=float(emp._var[0,0]))
# The reflection wrapper inherits the same merge problem, while remaining equivariant.
_,_,ref_cfg,ref_env,ref_actor,ref_critic,ref_router,_,ref_sym=build_saved('v2','biHAM',10000)
rn=ReflectionNormalization(100,'cpu',ref_sym,ref_cfg.algo.config.actor_obs_keys)
positive=torch.zeros(4,100);positive[:,34]=.5
rn(positive);first_variance=float(rn._var[0,34]);rn(positive);rn.eval()
reflection_var=dict(first_batch_variance=first_variance,second_batch_variance=float(rn._var[0,34]),
 expected_symmetric_second_moment=.25,mean=float(rn._mean[0,34]),actual_positive_input_normalized_vy=float(rn(positive,update=False)[0,34]))
del ref_actor,ref_critic,ref_router;gc.collect()

# Compatibility of the public configuration loader in the CURRENT CPU environment.
loader_tests=[]
for cohort,model in [('v2','biHAM'),('v2','MLP'),('native','MLP')]:
 if cohort=='native':cp=next((Q/'vanilla_mlp_30k_native_io_gpu2/training').rglob('model_0010000.pt'))
 else:
  manifest=json.loads((cohorts[cohort]/'manifest.json').read_text());cp=next(Path(manifest['models'][model]).rglob('model_0010000.pt'))
 try:
  cfg,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(cp)))
  loader_tests.append(dict(cohort=cohort,model=model,success=True))
 except Exception as e:loader_tests.append(dict(cohort=cohort,model=model,success=False,error=type(e).__name__,message=str(e)[:1200]))
write_json('current_loader_compatibility.json',loader_tests)

# Native matched-state rendering reproduction with CPU fakes honoring actual data contracts.
sys.path.insert(0,str(R/'scripts'))
from evaluate_command_holdout import render_case
from holosoma.managers.command.terms.locomotion import LocomotionGait
class RestoreSim:
 def __init__(self):
  raw=torch.zeros(1,13);raw[0,2]=.8;raw[0,3]=1.
  self.robot_root_states=RootStatesProxy(raw)
  self.dof_pos=torch.zeros(1,29);self.dof_vel=torch.zeros(1,29)
  self.video_recorder=NS(stop_recording=lambda:None)
  self.received_dof=None
 @property
 def dof_state(self):return torch.cat([self.dof_pos[...,None],self.dof_vel[...,None]],-1)
 def set_actor_root_state_tensor_robots(self,*args):pass
 def set_dof_state_tensor_robots(self,ids,state):
  self.received_dof=state.clone();self.dof_pos[:]=state[...,0];self.dof_vel[:]=state[...,1]
 def refresh_sim_tensors(self):pass
def restore_test(target_position,target_phase):
 sim=RestoreSim()
 env=NS(device='cpu',simulator=sim,num_dof=29,episode_length_buf=torch.zeros(1,dtype=torch.long),dt=.02)
 manager=NS(commands=torch.zeros(1,3))
 gait=object.__new__(LocomotionGait);gait.env=env;gait.manager=manager
 gait.phase=torch.zeros(1,2);gait.phase_offset=torch.tensor([[.5,.5-math.pi]]);gait.phase_dt=torch.ones(1,1)*.1;gait.stand_phase_value=0.
 manager.get_state=lambda name:gait
 env.command_manager=manager
 obs=NS(cfg=NS(groups={'actor_obs':NS(history_length=1)}))
 obs._compute_group=lambda *args:torch.cat((sim.dof_pos,torch.sin(gait.phase),torch.cos(gait.phase)),-1)
 obs.get_observation_layout=lambda keys:[dict(name='dof_pos',dim=29,scale=1.,history_length=1),dict(name='sin_phase',dim=2,scale=1.,history_length=1),dict(name='cos_phase',dim=2,scale=1.,history_length=1)]
 env.observation_manager=obs
 env.terrain_manager=NS(get_state=lambda name:NS(env_origins=torch.zeros(1,3)))
 env.set_is_evaluating=lambda:None
 def reset():
  sim.dof_pos.zero_();sim.dof_vel.zero_();env.episode_length_buf.zero_()
 env.reset_all=reset;env._pre_compute_observations_callback=lambda:None
 wanted_phase=torch.tensor([target_phase,target_phase-math.pi])
 initial=dict(root=[0.,0.,.8,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
  dof=[[target_position,.3 if target_position else 0.] for _ in range(29)],
  actor_obs=[target_position]*29+torch.sin(wanted_phase).tolist()+torch.cos(wanted_phase).tolist(),domi_clock=None)
 try:
  render_case(env,NS(config=NS(actor_obs_keys=['actor_obs'])),lambda x:None,
    dict(command=[0.,.6,0.]),O,dict(seed=43,video_duration_s=0.,warmup_s=0.),initial)
  outcome='unexpected success'
 except Exception as e:outcome=type(e).__name__+': '+str(e).splitlines()[0]
 return dict(target_dof_position=target_position,actual_dof_position=float(sim.dof_pos[0,0]),
   setter_received_position=float(sim.received_dof[0,0,0]) if sim.received_dof is not None else None,
   target_phase=target_phase,actual_phase=float(gait.phase[0,0]),outcome=outcome)
restore_results=[restore_test(.2,.5),restore_test(0.,1.2)]
assert restore_results[0]['actual_dof_position']==0. and restore_results[0]['outcome'].startswith('AssertionError')
assert abs(restore_results[1]['actual_phase']-.5)<1e-7 and restore_results[1]['outcome'].startswith('AssertionError')
findings=dict(rotation_scipy_max_error=rotation_error,native_variance=native_var,reflection_variance=reflection_var,
 matched_state_native_reproductions=restore_results,syntax_error_count=len(syntax_errors),
 restore_scope='Latent initial!=None branch on native IsaacSim. Latest native MLP fixed-seed videos use initial=None and are not affected.',
 snapshot_issues=json.loads((O/'native_source_provenance.json').read_text()))
write_json('confirmed_code_findings.json',findings)
print('SOURCE_SCAN',sum((R/p).suffix=='.py' for p in baseline_hashes),'syntax errors',len(syntax_errors))
print('CONFIRMED_FINDINGS',findings)
print('LOADER_TESTS',loader_tests)
