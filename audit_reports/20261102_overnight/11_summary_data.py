# Consolidated factual tables, source locations and one unused-configuration replay test.
from unittest.mock import patch
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer
rbtest=SimpleReplayBuffer(1,4,1,1,1,n_steps=3,gamma=.9,device='cpu')
rbtest.ptr=4
rbtest.rewards[0]=torch.tensor([1.,2.,3.,4.])
rbtest.next_observations[0,:,0]=torch.tensor([1.,2.,3.,4.])
with patch.object(torch,'randint',return_value=torch.tensor([[2]])):
 result=rbtest.sample(1)
replay_finding=dict(configuration_num_steps=3,buffer_size=4,ptr=4,forced_start_index=2,expected_reward=3+.9*4,observed_reward=float(result['next']['rewards'][0]),expected_effective_steps=2,observed_effective_steps=float(result['next']['effective_n_steps'][0]),bootstrap_observation=float(result['next']['observations'][0,0]),applies_to_current_experiments=False,current_num_steps=ncfg.algo.config.num_steps,reason='Synthetic truncation at newest ring-buffer entry stops next_obs but does not stop reward accumulation or effective_n_steps.')
assert abs(replay_finding['observed_reward']-7.41)<1e-5 and replay_finding['current_num_steps']==1
write_json('unused_nstep_replay_finding.json',replay_finding)
orig_files=['src/holosoma/holosoma/train_agent.py','src/holosoma/holosoma/agents/fast_sac/fast_sac.py','src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py','src/holosoma/holosoma/managers/command/terms/locomotion.py','src/holosoma/holosoma/managers/observation/terms/locomotion.py','src/holosoma/holosoma/managers/observation/manager.py','src/holosoma/holosoma/envs/locomotion/locomotion_manager.py']
original_parity=[dict(path=p,baseline_commit='b739a6c',byte_identical=git('show','b739a6c:'+p)==(R/p).read_text()) for p in orig_files]
write_json('original_baseline_parity.json',original_parity)
references={}
ref_files=orig_files+['scripts/evaluate_command_holdout.py','src/holosoma/holosoma/managers/command/terms/three_direction.py','src/holosoma/holosoma/agents/modules/bilateral/normalization.py','src/holosoma/holosoma/agents/fast_sac/command_audit.py','src/holosoma/holosoma/utils/eval_utils.py','src/holosoma/holosoma/agents/fast_sac/bilateral_fast_sac.py','src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py','src/holosoma/holosoma/managers/command/terms/domi_gait.py','src/holosoma/holosoma/simulator/isaacsim/isaacsim.py','src/holosoma/holosoma/config_values/loco/g1/experiment.py']
for p in ref_files:
 tree=ast.parse((R/p).read_text())
 locations={}
 def visit_functions(node,prefix=''):
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    qualified=prefix+n.name;locations[qualified]=n.lineno;visit_functions(n,qualified+'.')
   else:visit_functions(n,prefix)
 visit_functions(tree)
 references[p]=locations
write_json('source_function_locations.json',references)
rows=[r for r in metric_rows if r['cohort']=='v2' and r['iteration']==10000 and r['direction'] in ('left','right')]
rows += [r for r in metric_rows if r['cohort']=='native' and r['iteration'] in (10000,30000) and r['direction'] in ('left','right')]
write_json('comparison_lateral_current.json',rows)
with (O/'comparison_lateral_current.csv').open('w',newline='') as stream:
 fields=['cohort','model','iteration','direction','speed','rmse','vx_rmse','vy_rmse','yaw_rmse','mean_vx','mean_vy','successes','trials','failure_rate','bias_fraction']
 writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
 for r in rows:
  row={k:r[k] for k in fields if k in r};row.update(mean_vx=r['mean_velocity'][0],mean_vy=r['mean_velocity'][1]);writer.writerow(row)
ar=json.loads((O/'architecture_validation.json').read_text())
events=json.loads((O/'tensorboard_validation.json').read_text())
video=json.loads((O/'video_integrity.json').read_text())
stats=dict(checkpoints=len(checkpoints),checkpoint_bytes=sum(p.stat().st_size for p in checkpoints),configuration_files=len(config_inventory),terminal_log_files=len(log_inventory),metric_cases=len(metric_rows),metric_trials=sum(r['trials'] for r in metric_rows),syntax_parsed_python_files=sum(str(p).endswith('.py') for p in baseline_hashes),source_files_hashed=len(baseline_hashes),tensorboard_event_files=len(events),videos=len(video),architecture_checkpoints=len(ar),cpu_regression_scripts=3,raw_initial_observations=sum(r['samples'] for r in obs_validation))
write_json('audit_counts.json',stats)
write_json('historical_training_argv.json',dict(native=json.loads((native_run/'command_audit_metadata.json').read_text())['argv'],v2_MLP=em['actual_training_command'],scope='Recorded historical argv; do not infer old runs from current runner defaults.'))
print('AUDIT_COUNTS',stats,flush=True)
print('LATERAL_TABLE',[(r['cohort'],r['model'],r['iteration'],r['direction'],r['speed'],round(r['rmse'],6),[round(x,6) for x in r['mean_velocity']],r['successes']) for r in rows],flush=True)
