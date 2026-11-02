# Recompute every canonical metric from per-trial data, without simulation.
import numpy as np
cohorts={
 'v2':Q/'corrected_v2_20261029_0513/evaluation',
 'native':Q/'vanilla_mlp_30k_native_io_gpu2/evaluation',
 'legacy_v1':Q/'deploy_eval_four_way_10k',
}
metric_checks=[];metric_rows=[];initial_records={};metric_sources={}
for cohort,folder in cohorts.items():
 if not folder.exists():continue
 manifest=json.loads((folder/'manifest.json').read_text())
 protocol=manifest['protocol']
 models=list(manifest.get('models',{'MLP':None}))
 if cohort=='native':models=['MLP']
 for model in models:
  for path in sorted((folder/'videos'/model).glob('model_*/*/metrics.json')):
   row=json.loads(path.read_text());trials=row.get('per_trial',[])
   problems=[];valid=[t for t in trials if t.get('linear_rmse') is not None]
   if len(trials)!=protocol['trials']:problems.append('trial_count')
   if not valid:continue
   avg=lambda key:sum(float(t[key]) for t in valid)/len(valid)
   recalculated=avg('linear_rmse')
   if abs(recalculated-row['mean_linear_rmse'])>1e-8:problems.append('mean_linear_rmse')
   successes=0;decomposed=[]
   observations=[];physical=[]
   for t in trials:
    if t.get('linear_rmse') is not None:
     if abs(t['linear_rmse']**2-t['vx_rmse']**2-t['vy_rmse']**2)>1e-5:problems.append('axis_rmse_identity')
     count=t['measured_steps']
     if count<=0:problems.append('nonpositive_measurement_count')
     success=not t['failure'] and not t.get('early_timeout',False) and t['survived_s']>=protocol['duration_s']-.021 and t['linear_rmse']<=protocol['linear_rmse_limit'] and t['yaw_rmse']<=protocol['yaw_rmse_limit']
     successes+=int(success)
     if bool(t['success'])!=bool(success):problems.append('trial_success_rule')
     bias=sum((float(t['mean_velocity'][k])-row['command'][k])**2 for k in (0,1))
     variability=t['linear_rmse']**2-bias
     if variability< -1e-5:problems.append('negative_velocity_variance')
     decomposed.append((bias,max(0.,variability),t['linear_rmse']**2))
    state=t.get('initial_state')
    if state:
     obs=np.asarray(state['actor_obs'],dtype=np.float64)
     command=obs[[33,34,32]]
     if not np.allclose(command,row['command'],atol=2e-7,rtol=0):problems.append('initial_actor_command_mismatch')
     observations.append(obs.tolist())
     phys=np.concatenate((np.asarray(state['root']).reshape(-1),np.asarray(state['dof']).reshape(-1)))
     physical.append(phys.tolist())
   if successes!=row['successes']:problems.append('success_count')
   initial_records[(cohort,model,row['iteration'],row['name'])]=dict(actor_obs=observations,physical=physical)
   unique6=len({tuple(np.round(p,6)) for p in physical})
   unique3=len({tuple(np.round(p,3)) for p in physical})
   mean_v=[sum(t['mean_velocity'][k] for t in valid)/len(valid) for k in (0,1)]
   decomp=np.asarray(decomposed).mean(0)
   rec=dict(cohort=cohort,model=model,iteration=row['iteration'],name=row['name'],direction=row['direction'],speed=row['speed'],
      command=row['command'],rmse=recalculated,rmse_sd=float(np.std([t['linear_rmse'] for t in valid],ddof=1)) if len(valid)>1 else None,
      vx_rmse=avg('vx_rmse'),vy_rmse=avg('vy_rmse'),yaw_rmse=avg('yaw_rmse'),mean_velocity=mean_v,
      bias_mse=float(decomp[0]),velocity_variability_mse=float(decomp[1]),mean_mse=float(decomp[2]),
      bias_fraction=float(decomp[0]/decomp[2]) if decomp[2]>0 else 0.,
      trials=len(trials),measured_trials=len(valid),successes=successes,failure_rate=row['failure_rate'],mean_survival_s=row['mean_survival_s'],
      measured_steps_min=min(t['measured_steps'] for t in valid),measured_steps_max=max(t['measured_steps'] for t in valid),
      initial_physical_unique_round6=unique6,initial_physical_unique_round3=unique3,
      source=str(path.relative_to(R)),checkpoint=row.get('checkpoint'))
   metric_rows.append(rec);metric_sources[(cohort,model,row['iteration'],row['name'])]=row
   metric_checks.append(dict(source=str(path.relative_to(R)),issues=sorted(set(problems))))
write_json('metric_recomputation.json',metric_rows)
write_json('metric_consistency_checks.json',metric_checks)
write_json('metric_recomputation_issues.json',[x for x in metric_checks if x['issues']])
# Re-check summary tables against the detailed files (summary often omits per_trial).
summary_checks=[]
for cohort,folder in cohorts.items():
 p=folder/'summary.json'
 if not p.exists():continue
 summary=json.loads(p.read_text());bad=[];checked=0
 for row in summary:
  key=(cohort,row['model'],row['iteration'],row['name'])
  if key not in metric_sources:bad.append(dict(key=list(key),issue='missing detailed metric'));continue
  actual=metric_sources[key];checked+=1
  for name in ('mean_linear_rmse','mean_vx_rmse','mean_vy_rmse','mean_yaw_rmse','successes','failure_rate'):
   if name in row and row[name]!=actual.get(name):bad.append(dict(key=list(key),issue=name))
 summary_checks.append(dict(cohort=cohort,rows=len(summary),checked=checked,issues=bad))
write_json('summary_consistency.json',summary_checks)
# Compare initial physical states across the four v2 models.
paired=[]
for direction in ('left','right'):
 for speed in (.3,.6):
  name=f'{direction}_{speed:.1f}mps';arrays={}
  for model in ('biHAM','sym_aug_HAM','HAM','MLP'):
   arrays[model]=np.asarray(initial_records[('v2',model,10000,name)]['physical'])
  reference=arrays['biHAM']
  paired.append(dict(case=name,max_abs_difference_from_biHAM={m:float(np.abs(a-reference).max()) for m,a in arrays.items()}))
write_json('paired_initial_states_v2.json',paired)
selected=[r for r in metric_rows if r['iteration']==10000 and r['direction'] in ('left','right')]
write_json('all_models_10k_lateral.json',selected)
print('METRICS',len(metric_rows),'ISSUES',sum(bool(x['issues']) for x in metric_checks))
print('SUMMARIES',[(r['cohort'],r['rows'],len(r['issues'])) for r in summary_checks])
print('FINAL_LATERAL',[(r['cohort'],r['model'],r['name'],round(r['rmse'],5),[round(x,4) for x in r['mean_velocity']],r['successes'],round(r['bias_fraction'],3),r['initial_physical_unique_round3']) for r in selected])
print('PAIRED',paired)
