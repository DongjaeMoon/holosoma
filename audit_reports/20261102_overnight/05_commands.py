# Actual command exposure and source sampler checks, all on CPU.
import csv
from holosoma.managers.command.terms.three_direction import ThreeDirectionCommand
native_root=Q/'vanilla_mlp_30k_native_io_gpu2'
native_run=next((native_root/'training').glob('*/*/commands_policy.csv')).parent
with (native_run/'commands_policy.csv').open() as stream:csv_rows=list(csv.DictReader(stream))
columns={key:np.asarray([float(r[key]) for r in csv_rows]) for key in ('step','env_id','vx','vy','yaw','manager_vx','manager_vy','manager_yaw','normalized_vx','normalized_vy','normalized_yaw','body_vx','body_vy','body_yaw_rate')}
status=json.loads((native_run/'command_audit_status.json').read_text())
assert len(csv_rows)==30001 and np.array_equal(columns['step'],np.arange(30001))
assert all(np.array_equal(columns[k],columns['manager_'+k]) for k in ('vx','vy','yaw'))
assert (columns['vy']>=0).all()
audit_counts=dict(csv_rows=len(csv_rows),csv_env_ids=np.unique(columns['env_id']).tolist(),
    policy_inputs=status['policy_command_samples'],policy_inputs_per_iteration=status['policy_command_samples']/30001,
    replay_current_plus_next_samples=status['replay_command_samples'],
    replay_update_blocks=status['replay_command_samples']/(2*8192*8),
    policy_negative=status['negative_policy_vy_count'],replay_negative=status['negative_replay_vy_count'],
    all_environment_direction_ratios={k:v/status['policy_command_samples'] for k,v in status['command_counts'].items()},
    env0={},zero_command_negative_normalized_vy=float(np.mean(columns['normalized_vy'][columns['vy']==0]<0)),
    physical_motion_scope='env 0 only; per-timestep motion signs are not issued-command signs')
for k in ('vx','vy','yaw'):
 audit_counts['env0'][k]=dict(raw_min=float(columns[k].min()),raw_max=float(columns[k].max()),
    normalized_min=float(columns['normalized_'+k].min()),normalized_max=float(columns['normalized_'+k].max()),
    normalized_negative_fraction=float(np.mean(columns['normalized_'+k]<0)))
for label,mask in [('all',np.ones(len(csv_rows),bool)),('left_command',columns['vy']>0),('zero_lateral_command',columns['vy']==0)]:
 audit_counts[label+'_motion']=dict(rows=int(mask.sum()),body_vy_negative_fraction=float(np.mean(columns['body_vy'][mask]<-.001)),
    body_vy_min=float(columns['body_vy'][mask].min()),body_vy_max=float(columns['body_vy'][mask].max()))
# Check first logged normalization against a fresh native normalizer.
first=csv_rows[0]
audit_counts['initial_normalizer_default_max_error']=max(abs(float(first['normalized_'+k])-float(first[k])/1.01) for k in ('vx','vy','yaw'))
write_json('native_command_exposure.json',audit_counts)

# Replay source code's exact sampler with a large CPU-only environment array.
N=200000
sampler=object.__new__(ThreeDirectionCommand)
sampler.env=NS(device='cpu')
sampler.commands=torch.zeros(N,3)
sampler.command_ranges=dict(lin_vel_x=(-1.,1.),lin_vel_y=(-1.,1.),ang_vel_yaw=(-1.,1.))
sampler.stand_prob=.2
transient={}
def observe_frequency(ids):
 transient['negative_parent_draws_before_override']=int((sampler.commands[ids,1]<0).sum())
sampler.manager=NS(get_state=lambda name:NS(resample_frequency=observe_frequency) if name=='locomotion_gait' else None)
torch.manual_seed(43);sampler._resample(torch.arange(N))
c=sampler.commands.numpy()
categories=dict(forward=(c[:,0]>0)&(c[:,1]==0),backward=(c[:,0]<0)&(c[:,1]==0),
    left=(c[:,1]>0)&(c[:,0]==0),right=(c[:,1]<0),standing=(c==0).all(-1))
sampler_check=dict(samples=N,final_negative_vy=int((c[:,1]<0).sum()),both_axes_nonzero=int(((c[:,0]!=0)&(c[:,1]!=0)).sum()),
    ratios={k:float(v.mean()) for k,v in categories.items()},yaw_range=[float(c[:,2].min()),float(c[:,2].max())],transient=transient,
    theoretical_no_right_moments=dict(vx_mean=0.,vx_variance=8/45,vy_mean=2/15,vy_variance=16/225,yaw_mean=0.,yaw_variance=4/15),
    explanation='Parent sampler draws negative vy temporarily, then ThreeDirectionCommand overwrites it before policy consumption. The frequency callback is inspected separately.')
assert sampler_check['final_negative_vy']==0 and sampler_check['both_axes_nonzero']==0
write_json('sampler_cpu_validation.json',sampler_check)
# Existing scheduled command changes provide observational closed-loop evidence.
switch=native_root/'video_preview_10k/switch/commands.csv'
with switch.open() as stream:switch_rows=list(csv.DictReader(stream))
segments=[('stand_start',0,3),('forward',3,8),('backward',8,13),('left',13,19),('right',19,25),('stand_end',25,28)]
switch_summary=[]
for name,start,end in segments:
 stable=[r for r in switch_rows if start+min(2.,(end-start)/2)<=float(r['time_s'])<end]
 switch_summary.append(dict(segment=name,start=start,end=end,stable_rows=len(stable),
  mean_target=[float(stable[0][k]) for k in ('command_vx','command_vy','command_yaw')],
  mean_actual=[sum(float(r[k]) for r in stable)/len(stable) for k in ('actual_vx','actual_vy','actual_yaw')]))
write_json('existing_command_switch_evidence.json',dict(segments=switch_summary,limitation='Sequential states differ; this is not a matched-initial-state command-clamp ablation.'))
print('EXPOSURE',audit_counts)
print('SAMPLER',sampler_check)
print('EXISTING_SWITCH',switch_summary)
