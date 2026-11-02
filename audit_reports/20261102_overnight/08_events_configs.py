# TensorBoard scalar validation and cross-model configuration provenance.
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray
event_paths=sorted({p.resolve() for p in Q.rglob('events.out.tfevents.*') if p.is_file()})
event_reports=[]
for idx,p in enumerate(event_paths):
 tags={};bad=[]
 try:
  for event in EventFileLoader(str(p)).Load():
   for value in event.summary.value:
    try:
     if value.HasField('simple_value'):number=float(value.simple_value)
     elif value.HasField('tensor'):
      arr=make_ndarray(value.tensor)
      if arr.size!=1 or arr.dtype.kind not in 'biufc':continue
      number=float(arr.reshape(-1)[0])
     else:continue
    except (ValueError,TypeError,IndexError):continue
    stat=tags.setdefault(value.tag,dict(count=0,first_step=int(event.step),last_step=int(event.step),first=None,last=None,minimum=None,maximum=None))
    stat['count']+=1;stat['last_step']=int(event.step)
    if not math.isfinite(number):
     if len(bad)<30:bad.append(dict(tag=value.tag,step=int(event.step),value=str(number)))
     continue
    if stat['first'] is None:stat['first']=number
    stat['last']=number
    stat['minimum']=number if stat['minimum'] is None else min(stat['minimum'],number)
    stat['maximum']=number if stat['maximum'] is None else max(stat['maximum'],number)
  event_reports.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,tags=tags,nonfinite_examples=bad))
 except Exception as e:event_reports.append(dict(path=str(p),error=repr(e)))
 if (idx+1)%20==0:print('EVENT_PROGRESS',idx+1,'/',len(event_paths),flush=True)
write_json('tensorboard_validation.json',event_reports)
# Verify terminal evidence copy and native summary counters.
evidence=Q/'corrected_v2_20261029_0513/evidence/MLP_training_terminal.txt'
evidence_digest=hashlib.sha256(evidence.read_bytes()).hexdigest()
matches=[]
for meta in json.loads((O/'log_inventory.json').read_text()):
 p=R/meta['path']
 if p!=evidence and p.stat().st_size==evidence.stat().st_size and hashlib.sha256(p.read_bytes()).hexdigest()==evidence_digest:matches.append(str(p.relative_to(R)))
write_json('terminal_evidence_copy.json',dict(evidence=str(evidence.relative_to(R)),sha256=evidence_digest,byte_identical_original_logs=matches,
 limitation='The old v2 terminal/config proves configured sampler selection, not every historically consumed command sample.'))
# Original PPO archives use different checkpoint keys; empty actor digest is not an unchanged policy.
for rec in checkpoint_reports:
 if rec.get('actor_state_numel')==0:rec['actor_state_digest']=None;rec['actor_digest_applicable']=False
 else:rec['actor_digest_applicable']=True
write_json('checkpoint_validation.json',checkpoint_reports)
for rec in summaries:
 entries=[x for x in checkpoint_reports if str(Path(x['path']).parent)==rec['directory']]
 valid=[x['actor_state_digest'] for x in entries if x.get('actor_state_digest')]
 rec['actor_digests_unique']=len(set(valid)) if valid else None
write_json('checkpoint_group_summary.json',summaries)

training_models={}
v2_manifest=json.loads((cohorts['v2']/'manifest.json').read_text())
for model,location in v2_manifest['models'].items():
 p=next(Path(location).rglob('holosoma_config.yaml'));training_models[model]=yaml.safe_load(p.read_text())
native_cfg=yaml.safe_load((native_run/'holosoma_config.yaml').read_text())
def flatten(value,prefix=''):
 out={}
 if isinstance(value,dict):
  for key,v in value.items():out.update(flatten(v,prefix+'.'+key if prefix else key))
 else:out[prefix]=value
 return out
sections=('robot','simulator','terrain','randomization','reward','termination','action','observation','command','curriculum')
mdp_diffs={}
for model,cfg in training_models.items():
 diffs=[]
 for section in sections:
  a=flatten(training_models['biHAM'].get(section),section);b=flatten(cfg.get(section),section)
  diffs.extend(dict(field=k,biHAM=a.get(k),other=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k))
 mdp_diffs[model]=diffs
native_diffs=[]
for section in sections:
 a=flatten(training_models['MLP'].get(section),section);b=flatten(native_cfg.get(section),section)
 native_diffs.extend(dict(field=k,v2_MLP=a.get(k),native_MLP=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k))
optimizer_details=[]
for model in ('biHAM','sym_aug_HAM','HAM','MLP'):
 cp=next(Path(v2_manifest['models'][model]).rglob('model_0010000.pt'))
 state=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
 groups={}
 for key,val in state.items():
  if 'optimizer' in key and isinstance(val,dict) and 'param_groups' in val:
   groups[key]=[{k:plain(v) for k,v in g.items() if k!='params'} for g in val['param_groups']]
 optimizer_details.append(dict(model=model,optimizers=groups))
 del state
write_json('training_configuration_comparison.json',dict(v2_mdp_differences_from_biHAM=mdp_diffs,
 native_vs_v2_MLP_mdp_differences=native_diffs,optimizer_details=optimizer_details,
 fresh_start=[dict(model=m,checkpoint=c.get('training',{}).get('checkpoint'),teacher_checkpoint=c.get('training',{}).get('teacher_checkpoint')) for m,c in training_models.items()]+[dict(model='native_MLP',checkpoint=native_cfg['training'].get('checkpoint'),teacher_checkpoint=native_cfg['training'].get('teacher_checkpoint'))]))
# Store compact error classes rather than treating all logged graphics errors as training failures.
alerts=json.loads((O/'canonical_log_alerts.json').read_text())
write_json('selected_log_error_examples.json',[dict(path=r['path'],counts=r['severity_counts'],examples=r['examples']) for r in alerts if 'training_terminal' in r['path'] or 'supervisor' in r['path'] or '_0001000_' in r['path'] or 'highlight_MLP.log' in r['path']])
black=[r['path'] for r in video_reports if any(f.get('near_black_fraction',0)>.995 for f in r.get('frames_checked',[]))]
write_json('black_scene_suspects.json',black)
print('EVENTS',len(event_reports),'read errors',sum('error' in e for e in event_reports),'nonfinite scalar examples',sum(len(e.get('nonfinite_examples',[])) for e in event_reports))
print('V2_MDP_DIFFERENCES',{k:len(v) for k,v in mdp_diffs.items()})
print('NATIVE_VS_V2_MDP_DIFFERENCE_COUNT',len(native_diffs))
print('FRESH_START',[(m,c.get('training',{}).get('checkpoint'),c.get('training',{}).get('teacher_checkpoint')) for m,c in training_models.items()])
print('TERMINAL_ORIGINAL_MATCHES',matches)
print('BLACK_SCENE_SUSPECTS',black)
print('HIGHLIGHTS',[(r['cohort'],r['selection'].get('iteration'),r['selection'].get('case'),r['index_mentions_final_checkpoint']) for r in presentation])
