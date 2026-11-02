# Check every saved checkpoint on CPU, including optimizer tensors.
import dataclasses,gc
checkpoint_reports=[]
def plain(value):
 if dataclasses.is_dataclass(value):return plain(dataclasses.asdict(value))
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [plain(v) for v in value]
 if isinstance(value,(str,int,float,bool)) or value is None:return value
 return str(value)
def walk_tensors(value,prefix=''):
 if torch.is_tensor(value):yield prefix,value
 elif isinstance(value,dict):
  for k,v in value.items():yield from walk_tensors(v,prefix+'/'+str(k))
 elif isinstance(value,(tuple,list)):
  for k,v in enumerate(value):yield from walk_tensors(v,prefix+'/'+str(k))
def tensor_digest(tensors):
 h=hashlib.sha256()
 for k,t in tensors.items():
  if not torch.is_tensor(t):continue
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
  h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
for index,p in enumerate(checkpoints):
 rec=dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,filename_step=int(p.stem.split('_')[-1]))
 try:
  state=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
  rec['global_step']=int(state.get('global_step',state.get('iter',-1)))
  rec['step_matches_filename']=rec['global_step']==rec['filename_step']
  rec['keys']=list(state)
  bad=[];float_count=0
  for key,tensor in walk_tensors(state):
   if tensor.is_floating_point() or tensor.is_complex():
    float_count+=tensor.numel()
    if not bool(torch.isfinite(tensor).all()):bad.append(key)
  rec['floating_elements_checked']=float_count
  rec['nonfinite_tensors']=bad
  ac=plain(state.get('experiment_config',{}))
  rec['config']=dict(algo=ac.get('algo',{}).get('_target_'),algo_config=ac.get('algo',{}).get('config'),training=ac.get('training'),simulator=ac.get('simulator',{}).get('config',{}).get('name'))
  actor=state.get('actor_state_dict',{})
  rec['actor_state_digest']=tensor_digest(actor)
  rec['actor_state_numel']=sum(t.numel() for t in actor.values() if torch.is_tensor(t))
  rec['actor_keys']=list(actor)
  rec['normalizers']={}
  for key in ('obs_normalizer_state','critic_obs_normalizer_state'):
   n=state.get(key)
   if not n:continue
   m=n['_mean'].flatten();v=n['_var'].flatten();s=n['_std'].flatten()
   ids=(33,34,32) if key=='obs_normalizer_state' else (36,37,35)
   rec['normalizers'][key]=dict(dim=m.numel(),count=int(n['count']),
      min_variance=float(v.min()),std_sqrt_var_max_error=float((s-v.sqrt()).abs().max()),
      commands={name:dict(index=i,mean=float(m[i]),variance=float(v[i]),std=float(s[i]))
                for name,i in zip(('vx','vy','yaw'),ids)})
  rec['optimizer_steps']={}
  for key,value in state.items():
   if 'optimizer' in key and isinstance(value,dict) and 'state' in value:
    steps=[float(v['step']) for v in value['state'].values() if isinstance(v,dict) and 'step' in v]
    rec['optimizer_steps'][key]=dict(states=len(steps),min=min(steps) if steps else None,max=max(steps) if steps else None,unique=sorted(set(steps))[:12])
  digest=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
  rec['file_sha256']=digest.hexdigest()
  del state,actor
 except Exception as e:rec['error']=repr(e)
 checkpoint_reports.append(rec)
 if (index+1)%20==0:print('CHECKPOINT_PROGRESS',index+1,'/',len(checkpoints),flush=True)
 gc.collect()
write_json('checkpoint_validation.json',checkpoint_reports)
bad_files=[r for r in checkpoint_reports if r.get('error') or r.get('nonfinite_tensors') or not r.get('step_matches_filename')]
groups=collections.defaultdict(list)
for r in checkpoint_reports:groups[str(Path(r['path']).parent)].append(r)
summaries=[]
for group,items in groups.items():
 items=sorted(items,key=lambda r:r['filename_step'])
 summaries.append(dict(directory=group,count=len(items),first=items[0]['filename_step'],last=items[-1]['filename_step'],
    actor_digests_unique=len({r.get('actor_state_digest') for r in items}),
    algo=items[-1].get('config',{}).get('algo'),last_optimizer_steps=items[-1].get('optimizer_steps'),
    bad_files=[r['path'] for r in items if r in bad_files]))
write_json('checkpoint_group_summary.json',summaries)
print('CHECKPOINT_VALIDATED',len(checkpoint_reports),'problematic',len(bad_files))
print('CHECKPOINT_PROBLEMS',[(r['path'],r.get('error'),r.get('nonfinite_tensors'),r.get('step_matches_filename')) for r in bad_files])
print('GROUPS',[(s['directory'],s['count'],s['first'],s['last'],s['actor_digests_unique']) for s in summaries])
