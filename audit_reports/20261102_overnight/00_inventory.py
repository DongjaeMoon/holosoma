# This file is executed only in the dedicated CPU audit session.
import yaml,torch
torch.set_num_threads(1)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
Q=R/'logs/bilateral_queue'
checkpoints=sorted(Q.rglob('model_*.pt'))
checkpoint_inventory=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,mtime=p.stat().st_mtime,symlink=p.is_symlink()) for p in checkpoints]
write_json('checkpoint_inventory.json',checkpoint_inventory)
config_inventory=[]
for p in sorted(Q.rglob('holosoma_config.yaml')):
 try:
  c=yaml.safe_load(p.read_text());a=c.get('algo',{}).get('config',{})
  config_inventory.append(dict(path=str(p.relative_to(R)),algo=c.get('algo',{}).get('_target_'),simulator=c.get('simulator',{}).get('config',{}).get('name'),seed=c.get('training',{}).get('seed'),num_envs=c.get('training',{}).get('num_envs'),iterations=a.get('num_learning_iterations'),use_symmetry=a.get('use_symmetry'),symmetry_compatible_io=a.get('symmetry_compatible_io'),bilateral=a.get('bilateral'),command=c.get('command'),phase=c.get('observation',{}).get('groups',{}).get('actor_obs',{}).get('terms',{}).get('sin_phase',{}).get('func')))
 except Exception as e:config_inventory.append(dict(path=str(p.relative_to(R)),error=repr(e)))
write_json('config_inventory.json',config_inventory)
log_inventory=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size) for p in sorted(Q.rglob('*')) if p.is_file() and p.suffix in ('.log','.txt')]
write_json('log_inventory.json',log_inventory)
statuses=[]
for p in sorted(Q.glob('*/status.json')):
 try:statuses.append(dict(path=str(p.relative_to(R)),data=json.loads(p.read_text())))
 except Exception as e:statuses.append(dict(path=str(p.relative_to(R)),error=repr(e)))
write_json('campaign_statuses.json',statuses)
print('ENV',sys.version.split()[0],torch.__version__,'cuda',torch.cuda.is_available())
print('INVENTORY',len(checkpoints),'checkpoints',round(sum(p.stat().st_size for p in checkpoints)/2**30,2),'GiB',len(config_inventory),'configs',len(log_inventory),'logs',round(sum(x['bytes'] for x in log_inventory)/2**20,2),'MiB')
print('STATUS',[(x['path'].split('/')[2],x.get('data',{}).get('phase',x.get('data',{}).get('status'))) for x in statuses])
print('CONFIGS',[(x['path'].split('/')[2],x.get('algo','').split('.')[-1],x.get('simulator'),x.get('iterations'),x.get('use_symmetry'),x.get('symmetry_compatible_io')) for x in config_inventory])
