# Call the real batch preparation method with controlled CPU tensors. No updates.
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from tensordict import TensorDict
batch_reports=[]
for cohort,model in [('v2','MLP'),('v2','biHAM'),('v2','HAM'),('v2','sym_aug_HAM'),('native','MLP')]:
 _,_,config,env,actor,critic,router,norm,symmetry=build_saved(cohort,model,10000)
 env.num_envs=2
 na=norm._mean.numel();nc=na+3
 def sample_controlled(size):
  count=env.num_envs*size
  obs=torch.zeros(count,na);obs[:,34]=.5
  crit=torch.zeros(count,nc);crit[:,37]=.5
  return TensorDict(dict(observations=obs,critic_observations=crit,actions=torch.zeros(count,29),next=dict(observations=obs.clone(),critic_observations=crit.clone(),rewards=torch.ones(count),dones=torch.zeros(count),truncations=torch.zeros(count),effective_n_steps=torch.ones(count))),[count])
 fake=NS(config=config.algo.config,env=env,rb=NS(sample=sample_controlled),symmetry_utils=symmetry,command_audit=None)
 identity=lambda x,**kwargs:x
 batches=FastSACAgent._sample_and_prepare_batches(fake,2,8,identity,identity)
 negatives=[int((b['observations'][:,34]<0).sum()) for b in batches]
 positives=[int((b['observations'][:,34]>0).sum()) for b in batches]
 actor_indices=[i for i in range(8) if i%config.algo.config.policy_frequency==1]
 rec=dict(cohort=cohort,model=model,use_symmetry=config.algo.config.use_symmetry,batch_sizes=[b.batch_size[0] for b in batches],negative_counts_per_update=negatives,positive_counts_per_update=positives,actor_update_indices=actor_indices,actor_negative_samples=sum(negatives[i] for i in actor_indices),actor_positive_samples=sum(positives[i] for i in actor_indices))
 assert sum(negatives)>0 if config.algo.config.use_symmetry else sum(negatives)==0
 batch_reports.append(rec)
write_json('actual_batch_preparation_direction_check.json',batch_reports)
print('BATCH_DIRECTION_CHECK',[(x['cohort'],x['model'],x['actor_negative_samples'],x['actor_positive_samples']) for x in batch_reports],flush=True)
