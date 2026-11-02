# Separate legacy normalizer defects from native standing-phase convention mismatch.
phase_diagnostics=[]
for cohort in ('legacy_v1','v2'):
 _,state,cfg,env,actor,critic,router,norm,physical=build_saved(cohort,'biHAM',10000)
 intrinsic=DomiSymmetry(env)
 allobs=torch.tensor([obs for key,record in initial_records.items() if key[:3]==(cohort,'biHAM',10000) for obs in record['actor_obs']])
 phase_compatible=(allobs[:,35:37].sum(-1).abs()<1e-6)&(allobs[:,98:100].sum(-1).abs()<1e-6)
 reflect=lambda z:physical.mirror_xz_plane(z,env,cfg.algo.config.actor_obs_keys)
 domir=lambda z:intrinsic.mirror_xz_plane(z,env,cfg.algo.config.actor_obs_keys)
 reflected_mean=domir(norm._mean)
 mean=.5*(norm._mean+reflected_mean)
 variance=.5*(norm._var+domir(norm._var).abs())+.25*(norm._mean-reflected_mean).square()
 projection=lambda z:(z-mean)/(variance.sqrt()+norm.eps)
 for label,mask in [('antisymmetric_phase_pair',phase_compatible),('non_antisymmetric_phase_pair',~phase_compatible)]:
  x=allobs[mask]
  if not len(x):continue
  x=x[::max(1,len(x)//96)]
  with torch.no_grad():
   rx=reflect(x)
   raw=actor(rx)[0]-physical.mirror_action_xz_plane(actor(x)[0])
   trained=actor(norm(rx,update=False))[0]-physical.mirror_action_xz_plane(actor(norm(x,update=False))[0])
   projected=actor(projection(rx))[0]-physical.mirror_action_xz_plane(actor(projection(x))[0])
  phase_diagnostics.append(dict(cohort=cohort,subset=label,sampled_observations=len(x),physical_reflection_type=type(physical).__name__,raw_actor_physical_reflection_rms=float(raw.square().mean().sqrt()),trained_policy_physical_reflection_rms=float(trained.square().mean().sqrt()),domi_projected_normalizer_physical_reflection_rms=float(projected.square().mean().sqrt()),cos_phase_first=x[0,35:37].tolist()))
write_json('legacy_phase_normalizer_isolation.json',phase_diagnostics)
print('HISTORICAL_PHASE_ISOLATION',phase_diagnostics,flush=True)
