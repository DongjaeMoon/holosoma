# Validate saved raw actor observations against saved physical states without simulation.
from scipy.spatial.transform import Rotation
_,_,ncfg,_,_,_,_,_,_=build_saved('native','MLP',10000)
default=np.array([ncfg.robot.init_state.default_joint_angles[n] for n in ncfg.robot.dof_names])
obs_validation=[]
for key,record in initial_records.items():
 x=np.asarray(record['actor_obs'],dtype=np.float64)
 physical=np.asarray(record['physical'],dtype=np.float64)
 root=physical[:,:13]; dof=physical[:,13:].reshape(-1,29,2)
 rot=Rotation.from_quat(root[:,3:7])
 expected=dict(base_ang_vel=rot.inv().apply(root[:,10:13])*.25,dof_pos=dof[:,:,0]-default,dof_vel=dof[:,:,1]*.05,projected_gravity=rot.inv().apply(np.tile([0,0,-1.],(len(x),1))))
 slices=dict(base_ang_vel=slice(29,32),dof_pos=slice(37,66),dof_vel=slice(66,95),projected_gravity=slice(95,98))
 errors={n:float(np.abs(x[:,slices[n]]-v).max()) for n,v in expected.items()}
 obs_validation.append(dict(cohort=key[0],model=key[1],iteration=key[2],case=key[3],samples=len(x),maximum_absolute_errors=errors))
write_json('saved_initial_observation_validation.json',obs_validation)
maxerrors={n:max(v['maximum_absolute_errors'][n] for v in obs_validation) for n in slices}
print('INITIAL_OBSERVATION_MAX_ERRORS',maxerrors,flush=True)
