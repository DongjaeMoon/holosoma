"""CPU regression checks: known tracking errors and exclusion of reset episodes."""
from types import SimpleNamespace as NS
from unittest.mock import patch

from holosoma.utils.safe_torch_import import torch
import evaluate_command_holdout as evaluation
import holosoma.managers.observation.terms.locomotion as observations


def check(suite='linear'):
    case_list = evaluation.cases(suite)
    n = len(case_list) * 2
    command = torch.tensor([c['command'] for c in case_list]).repeat_interleave(2, 0)
    env = NS(device='cpu', num_envs=n, num_dof=29, dt=.1)
    env.simulator = NS(robot_root_states=torch.zeros(n,13), dof_state=torch.zeros(n*29,2))
    env.terrain_manager = NS(get_state=lambda _:NS(env_origins=torch.zeros(n,3)))
    env.termination_manager = NS(terminated=torch.zeros(n,dtype=torch.bool))
    env.time_out_buf = torch.zeros(n,dtype=torch.bool)
    env.command_manager = NS(commands=command.clone(), get_state=lambda _: None)
    env._check_termination = lambda:None
    obs = {'actor_obs':torch.zeros(n,100)}
    env.velocity, env.yaw = command.clone(), torch.zeros(n)
    step = [0]

    def advance(_):
        step[0] += 1
        env.velocity = command.clone()
        env.velocity[1:,0] += .3
        env.velocity[1:,1] += .4
        env.yaw[:] = command[:,2] + .2
        env.termination_manager.terminated.zero_()
        if step[0] == 4:
            env.termination_manager.terminated[1] = True
        if step[0] > 4:
            env.velocity[1,:2] = 1000  # Must NEVER enter the failed first episode's metric.
        env._check_termination()
        env.termination_manager.terminated.zero_()  # Native env resets its mask in the same step.
        return obs, None, None, None

    env.step = advance
    protocol = dict(trials=2,seed=1,duration_s=1.,warmup_s=.2,linear_rmse_limit=.2,yaw_rmse_limit=.3,cases=case_list)
    with patch.object(evaluation,'reset_trial',return_value=(obs,'test')), \
         patch.object(observations,'base_lin_vel',side_effect=lambda _:env.velocity), \
         patch.object(observations,'base_ang_vel',side_effect=lambda _:torch.stack((env.yaw*0,env.yaw*0,env.yaw),-1)):
        rows = evaluation.measure(env,NS(config=NS(actor_obs_keys=['actor_obs'])),lambda _:None,protocol)
    perfect, failed = rows[0]['per_trial']
    assert perfect['linear_rmse'] == 0 and perfect['success']
    assert failed['failure'] and not failed['success'] and failed['measured_steps'] == 2
    assert abs(failed['linear_rmse']-.5)<1e-6 and abs(failed['survived_s']-.4)<1e-6
    assert abs(rows[0]['mean_linear_rmse']-.25)<1e-6 and rows[0]['failure_rate']==.5
    assert len(perfect['initial_state']['dof'])==29
    assert rows[1]['measured_trials']==2 and abs(rows[1]['mean_linear_rmse']-.5)<1e-6
    assert all(abs(row['mean_yaw_rmse']-.2)<1e-6 for row in rows)
    lo,hi=evaluation.wilson(0,32)
    assert abs(lo)<1e-12 and 0 < hi < .2
    print(f'PASS ({suite}): planar/yaw error, warmup, first-failure capture, auto-reset exclusion, DOF snapshot, Wilson interval')


if __name__=='__main__':
    check()
    check('ccw')
