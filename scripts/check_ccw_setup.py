"""CPU-only validation of root IsaacSim presets, command lifecycle and launch plan."""
from pathlib import Path
from types import SimpleNamespace as NS
from holosoma.utils.safe_torch_import import torch
from holosoma.config_values import experiment as registry
from holosoma.config_values.loco.g1 import experiment as presets
from holosoma.managers.command.manager import CommandManager
from evaluate_command_holdout import cases, build_config
from run_corrected_campaign import build_plan


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    configs = []
    for model in ('bilateral', 'ham_sym_aug', 'ham', 'mlp'):
        key = f'g1_29dof_{model}_no_right_no_clockwise_fast_sac_v3'
        cfg = registry.EXPERIMENT_REGISTRY[key]
        assert 'isaacsim' in cfg.simulator._target_.lower()
        assert cfg.training.checkpoint is None and cfg.training.teacher_checkpoint is None
        assert cfg.algo.config.symmetry_compatible_io and not cfg.algo.config.compile
        old = getattr(presets, f'g1_29dof_{model}_no_right_fast_sac_v2')
        assert old.command.setup_terms['locomotion_command'].params['command_ranges']['ang_vel_yaw'] == [-1., 1.]
        env = NS(device='cpu', num_envs=4096, dt=.02, is_evaluating=False,
                 episode_length_buf=torch.zeros(4096, dtype=torch.long))
        env.command_manager = manager = CommandManager(cfg.command, env, 'cpu')
        manager.setup()
        for reset in (True, False):
            if reset:
                manager.reset(torch.arange(4096))
            else:
                env.episode_length_buf[:] = 500
                manager.step()
            c = manager.commands
            assert (c[:, 1:] >= 0).all() and (c[:, 1:] <= 1).all()
            assert ((c[:, 0] == 0) | (c[:, 1] == 0)).all()
            assert .15 < (c == 0).all(-1).float().mean() < .25
            assert (c[:, 0] > 0).any() and (c[:, 0] < 0).any() and (c[:, 1] > 0).any()
        configs.append(cfg)
    assert [c.algo.config.use_symmetry for c in configs] == [False, True, False, False]
    assert all(c.command == configs[0].command for c in configs)
    suite = cases('ccw')
    assert len(suite) == 25 and suite[:16] == cases()
    assert len({c['name'] for c in suite}) == 25
    protocol = dict(cases=suite, trials=32, seed=20261029, duration_s=20., video_decimation=3)
    cfg = build_config(configs[-1], protocol, Path('/tmp/ccw_config_only'), False)
    assert cfg.training.num_envs == 800 and 'isaacsim' in cfg.simulator._target_.lower()
    # Verify command refresh against the root observation API without physics.
    from dataclasses import replace
    from holosoma.managers.observation.manager import ObservationManager
    from evaluate_command_holdout import refresh_observations
    group = configs[-1].observation.groups['actor_obs']
    obs_cfg = replace(configs[-1].observation, groups={'actor_obs': replace(group, enable_noise=False,
        terms={key: group.terms[key] for key in ('command_lin_vel', 'command_ang_vel')})})
    env = NS(device='cpu', num_envs=2, command_manager=NS(commands=torch.tensor([[0.,.6,.3],[0.,-.6,-.3]]),
        get_state=lambda _:NS(step=lambda:None)))
    env.observation_manager = ObservationManager(obs_cfg, env, 'cpu')
    refresh_observations(env)
    torch.testing.assert_close(env.obs_buf_dict['actor_obs'], env.command_manager.commands[:,[2,0,1]])
    plan = build_plan(Path('/tmp/ccw_plan_only'), 'ccw-v3')
    assert len(plan['runs']) == 4 and len(plan['evaluation_cases']) == 25 and not plan['training_started']
    assert {job['gpu'] for job in plan['training']} == {3, 6}
    assert all('isaacsim' in job['command'] for job in plan['training'])
    import holosoma.agents.fast_sac.fast_sac_agent as agent
    assert '/.isaacgym/' not in agent.__file__, agent.__file__
    print('PASS: four root IsaacSim presets; reset/step vy>=0 and yaw>=0; standing; v2 unchanged; augmentation flags; 25-case evaluation; plan only.')


if __name__ == '__main__':
    main()
