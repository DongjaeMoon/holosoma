"""CPU-only checks of baseline isolation, real command lifecycle, audit and preparation."""
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

from holosoma.utils.safe_torch_import import torch
from holosoma.config_values.loco.g1 import experiment as presets
from holosoma.managers.command.manager import CommandManager
from holosoma.managers.command.terms.locomotion import LocomotionGait
from holosoma.agents.fast_sac.command_audit import CommandAudit
from run_vanilla_mlp_30k import build_plan, summarize
from evaluate_command_holdout import build_config


class Layout:
    def get_observation_layout(self, keys):
        return [dict(name='command_lin_vel', history_length=1, dim=2, scale=1.),
                dict(name='command_ang_vel', history_length=1, dim=1, scale=1.)]


class Writer:
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass


def main():
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty.'
    torch.set_num_threads(1)
    torch.manual_seed(43)
    base = presets.g1_29dof_fast_sac
    for mode in ('no_right', 'left_only'):
        cfg = getattr(presets, f'g1_29dof_mlp_{mode}_vanilla_fast_sac_30k')
        expected = asdict(base)
        expected['algo']['config'].update(use_symmetry=False, num_learning_iterations=30000)
        term = expected['command']['setup_terms']['locomotion_command']
        if mode == 'no_right':
            term['func'] = 'holosoma.managers.command.terms.three_direction:ThreeDirectionCommand'
        else:
            term['params']['command_ranges'].update(lin_vel_x=[0., 0.], lin_vel_y=[0., 1.])
        assert asdict(cfg) == expected, 'Unexpected non-vanilla setting.'
        assert cfg.algo.config.compile and cfg.algo.config.obs_normalization
        assert not cfg.algo.config.symmetry_compatible_io
        env = NS(device='cpu', num_envs=4096, dt=.02, is_evaluating=False,
            episode_length_buf=torch.zeros(4096, dtype=torch.long), simulator=NS(),
            observation_manager=Layout())
        manager = CommandManager(cfg.command, env, 'cpu')
        env.command_manager = manager
        manager.setup()
        assert type(manager.get_state('locomotion_gait')) is LocomotionGait
        manager.reset(None)
        command = manager.commands
        assert (command[:, 1] >= 0).all() and (command[:, 1] > 0).any()
        assert (command[:, 2] > 0).any() and (command[:, 2] < 0).any()
        if mode == 'no_right':
            assert (command[:, 0] > 0).any() and (command[:, 0] < 0).any()
            assert not ((command[:, 0] != 0) & (command[:, 1] != 0)).any()
        else:
            assert command[:, 0].count_nonzero() == 0
        command[:, 1] = -1
        env.episode_length_buf.fill_(500)
        manager.step()
        assert (command[:, 1] >= 0).all(), 'Periodic resampling bypassed holdout.'
        gait = manager.get_state('locomotion_gait')
        command[:] = 0
        gait.step()
        from holosoma.managers.observation.terms.locomotion import sin_phase, cos_phase
        torch.testing.assert_close(sin_phase(env), torch.zeros(4096, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(cos_phase(env), -torch.ones(4096, 2))
        env.simulator.robot_root_states = torch.zeros(4096, 13)
        env.simulator.robot_root_states[:, 6] = 1
        env.base_quat = env.simulator.robot_root_states[:, 3:7]
        with tempfile.TemporaryDirectory() as temp:
            audit = CommandAudit(env, replace(cfg.algo.config, command_audit=True,
                                             command_audit_csv_envs=1), temp, Writer())
            obs = command.clone()
            audit.record_policy(0, obs, obs)
            audit.record_replay(dict(observations=obs, next=dict(observations=obs)))
            audit.flush()
            assert audit.samples == 4096 and audit.replay_samples == 8192
            assert len(list(csv.DictReader((Path(temp)/'commands_policy.csv').open()))) == 1
            obs[-1, 1] = -.5
            try:
                audit.record_policy(1, obs, obs)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Negative command outside CSV subset escaped detection.')
            audit.close()
            records = list(csv.DictReader((Path(temp)/'commands_policy.csv').open()))
            assert records[-1]['env_id'] == '4095' and float(records[-1]['vy']) < 0
    assert presets.g1_29dof_mlp_no_right_fast_sac_v2.algo.config.symmetry_compatible_io
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        output = root/'not_started'
        plan = build_plan(output)
        assert plan['checkpoint_iterations'] == list(range(1000, 30001, 1000))
        assert len(plan['protocol']['cases']) == 17 and plan['protocol']['trials'] == 32
        from holosoma.config_values.experiment import get_annotated_experiment_config
        from holosoma.utils.config_registry import parse_config
        previous_argv = sys.argv
        try:
            sys.argv = plan['training_command'][1:]
            parsed = parse_config(get_annotated_experiment_config)
        finally:
            sys.argv = previous_argv
        assert parsed.algo.config.command_audit_csv_envs == 1 and parsed.algo.config.command_audit
        assert parsed.algo.config.compile and not parsed.algo.config.symmetry_compatible_io
        evaluated = build_config(parsed, plan['protocol'], root/'eval_config', False)
        assert not evaluated.algo.config.command_audit and not evaluated.algo.config.symmetry_compatible_io
        assert evaluated.command.setup_terms['locomotion_gait'] == base.command.setup_terms['locomotion_gait']
        assert evaluated.training.num_envs == 17*32
        subprocess.run([sys.executable, 'scripts/run_vanilla_mlp_30k.py', '--output', str(output),
                        '--plan-file', str(root/'prepared.json')], check=True, stdout=subprocess.DEVNULL)
        assert not output.exists(), 'Default invocation created an execution directory.'
        assert json.loads((root/'prepared.json').read_text())['training_started'] is False
        for case in plan['protocol']['cases']:
            row = dict(case, model='MLP', iteration=30000, trials=32,
                       mean_linear_rmse=.2, mean_vy_rmse=.1, failure_rate=0., mean_survival_s=20.)
            if case['direction'] == 'right':
                row['mean_linear_rmse'] = .3 if case['speed'] == .3 else .5
            target = root/'evaluation'/'videos'/'MLP'/'model_0030000'/case['name']/'metrics.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(row))
        summarize(root/'evaluation')
        result = json.loads((root/'evaluation'/'right_rmse_by_checkpoint.json').read_text())[0]
        assert abs(result['right_mean_linear_rmse']-.4) < 1e-9
    print('PASS: only intended baseline differences; native phase; reset/periodic no-right commands; '
          'all-env audit despite compact CSV; 30 checkpoints; dry-run starts nothing; right RMSE aggregation.')


if __name__ == '__main__':
    main()
