"""Opt-in recording of commands actually consumed by a no-right policy."""
import atexit
import csv
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys

from holosoma.utils.safe_torch_import import torch


class CommandAudit:
    def __init__(self, env, config, log_dir, writer):
        self.env, self.config, self.writer = env, config, writer
        self.root = Path(log_dir)
        self.csv_envs = config.command_audit_csv_envs
        if self.csv_envs is not None and self.csv_envs < 1:
            raise ValueError('command_audit_csv_envs must be positive or None.')
        indices, scales, offset = {}, {}, 0
        for term in env.observation_manager.get_observation_layout(config.actor_obs_keys):
            if term['history_length'] != 1:
                raise ValueError('Command audit requires single-frame observations.')
            indices[term['name']] = list(range(offset, offset + term['dim']))
            scales[term['name']] = term['scale']
            offset += term['dim']
        self.indices = indices['command_lin_vel'] + indices['command_ang_vel']
        self.scales = torch.tensor([scales['command_lin_vel']] * 2 +
                                   [scales['command_ang_vel']], device=env.device)
        if len(self.indices) != 3 or (self.scales <= 0).any():
            raise ValueError('Expected positively scaled [vx, vy, yaw] commands.')
        self.samples, self.negative, self.replay_samples, self.replay_negative = 0, 0, 0, 0
        self.minimum, self.last_step, self.closed = float('inf'), -1, False
        self.counts = dict(forward=0, backward=0, left=0, right=0, standing=0, other=0)
        self.stream = (self.root/'commands_policy.csv').open('w', buffering=65536)
        self.csv = csv.writer(self.stream)
        self.csv.writerow(['step', 'env_id', 'vx', 'vy', 'yaw', 'manager_vx', 'manager_vy',
                           'manager_yaw', 'normalized_vx', 'normalized_vy', 'normalized_yaw',
                           'body_vx', 'body_vy', 'body_yaw_rate'])
        sources = self.root/'command_audit_sources'
        sources.mkdir(exist_ok=True)
        paths = [Path(__file__), Path(inspect.getfile(type(env.command_manager))),
                 Path(inspect.getfile(type(env.command_manager.get_state('locomotion_command')))),
                 Path(inspect.getfile(type(env.command_manager.get_state('locomotion_gait')))),
                 Path(inspect.getfile(type(env.observation_manager))),
                 Path(__file__).with_name('fast_sac_agent.py'),
                 Path(__file__).with_name('fast_sac_utils.py')]
        hashes = {}
        for path in paths:
            shutil.copy2(path, sources/path.name)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = dict(argv=sys.argv, source_sha256=hashes, num_envs=env.num_envs,
            policy_commands='Actor observation immediately before normalization; CSV values undo observation scales.',
            replay_commands='Raw sampled current/next observations after optional augmentation, before normalization.',
            normalized_commands='Exact normalized actor input channels; not physical command units.',
            csv_envs=self.csv_envs, counters_cover='All policy environments and all sampled replay commands.',
            csv_policy='First N environments (all if None), plus every negative-vy violation.',
            obs_normalization=config.obs_normalization,
            symmetry_compatible_io=config.symmetry_compatible_io,
            use_symmetry=config.use_symmetry, yaw_range=list(env.command_manager.get_state(
                'locomotion_command').command_ranges['ang_vel_yaw']))
        (self.root/'command_audit_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        saved = self.root/'holosoma_config.yaml'
        if saved.exists():
            print('COMMAND AUDIT: full training config\n' + saved.read_text(), flush=True)
        print('COMMAND AUDIT enabled: policy inputs + replay; strict vy<0 detection.', flush=True)
        atexit.register(self.close)

    def record_policy(self, step, obs, normalized):
        from holosoma.managers.observation.terms.locomotion import get_base_lin_vel, get_base_ang_vel

        self.last_step = int(step)
        command = (obs[:, self.indices] / self.scales).detach().float().cpu()
        norm = normalized[:, self.indices].detach().float().cpu()
        manager = self.env.command_manager.commands[:, :3].detach().float().cpu()
        motion = torch.cat((get_base_lin_vel(self.env)[:, :2],
                            get_base_ang_vel(self.env)[:, 2:3]), -1).detach().float().cpu()
        limit = command.shape[0] if self.csv_envs is None else min(self.csv_envs, command.shape[0])
        csv_ids = sorted(set(range(limit)) | set((command[:, 1] < 0).nonzero().flatten().tolist()))
        for env_id in csv_ids:
            self.csv.writerow([step, env_id, *command[env_id].tolist(), *manager[env_id].tolist(),
                               *norm[env_id].tolist(), *motion[env_id].tolist()])
        self.samples += command.shape[0]
        self.negative += int((command[:, 1] < 0).sum())
        self.minimum = min(self.minimum, float(command[:, 1].min()))
        moving = command[:, :2].abs() > 1e-7
        categories = dict(forward=(command[:, 0] > 1e-7) & ~moving[:, 1],
            backward=(command[:, 0] < -1e-7) & ~moving[:, 1],
            left=(command[:, 1] > 1e-7) & ~moving[:, 0],
            right=(command[:, 1] < -1e-7) & ~moving[:, 0],
            standing=(command.abs() <= 1e-7).all(-1))
        for key, mask in categories.items():
            self.counts[key] += int(mask.sum())
        self.counts['other'] = self.samples - sum(self.counts[k] for k in categories)
        for i, key in enumerate(('vx', 'vy', 'yaw')):
            self.writer.add_scalar('Command/'+key, float(command[:, i].mean()), step)
            self.writer.add_scalar('CommandNormalized/'+key, float(norm[:, i].mean()), step)
            self.writer.add_scalar('Motion/'+key, float(motion[:, i].mean()), step)
        self.writer.add_scalar('CommandAudit/negative_vy_total', self.negative, step)
        self.writer.add_scalar('CommandAudit/min_vy_so_far', self.minimum, step)
        for key, count in self.counts.items():
            self.writer.add_scalar('CommandRatio/'+key, count/self.samples, step)
        if step % 100 == 0 or self.negative:
            self.flush()
            preview = command[:min(limit, 4)].tolist()
            print(f'COMMAND AUDIT step={step} first_commands={preview} checked_envs={len(command)} '
                  f'negative_vy_total={self.negative} min_vy={self.minimum}', flush=True)
        if self.negative:
            raise RuntimeError('Negative vy command found in actual actor inputs; see commands_policy.csv.')

    def record_replay(self, data):
        for block in (data, data['next']):
            vy = block['observations'][:, self.indices[1]] / self.scales[1]
            self.replay_samples += vy.numel()
            self.replay_negative += int((vy < 0).sum().item())
        self.writer.add_scalar('CommandAudit/replay_negative_vy_total', self.replay_negative, self.last_step)
        if self.replay_negative:
            self.flush()
            raise RuntimeError('Negative vy command found in actual replay training inputs.')

    def flush(self):
        self.stream.flush()
        status = 'failed' if self.negative or self.replay_negative else (
            'complete' if self.last_step >= self.config.num_learning_iterations else 'running')
        report = dict(status=status, last_step=self.last_step, policy_command_samples=self.samples,
            min_policy_vy=None if self.samples == 0 else self.minimum,
            negative_policy_vy_count=self.negative, replay_command_samples=self.replay_samples,
            negative_replay_vy_count=self.replay_negative, command_counts=self.counts)
        path = self.root/'command_audit_status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)
        self.writer.flush()

    def close(self):
        if not self.closed:
            self.flush()
            self.stream.close()
            self.closed = True
