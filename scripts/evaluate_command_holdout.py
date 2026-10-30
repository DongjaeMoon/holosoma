"""Environment evaluation and videos for the four command-holdout policies."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random


def cases(suite='linear'):
    angles = dict(right=-90, forward_right=-45, backward_right=-135,
                  forward=0, backward=180, left=90, forward_left=45, backward_left=135)
    result = [dict(name=f'{name}_{speed:.1f}mps', direction=name, speed=speed,
                 command=[round(speed * math.cos(math.radians(angle)), 8),
                          round(speed * math.sin(math.radians(angle)), 8), 0.0])
            for speed in (0.3, 0.6) for name, angle in angles.items()]
    if suite == 'linear':
        return result
    if suite != 'ccw':
        raise ValueError(f'Unknown command suite: {suite}')
    result.append(dict(name='standing', direction='standing', speed=0., command=[0., 0., 0.]))
    for rate in (.3, .6):
        for turn, sign in (('ccw', 1), ('cw', -1)):
            result.append(dict(name=f'{turn}_{rate:.1f}radps', direction=turn, speed=0.,
                               command=[0., 0., sign * rate]))
    for side, vy in (('left', .6), ('right', -.6)):
        for turn, yaw in (('ccw', .6), ('cw', -.6)):
            result.append(dict(name=f'{side}_0.6mps_{turn}_0.6radps',
                               direction=f'{side}_{turn}', speed=.6, command=[0., vy, yaw]))
    return result


def protocol_cases(protocol):
    return protocol.get('cases', cases())


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0, center-radius), min(1, center+radius)]


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_config(saved, protocol, output, video):
    from holosoma.config_types.logger import DisabledLoggerConfig
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.config_types.terrain import MeshType
    from holosoma.config_types.video import VideoConfig
    from holosoma.config_values.loco.g1.command import g1_29dof_command
    from holosoma.config_values.simulator import isaacsim
    cfg = saved.get_eval_config()
    cfg = replace(cfg, simulator=isaacsim)
    # Controlled nominal-physics flat-ground test, identical for every policy.
    # Keep curriculum setup (including the required average-episode tracker),
    # but freeze curriculum stepping and disable domain/observation noise.
    cfg = replace(cfg,
        training=replace(cfg.training, headless=True, num_envs=1 if video else len(protocol_cases(protocol))*protocol['trials'],
                         seed=protocol['seed'], export_onnx=False),
        robot=replace(cfg.robot, contact_pairs_multiplier=1),
        command=replace(cfg.command, setup_terms={**cfg.command.setup_terms,
            'locomotion_command': g1_29dof_command.setup_terms['locomotion_command']}),
        randomization=RandomizationManagerCfg(),
        curriculum=replace(cfg.curriculum, step_terms={}),
        observation=replace(cfg.observation, groups={k:replace(g, enable_noise=False) for k,g in cfg.observation.groups.items()}),
        terrain=replace(cfg.terrain, terrain_term=replace(cfg.terrain.terrain_term,
            func='holdout_terrain:EvaluationPlane', mesh_type=MeshType.PLANE, terrain_config={},
            num_rows=1, num_cols=1, terrain_length=256, terrain_width=256, horizontal_scale=1., scale_factor=1.)),
        logger=DisabledLoggerConfig(base_dir=str(output/'runtime'), video=VideoConfig(
            enabled=video, interval=10**9, save_dir=str(output/'runtime'), upload_to_wandb=False,
            playback_rate=1/protocol['video_decimation'])),
        algo=replace(cfg.algo, config=replace(cfg.algo.config, buffer_size=16, compile=False,
                                            command_audit=False, eval_callbacks=None)),
        simulator=replace(cfg.simulator, config=replace(cfg.simulator.config,
            sim=replace(cfg.simulator.config.sim, max_episode_length_s=protocol['duration_s']+2),
            isaacsim=replace(cfg.simulator.config.isaacsim,
                kit_args='--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0'))),
    )
    return cfg


def load_actor(algo, checkpoint):
    from holosoma.utils.safe_torch_import import torch
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    algo.actor.load_state_dict(state['actor_state_dict'])
    algo.obs_normalizer.load_state_dict(state['obs_normalizer_state'])
    return algo.get_inference_policy()


def refresh_observations(env):
    """Re-read single-frame terms after setting a command, without stepping physics."""
    manager = env.observation_manager
    if any(group.history_length != 1 for group in manager.cfg.groups.values()):
        raise ValueError('Controlled holdout evaluation requires single-frame observations.')
    env.command_manager.get_state('locomotion_gait').step()
    env.obs_buf_dict = {name: manager._compute_group(name, None) for name in manager.cfg.groups}


def reset_trial(env, commands, seed):
    import numpy as np
    from holosoma.utils.safe_torch_import import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env.set_is_evaluating()
    env.reset_all()
    env.command_manager.commands[:, :3] = commands
    # Recompute after overriding commands so the very first action sees the test command.
    refresh_observations(env)
    initial = torch.cat((env.simulator.robot_root_states, env.simulator.dof_pos), -1).detach().cpu().numpy()
    return env.obs_buf_dict, hashlib.sha256(initial.tobytes()).hexdigest()


def measure(env, algo, policy, protocol):
    """Count only each environment's FIRST episode; capture state before auto-reset."""
    from holosoma.utils.safe_torch_import import torch
    from holosoma.managers.observation.terms.locomotion import base_lin_vel, base_ang_vel
    trials = protocol['trials']
    case_list = protocol_cases(protocol)
    command = torch.tensor([c['command'] for c in case_list], device=env.device).repeat_interleave(trials, 0)
    obs, fingerprint = reset_trial(env, command, protocol['seed'])
    origins = env.terrain_manager.get_state('locomotion_terrain').env_origins
    initial_root = env.simulator.robot_root_states.detach().clone()
    initial_root[:, :3] -= origins
    initial_dof = env.simulator.dof_state.view(env.num_envs, env.num_dof, 2).detach().clone()
    initial_obs = torch.cat([obs[k] for k in algo.config.actor_obs_keys], -1).detach().clone()
    gait = env.command_manager.get_state('locomotion_gait')
    initial_clock = (torch.stack((gait.clock.phase, gait.clock.phase_period), -1).detach().cpu().tolist()
                     if hasattr(gait, 'clock') else [None] * env.num_envs)
    n, steps = env.num_envs, round(protocol['duration_s']/env.dt)
    warmup = round(protocol['warmup_s']/env.dt)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(alive)
    early_timeout = torch.zeros_like(alive)
    count = torch.zeros(n, device=env.device)
    lived = torch.zeros(n, device=env.device)
    squared_error = torch.zeros_like(count)
    squared_axes = torch.zeros(n, 2, device=env.device)
    squared_yaw = torch.zeros_like(count)
    yaw_sum = torch.zeros_like(count)
    velocity_sum = torch.zeros(n, 2, device=env.device)
    captured = {}
    curves = []
    original_check = env._check_termination

    def capture_before_reset():
        original_check()
        captured['velocity'] = base_lin_vel(env)[:, :2].clone()
        captured['yaw'] = base_ang_vel(env)[:, 2].clone()
        captured['failed'] = env.termination_manager.terminated.clone()
        captured['timeout'] = env.time_out_buf.clone()

    env._check_termination = capture_before_reset
    try:
        with torch.no_grad():
            for step in range(steps):
                env.command_manager.commands[:, :3] = command
                action = policy({'actor_obs':torch.cat([obs[k] for k in algo.config.actor_obs_keys], -1)})
                obs, _, _, _ = env.step({'actions':action})
                velocity, yaw = captured['velocity'], captured['yaw']
                finite = torch.isfinite(velocity).all(-1) & torch.isfinite(yaw)
                lived += alive
                if step >= warmup:
                    mask = alive & finite
                    squared_error += torch.where(mask, (velocity-command[:, :2]).square().sum(-1), 0.)
                    squared_axes += torch.where(mask[:, None], (velocity-command[:, :2]).square(), 0.)
                    squared_yaw += torch.where(mask, (yaw-command[:, 2]).square(), 0.)
                    yaw_sum += torch.where(mask, yaw, 0.)
                    velocity_sum += torch.where(mask[:, None], velocity, 0.)
                    count += mask
                if step % 5 == 0:
                    mask = alive & finite
                    weights = mask.reshape(len(case_list), trials)
                    v = torch.where(mask[:, None], velocity, 0.).reshape(len(case_list), trials, 2)
                    e = torch.where(mask, (velocity-command[:, :2]).square().sum(-1), 0.).reshape(len(case_list), trials)
                    y = torch.where(mask, yaw, 0.).reshape(len(case_list), trials)
                    denom = weights.sum(-1)
                    curves.append(((step+1)*env.dt, denom.cpu().tolist(),
                                   (v.sum(1)/denom.clamp_min(1)[:,None]).cpu().tolist(),
                                   (e.sum(1)/denom.clamp_min(1)).sqrt().cpu().tolist(),
                                   (y.sum(1)/denom.clamp_min(1)).cpu().tolist()))
                failed |= alive & (captured['failed'] | ~finite)
                early_timeout |= alive & captured['timeout'] & (step < steps-1)
                alive &= ~(captured['failed'] | captured['timeout'] | ~finite)
    finally:
        env._check_termination = original_check
    linear_rmse = (squared_error/count.clamp_min(1)).sqrt()
    yaw_rmse = (squared_yaw/count.clamp_min(1)).sqrt()
    survived = (~failed) & (~early_timeout) & (lived == steps)
    success = survived & (count > 0) & (linear_rmse <= protocol['linear_rmse_limit']) & (yaw_rmse <= protocol['yaw_rmse_limit'])
    rows = []
    for i, case in enumerate(case_list):
        ids = slice(i*trials,(i+1)*trials)
        values = []
        for j in range(i*trials,(i+1)*trials):
            valid = bool(count[j] > 0)
            values.append(dict(trial=j-i*trials, success=bool(success[j]), failure=bool(failed[j]),
                early_timeout=bool(early_timeout[j]), survived_s=float(lived[j]*env.dt), measured_steps=int(count[j]),
                linear_rmse=float(linear_rmse[j]) if valid else None,
                vx_rmse=float((squared_axes[j,0]/count[j]).sqrt()) if valid else None,
                vy_rmse=float((squared_axes[j,1]/count[j]).sqrt()) if valid else None,
                yaw_rmse=float(yaw_rmse[j]) if valid else None,
                mean_yaw_velocity=float(yaw_sum[j]/count[j]) if valid else None,
                mean_velocity=(velocity_sum[j]/count[j]).tolist() if valid else None,
                initial_state=dict(root=initial_root[j].cpu().tolist(), dof=initial_dof[j].cpu().tolist(),
                                   actor_obs=initial_obs[j].cpu().tolist(), domi_clock=initial_clock[j])))
        successes = int(success[ids].sum())
        measured = [v for v in values if v['measured_steps'] > 0]
        rows.append(dict(**case, trials=trials, successes=successes, success_rate=successes/trials,
            success_ci95=wilson(successes,trials), failure_rate=float(failed[ids].float().mean()),
            survival_rate=float(survived[ids].float().mean()),
            mean_survival_s=float(lived[ids].mean()*env.dt),
            mean_linear_rmse=sum(v['linear_rmse'] for v in measured)/len(measured) if measured else None,
            mean_vx_rmse=sum(v['vx_rmse'] for v in measured)/len(measured) if measured else None,
            mean_vy_rmse=sum(v['vy_rmse'] for v in measured)/len(measured) if measured else None,
            mean_yaw_rmse=sum(v['yaw_rmse'] for v in measured)/len(measured) if measured else None,
            mean_yaw_velocity=sum(v['mean_yaw_velocity'] for v in measured)/len(measured) if measured else None,
            measured_trials=len(measured), initial_state_sha256=fingerprint, per_trial=values,
            curve=[dict(time_s=t, active_trials=counts[i], mean_velocity=vel[i] if counts[i] else None,
                        planar_rmse=err[i] if counts[i] else None,
                        mean_yaw_velocity=yaw[i] if counts[i] else None) for t,counts,vel,err,yaw in curves]))
    return rows


def render_case(env, algo, policy, case, destination, protocol, initial=None):
    from holosoma.utils.safe_torch_import import torch
    from holosoma.managers.observation.terms.locomotion import base_lin_vel, base_ang_vel
    recorder = env.simulator.video_recorder
    recorder.stop_recording()
    command = torch.tensor([case['command']], device=env.device)
    obs, fingerprint = reset_trial(env, command, protocol['seed'])
    if initial is not None:
        ids = torch.tensor([0], device=env.device, dtype=torch.long)
        root = torch.tensor(initial['root'], device=env.device)
        root[:3] += env.terrain_manager.get_state('locomotion_terrain').env_origins[0]
        env.simulator.robot_root_states[0] = root
        env.simulator.dof_state.view(1, env.num_dof, 2)[0] = torch.tensor(initial['dof'], device=env.device)
        env.simulator.set_actor_root_state_tensor_robots(ids, env.simulator.robot_root_states)
        env.simulator.set_dof_state_tensor_robots(ids, env.simulator.dof_state)
        env.simulator.refresh_sim_tensors()
        env._pre_compute_observations_callback()
        expected = torch.tensor(initial['actor_obs'], device=env.device)
        # reset_all takes a zero-action step with a sampled command. Its gait
        # phase can therefore differ (standing vs moving) across batch sizes.
        # Restore the recorded phase using the actual observation term layout.
        components, offset = {}, 0
        for term in env.observation_manager.get_observation_layout(algo.config.actor_obs_keys):
            size = term['dim'] * term['history_length']
            components[term['name']] = expected[offset:offset+size] / term['scale']
            offset += size
        gait = env.command_manager.get_state('locomotion_gait')
        if initial.get('domi_clock') is not None:
            fraction, period = initial['domi_clock']
            gait.clock.phase_period[0] = period
            gait.clock.phase_offset[0] = fraction * period - env.episode_length_buf[0] * env.dt
            gait.step()
        else:
            gait.phase[0] = torch.atan2(components['sin_phase'], components['cos_phase'])
        refresh_observations(env)
        obs = env.obs_buf_dict
        actual = torch.cat([obs[k] for k in algo.config.actor_obs_keys], -1)[0]
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        fingerprint = hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
    recorder.stop_recording()
    recorder.config = replace(recorder.config, save_dir=str(destination))
    recorder.start_recording(0)
    frames, stopped = 0, False
    captured, errors, yaw_errors = {}, [], []
    original_check = env._check_termination
    def capture():
        original_check()
        captured['velocity'] = base_lin_vel(env)[0,:2].clone()
        captured['yaw'] = base_ang_vel(env)[0,2].clone()
    env._check_termination = capture
    try:
        with torch.no_grad():
            for _ in range(round(protocol['video_duration_s']/env.dt)):
                env.command_manager.commands[:, :3] = command
                action = policy({'actor_obs':torch.cat([obs[k] for k in algo.config.actor_obs_keys], -1)})
                obs, _, done, _ = env.step({'actions':action})
                frames += 1
                if frames*env.dt > protocol['warmup_s']:
                    errors.append(float((captured['velocity']-command[0,:2]).square().sum()))
                    yaw_errors.append(float((captured['yaw']-command[0,2]).square()))
                if bool(done[0]):
                    stopped = True
                    break
    finally:
        env._check_termination = original_check
    recorder.stop_recording()
    return dict(seed=protocol['seed'], initial_state_sha256=fingerprint,
                simulated_seconds=frames*env.dt, terminated=stopped,
                linear_rmse=math.sqrt(sum(errors)/len(errors)) if errors else None,
                yaw_rmse=math.sqrt(sum(yaw_errors)/len(yaw_errors)) if yaw_errors else None,
                selection='post-hoc matched-state replay' if initial else 'fixed seed illustrative rollout; not selected by success',
                files=[p.name for p in destination.glob('*.mp4')])


def log_metrics_tensorboard(output, model, iteration, rows):
    """Keep checkpoint RMSE and failure rates visible beside training runs."""
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(output/'tensorboard'/model)) as writer:
        for row in rows:
            for source, label in (('mean_linear_rmse', 'planar_mps'), ('mean_vx_rmse', 'vx_mps'),
                                  ('mean_vy_rmse', 'vy_mps'), ('mean_yaw_rmse', 'yaw_radps')):
                if row.get(source) is not None:
                    writer.add_scalar(f"RMSE/{row['name']}/{label}", row[source], iteration)
            for source in ('failure_rate', 'survival_rate', 'mean_survival_s'):
                if row.get(source) is not None:
                    writer.add_scalar(f"Evaluation/{row['name']}/{source}", row[source], iteration)
        for source, label in (('mean_linear_rmse', 'planar_mps'), ('mean_vy_rmse', 'vy_mps')):
            values = [row[source] for row in rows if row['direction'] == 'right' and row.get(source) is not None]
            if values:
                writer.add_scalar(f'RMSE/right_mean_{label}', sum(values)/len(values), iteration)


def worker(manifest, model, checkpoint, phase):
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.sim_utils import setup_simulation_environment, simulation_resource_session
    from holosoma.utils.helpers import get_class
    from holosoma.simulator.base_simulator.hooks import Phase
    spec = json.loads(manifest.read_text())
    protocol, output = spec['protocol'], Path(spec['output'])
    case_list = protocol_cases(protocol)
    target = output/'videos'/model/checkpoint.stem
    torch.set_num_threads(1)
    video = phase in ('video','highlight')
    video_cases = case_list
    if phase == 'video' and 'video_cases' in protocol:
        lookup = {c['name']: c for c in case_list}
        video_cases = [lookup[name] for name in protocol['video_cases']]
        if not video_cases:
            raise ValueError('video_cases must not be empty.')
    if phase == 'highlight':
        selection = json.loads((output/'highlight_selection.json').read_text())
        target = output/'videos'/(model+'_highlight')/checkpoint.stem
    if not video and all((target/c['name']/'metrics.json').exists() for c in case_list):
        return
    if phase == 'video' and all(list((target/c['name']).glob('*.mp4')) and (target/c['name']/'video.json').exists() for c in video_cases):
        return
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    cfg = build_config(saved, protocol, target, video)
    cfg.save_config(str(output/f'{model}_{checkpoint.stem}_{phase}_config.yaml'))
    # IsaacSim uses the selected CUDA device remapped to local index zero.
    device = 'cuda:0'
    torch.cuda.set_device(device)
    env, device, app = setup_simulation_environment(cfg, device=device)
    with simulation_resource_session(env, app):
        env._randomize_ctrl_delay = False  # Required reset flag when DR setup is deliberately disabled.
        algo = get_class(cfg.algo._target_)(env=env, config=cfg.algo.config, device=device, log_dir=str(target/'runtime'))
        algo.setup()
        policy = load_actor(algo, checkpoint)
        iteration = int(checkpoint.stem.split('_')[-1])
        if not video:
            rows = measure(env, algo, policy, protocol)
            for row in rows:
                row.update(model=model, checkpoint=str(checkpoint), iteration=iteration,
                           protocol=protocol, evaluation='controlled_flat_ground')
                atomic_json(target/row['name']/'metrics.json', row)
            log_metrics_tensorboard(output, model, iteration, rows)
            print(f'METRICS COMPLETE {model} {iteration}', flush=True)
        else:
            # Decimate only the capture hook; preserve simulation/control rate and playback time.
            for record in env.simulator.hooks._records[Phase.FRAME_END]:
                if record.name == 'video.capture_frame':
                    record.every = protocol['video_decimation']
                    record._counter = 0
            selected_cases = [c for c in case_list if c['name']==selection['case']] if phase=='highlight' else video_cases
            for case in selected_cases:
                destination = target/case['name']
                if list(destination.glob('*.mp4')) and (destination/'video.json').exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                initial = selection['rows'][model]['initial_state'] if phase=='highlight' else None
                video_protocol = dict(protocol,video_duration_s=protocol['duration_s']) if initial else protocol
                record = render_case(env, algo, policy, case, destination, video_protocol, initial)
                if phase=='highlight':
                    record.update(selected_trial=selection['trial'], selected_trial_rmse=selection['errors'][model],
                                  selection_rule=selection['selection'], requested_order_observed=selection['requested_order_observed'])
                if not record['files']:
                    raise RuntimeError(f'No video produced: {destination}')
                atomic_json(destination/'video.json', record)
                print(f'VIDEO COMPLETE {model} {iteration} {case["name"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--phase', choices=('metrics','video','highlight'), required=True)
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('gpu must be a nonnegative physical GPU index.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['HOLOSOMA_GRAPHICS_DEVICE_ID'] = str(args.gpu)
    os.environ['DISPLAY'] = ''
    worker(args.manifest, args.model, args.checkpoint, args.phase)
