"""Prepare a native-IO MLP holdout experiment; GPU work requires --execute.

Run inside ycho-holosoma after sourcing scripts/source_isaacsim_setup.sh.
Default: 30k, 4096 environments, seed 43, physical GPU 3, empirical normalization,
native gait, compile=True, no symmetry augmentation. Existing runs are preserved.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from evaluate_command_holdout import atomic_json, cases


def build_plan(output, gpu=3, num_envs=4096, seed=43, command_mode='no-right',
               iterations=30000, eval_every=1000):
    preset = f'g1-29dof-mlp-{command_mode}-vanilla-fast-sac-30k'
    training = [sys.executable, 'src/holosoma/holosoma/train_agent.py',
        f'exp:{preset}', 'simulator:isaacsim', 'logger:disabled',
        '--training.seed', str(seed), '--training.num-envs', str(num_envs),
        '--training.name', f'vanilla_mlp_{command_mode}_30k_seed{seed}',
        '--logger.base-dir', str(output/'training'),
        '--algo.config.num-learning-iterations', str(iterations),
        '--algo.config.obs-normalization', 'True',
        '--algo.config.symmetry-compatible-io', 'False',
        '--algo.config.use-symmetry', 'False', '--algo.config.compile', 'True',
        '--algo.config.command-audit', 'True', '--algo.config.command-audit-csv-envs', '1',
        '--simulator.config.isaacsim.kit-args=--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0']
    protocol = dict(version=4, suite='linear_and_standing', seed=20261029,
        trials=32, duration_s=20, warmup_s=2, linear_rmse_limit=.2, yaw_rmse_limit=.3,
        video_duration_s=8, video_decimation=3,
        cases=cases()+[dict(name='standing', direction='standing', speed=0., command=[0., 0., 0.])],
        environment='Nominal physics, flat ground, no pushes or observation noise.',
        policy='Saved actor and empirical-normalization statistics, deterministic mean; no learning.',
        primary_metric='Mean per-trial sqrt(mean((vx-target_vx)^2+(vy-target_vy)^2)) after warmup.',
        failed_trials='Only first episode; retain failure/survival and measured-step counts beside conditional RMSE.')
    worker = [sys.executable, 'scripts/evaluate_command_holdout.py',
        '--manifest', str(output/'evaluation'/'manifest.json'), '--model', 'MLP',
        '--checkpoint', '{checkpoint}', '--phase', 'metrics', '--gpu', str(gpu)]
    return dict(output=str(output), gpu=gpu, simulator='isaacsim', preset=preset,
        iterations=iterations, seed=seed, num_envs=num_envs, command_mode=command_mode,
        normalization='native EmpiricalNormalization (implementation unchanged)',
        phase='native LocomotionGait and native sin_phase/cos_phase',
        use_symmetry=False, compile=True, yaw_range=[-1., 1.], standing_probability=.2,
        command_audit='Every policy environment and sampled replay; CSV env 0 plus violations.',
        training_command=training, checkpoint_iterations=list(range(eval_every, iterations+1, eval_every)),
        evaluation_command_template=worker, protocol=protocol, training_started=False)


def run_child(command, root, environment, log_path, status, phase, **fields):
    print(shlex.join(command), flush=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=root, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        status(phase, pid=process.pid, command=command, log=str(log_path), **fields)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f'{phase} exited with {code}; see {log_path}')


def summarize(output):
    fields = ('model', 'iteration', 'name', 'direction', 'speed', 'trials',
              'mean_linear_rmse', 'mean_vx_rmse', 'mean_vy_rmse', 'mean_yaw_rmse',
              'mean_survival_s', 'failure_rate', 'survival_rate', 'measured_trials')
    rows = []
    for path in sorted(output.glob('videos/MLP/model_*/*/metrics.json')):
        value = json.loads(path.read_text())
        rows.append({key: value.get(key) for key in fields})
    rows.sort(key=lambda row: (row['iteration'], row['name']))
    atomic_json(output/'summary.json', rows)
    if not rows:
        return
    with (output/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    right = []
    for iteration in sorted({row['iteration'] for row in rows}):
        selected = [row for row in rows if row['iteration'] == iteration]
        report = dict(iteration=iteration)
        for direction in ('right', 'left', 'standing'):
            matches = [row for row in selected if row['direction'] == direction]
            for key in ('mean_linear_rmse', 'mean_vy_rmse', 'failure_rate', 'mean_survival_s'):
                values = [row[key] for row in matches if row[key] is not None]
                report[f'{direction}_{key}'] = sum(values)/len(values) if values else None
        for row in selected:
            if row['direction'] == 'right':
                report[f"right_{row['speed']:.1f}_planar_rmse"] = row['mean_linear_rmse']
                report[f"right_{row['speed']:.1f}_vy_rmse"] = row['mean_vy_rmse']
        right.append(report)
    atomic_json(output/'right_rmse_by_checkpoint.json', right)
    columns = list(dict.fromkeys(key for row in right for key in row))
    with (output/'right_rmse_by_checkpoint.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(right)


def execute(plan, root, stage):
    output, gpu = Path(plan['output']), plan['gpu']
    if stage != 'evaluate':
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'plan.json', dict(plan, training_started=True))
        prefix = f'CUDA_VISIBLE_DEVICES={gpu} CUDA_DEVICE_ORDER=PCI_BUS_ID '
        (output/'training_command.txt').write_text(prefix+shlex.join(plan['training_command'])+'\n')
    elif not (output/'plan.json').is_file():
        raise ValueError('Evaluation requires an existing campaign plan.json; training is never started implicitly.')
    started = time.time()
    def status(phase, **fields):
        atomic_json(output/'status.json', dict(phase=phase, gpu=gpu, stage=stage,
            updated=time.time(), elapsed_seconds=round(time.time()-started), **fields))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       DISPLAY='', OMP_NUM_THREADS='1')
    try:
        if stage != 'evaluate':
            run_child(plan['training_command'], root, environment, output/'training_terminal.log',
                      status, 'training')
        configs = list((output/'training').rglob('holosoma_config.yaml'))
        if len(configs) != 1:
            raise RuntimeError(f'Expected one training configuration, found {len(configs)}.')
        run = configs[0].parent
        checkpoints = {iteration: run/f'model_{iteration:07d}.pt'
                       for iteration in plan['checkpoint_iterations']}
        missing = [iteration for iteration, path in checkpoints.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f'Missing checkpoints: {missing}; training will not be restarted.')
        if stage == 'train':
            status('training_complete', run_directory=str(run), checkpoints=len(checkpoints))
            return
        evaluation = output/'evaluation'
        manifest = dict(output=str(evaluation), models={'MLP': str(run)}, protocol=plan['protocol'])
        manifest_path = evaluation/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Evaluation manifest differs; use the original plan/output.')
        atomic_json(manifest_path, manifest)
        total = len(checkpoints)
        for index, (iteration, checkpoint) in enumerate(checkpoints.items()):
            target = evaluation/'videos'/'MLP'/checkpoint.stem
            complete = all((target/case['name']/'metrics.json').is_file()
                           for case in plan['protocol']['cases'])
            if not complete:
                command = [str(checkpoint) if value == '{checkpoint}' else value
                           for value in plan['evaluation_command_template']]
                run_child(command, root, environment, evaluation/f'MLP_{iteration:07d}_metrics.log',
                    status, 'evaluating', iteration=iteration, jobs_complete=index, jobs_total=total)
            summarize(evaluation)
        from plot_holdout_evaluation import plot_results
        status('plotting', jobs_complete=total, jobs_total=total)
        plot_results(evaluation)
        status('complete', jobs_complete=total, jobs_total=total, run_directory=str(run),
               results=str(evaluation/'right_rmse_by_checkpoint.csv'))
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--command-mode', choices=('no-right', 'left-only'), default='no-right',
        help='no-right: forward/backward/left as before; left-only: vx=0, vy in [0,1]. Both retain yaw [-1,1].')
    parser.add_argument('--stage', choices=('all', 'train', 'evaluate'), default='all')
    parser.add_argument('--execute', action='store_true', help='Explicitly start GPU work. Omit to print only.')
    parser.add_argument('--plan-file', type=Path, help='Save the prepared plan without starting training.')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('gpu must be a nonnegative physical GPU index.')
    if args.num_envs < 1 or args.iterations < 1000 or args.iterations % 1000:
        parser.error('Use positive num-envs and iterations divisible by 1000.')
    if args.eval_every < 1000 or args.eval_every % 1000 or args.iterations % args.eval_every:
        parser.error('eval-every must be a multiple of 1000 and divide iterations.')
    if args.stage == 'evaluate' and args.output is None:
        parser.error('--stage evaluate requires --output pointing to the completed training campaign.')
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root/'logs'/'bilateral_queue'/
        ('vanilla_mlp_30k_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    plan_path = output/'plan.json'
    if args.stage == 'evaluate' and plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        plan['gpu'] = args.gpu
        plan['evaluation_command_template'][-1] = str(args.gpu)
    else:
        plan = build_plan(output, args.gpu, args.num_envs, args.seed,
                          args.command_mode, args.iterations, args.eval_every)
    if args.plan_file:
        atomic_json(args.plan_file.resolve(), plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    execute(plan, root, args.stage)


if __name__ == '__main__':
    main()
