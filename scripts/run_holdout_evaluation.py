"""Evaluate checkpoint ranges numerically on the assigned GPU; no video sweep."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_command_holdout import atomic_json, cases, protocol_cases


RUNS = dict(biHAM='20261028_150951_isaacgym',
            sym_aug_HAM='20261028_155752_isaacgym',
            HAM='20261028_174024_isaacgym_gpu3',
            MLP='20261028_175105_isaacgym_gpu6')


@contextmanager
def locked(path):
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def finished(output, model, iteration):
    directory=output/'videos'/model/f'model_{iteration:07d}'
    manifest = output/'manifest.json'
    case_list = protocol_cases(json.loads(manifest.read_text())['protocol']) if manifest.exists() else cases()
    return all((directory/case['name']/'metrics.json').exists() for case in case_list)


def update_status(output, gpu, **fields):
    with locked(output/'status.lock'):
        atomic_json(output/f'status_gpu{gpu}.json',dict(gpu=gpu,**fields))
        workers=[json.loads(p.read_text()) for p in sorted(output.glob('status_gpu*.json'))]
        complete=sum(finished(output,m,k) for m in RUNS for k in range(1000,10001,1000))
        state='complete' if complete==40 else ('failed' if any(w['status']=='failed' for w in workers) else 'running')
        atomic_json(output/'status.json',dict(status=state,phase='metrics',gpus=[w['gpu'] for w in workers],
                    jobs_complete=complete,jobs_total=40,workers=workers,video_policy='Selective videos after numeric evaluation'))


def summarize(output, plot_iteration=None, plots=True):
    with locked(output/'summary.lock'):
        _summarize(output, plot_iteration, plots)


def _summarize(output, plot_iteration=None, plots=True):
    rows = []
    for path in output.glob('videos/*/model_*/*/metrics.json'):
        if path.parts[-4] not in RUNS:
            continue
        value = json.loads(path.read_text())
        row = {k:value.get(k) for k in (
            'model','iteration','name','direction','speed','trials','mean_linear_rmse','mean_vx_rmse',
            'mean_vy_rmse','mean_yaw_rmse','mean_yaw_velocity','mean_survival_s','failure_rate','success_rate','survival_rate','measured_trials')}
        row['yaw_command'] = value['command'][2]
        rows.append(row)
    if rows:
        rows.sort(key=lambda r:(r['model'],r['iteration'],r['name']))
        with (output/'summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        atomic_json(output/'summary.json', rows)
        if plots:
            from plot_holdout_evaluation import plot_results
            plot_results(output, iteration=plot_iteration)


def main():
    global RUNS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-root',type=Path,default=Path('/workspace/holosoma/logs/bilateral_queue'))
    parser.add_argument('--gpu',type=int,choices=(3,6),default=3)
    parser.add_argument('--min-iteration',type=int,default=1000)
    parser.add_argument('--max-iteration',type=int,default=10000)
    parser.add_argument('--ascending',action='store_true')
    parser.add_argument('--wait-pid',type=int,help='Let an existing metric worker finish before taking over its queue.')
    parser.add_argument('--runs',type=Path,help='JSON model-to-training-directory mapping for a new campaign.')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--suite',choices=('linear','ccw'),default='linear')
    args=parser.parse_args()
    if not 1000<=args.min_iteration<=args.max_iteration<=10000 or args.min_iteration%1000 or args.max_iteration%1000:
        parser.error('Checkpoint bounds must be multiples of 1000 within 1000..10000.')
    if args.runs:
        RUNS=json.loads(args.runs.read_text())
        if set(RUNS)!={'biHAM','sym_aug_HAM','HAM','MLP'}:
            parser.error('The campaign must contain exactly the four comparison models.')
    output=args.output or args.log_root/'deploy_eval_four_way_10k'
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/f'runner_gpu{args.gpu}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=dict(version=3 if args.suite=='ccw' else 2,suite=args.suite,seed=20261029,trials=32,duration_s=20,warmup_s=2,
                  linear_rmse_limit=.2,yaw_rmse_limit=.3,video_duration_s=8,video_decimation=3,
                  environment='nominal physics, flat ground, observation noise OFF, no pushes',
                  primary_metric='Mean per-trial planar velocity RMSE over observed steps after 2s warmup',
                  failed_trials='Never count auto-reset episodes. Report survival/time with conditional error.',
                  policy='deterministic mean action, saved observation normalization; no learning',
                  cases=cases(args.suite))
    manifest=output/'manifest.json'
    with locked(output/'summary.lock'):
        if manifest.exists() and json.loads(manifest.read_text())['protocol'] != protocol:
            raise ValueError('Evaluation protocol differs: use a new output directory.')
        (output/'index.html').write_text(Path('scripts/holdout_gallery.html').read_text())
        atomic_json(manifest,dict(output=str(output),gpus=[3,6],models=RUNS,protocol=protocol))
    update_status(output,args.gpu,status='waiting_for_training',min_iteration=args.min_iteration,max_iteration=args.max_iteration)
    if args.wait_pid:
        update_status(output,args.gpu,status='waiting_for_existing_metric',pid=args.wait_pid,
                      min_iteration=args.min_iteration,max_iteration=args.max_iteration)
        while True:
            try:
                command=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes().decode().split('\0')
            except FileNotFoundError:
                break
            if 'scripts/evaluate_command_holdout.py' not in command:
                break
            time.sleep(1)
    # These terminal watchers validate their final checkpoints and wait for CUDA process exit.
    while not args.runs:
        states=[args.log_root/RUNS[m]/'stop_at_10000.json' for m in ('HAM','MLP')]
        values=[json.loads(p.read_text()) for p in states]
        if any(v['status'].startswith(('blocked','failed')) for v in values):
            raise RuntimeError('A training watcher failed; evaluation will not start.')
        if all(v['status']=='complete' for v in values):
            break
        time.sleep(10)
    models={}
    for model,run in RUNS.items():
        checkpoints={int(p.stem.split('_')[-1]):p for p in (args.log_root/run).rglob('model_*.pt')}
        expected=set(range(1000,10001,1000))
        if not expected.issubset(checkpoints):
            raise RuntimeError(f'{model}: missing checkpoints {expected-set(checkpoints)}')
        models[model]=checkpoints
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     HOLOSOMA_GRAPHICS_DEVICE_ID='3',DISPLAY='',OMP_NUM_THREADS='1',LOGURU_LEVEL='INFO')
    # Numeric results take priority. Preserve existing videos, but never schedule
    # the 640-clip sweep. Select any further illustrations after reviewing errors.
    iterations=list(range(args.min_iteration,args.max_iteration+1,1000))
    if not args.ascending:
        iterations.reverse()
    completed=sum(finished(output,model,iteration) for model in models for iteration in iterations)
    total=4*len(iterations)
    try:
        for iteration in iterations:
            changed=False
            for model,checkpoints in models.items():
                if finished(output,model,iteration):
                    continue
                update_status(output,args.gpu,status='running',phase='metrics',model=model,
                                        iteration=iteration,jobs_complete=completed,jobs_total=total,
                                        min_iteration=args.min_iteration,max_iteration=args.max_iteration)
                command=[sys.executable,'scripts/evaluate_command_holdout.py','--manifest',str(manifest),
                         '--model',model,'--checkpoint',str(checkpoints[iteration]),'--phase','metrics','--gpu',str(args.gpu)]
                with (output/f'{model}_{iteration}_metrics.log').open('a') as log:
                    subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                completed+=1
                changed=True
                summarize(output, plots=False)
            if changed:
                summarize(output, plot_iteration=iteration)
        summarize(output)
        update_status(output,args.gpu,status='complete',phase='metrics',jobs_complete=completed,jobs_total=total)
    except Exception as error:
        update_status(output,args.gpu,status='failed',phase='metrics',error=str(error))
        raise


if __name__=='__main__':
    main()
