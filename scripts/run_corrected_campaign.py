"""Root-checkout IsaacSim/FastSAC campaign. Prints a plan unless --execute is supplied."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_command_holdout import atomic_json, cases


PLAN = {3: [('biHAM', 'bilateral'), ('MLP', 'mlp')],
        6: [('sym_aug_HAM', 'ham-sym-aug'), ('HAM', 'ham')]}


def build_plan(output, variant):
    tail = 'no-right-no-clockwise-fast-sac-v3' if variant == 'ccw-v3' else 'no-right-fast-sac-v2'
    suite = 'ccw' if variant == 'ccw-v3' else 'linear'
    queues, runs = [], {}
    for gpu, entries in PLAN.items():
        presets = [f'g1-29dof-{suffix}-{tail}' for _, suffix in entries]
        for index, ((model, _), preset) in enumerate(zip(entries, presets), 1):
            runs[model] = str(output/f'gpu{gpu}'/f'{index}_{preset}_seed43')
        queues.append(dict(gpu=gpu, command=[sys.executable, 'scripts/run_bilateral_queue.py',
            '--gpu', str(gpu), '--simulator', 'isaacsim', '--presets', *presets,
            '--num-envs', '4096', '--seed', '43', '--iterations', '10000',
            '--skip-render', '--output', str(output/f'gpu{gpu}')]))
    evaluation = [dict(gpu=gpu, command=[sys.executable, 'scripts/run_holdout_evaluation.py',
        '--gpu', str(gpu), '--runs', str(output/'runs.json'), '--output', str(output/'evaluation'),
        '--suite', suite, '--min-iteration', str(low), '--max-iteration', str(high)])
        for gpu, low, high in ((3, 6000, 10000), (6, 1000, 5000))]
    return dict(checkout=str(Path(__file__).resolve().parents[1]), simulator='isaacsim',
                variant=variant, output=str(output), runs=runs, training=queues, evaluation=evaluation,
                evaluation_cases=cases(suite), trials_per_case=32, checkpoints=40,
                environment_commands='forward/backward/left; yaw in [0,1]; standing 20%' if suite=='ccw' else 'v2',
                augmentation=dict(biHAM=False, MLP=False, sym_aug_HAM=True, HAM=False),
                note='sym_aug_HAM also trains on reflected rightward/clockwise synthetic samples',
                training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--variant', choices=('corrected-v2', 'ccw-v3'), default='ccw-v3')
    parser.add_argument('--execute', action='store_true', help='Start the prepared training and evaluation queue.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan-file', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root/'logs'/'bilateral_queue'/
        (args.variant.replace('-', '_')+'_isaacsim_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    plan = build_plan(output, args.variant)
    if args.plan_file:
        atomic_json(args.plan_file, plan)
    if args.dry_run or not args.execute:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    evaluation = output / 'evaluation'
    evaluation.mkdir()
    (output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{args.variant}: IsaacSim FastSAC</h1>'
        '<p>GPU 3: biHAM → MLP. GPU 6: sym-aug HAM → HAM. All 10,000 iterations.</p>'
        '<p><a href="status.json">Live queue status</a> · <a href="evaluation/">Metrics / selected videos</a></p>'
        '<p>Normalization, domi phase and phase augmentation corrected. Old results are preserved separately.</p>')
    (evaluation/'index.html').write_text('<meta charset="utf-8"><h1>Waiting for corrected training</h1>'
        '<p>All 1k–10k metrics will run automatically.</p>'
        '<a href="../status.json">Queue status</a>')
    env = dict(os.environ, DISPLAY='', OMP_NUM_THREADS='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    started = time.time()
    runs = {}
    def status(phase, **fields):
        atomic_json(output/'status.json', dict(phase=phase,started=started,updated=time.time(),
            elapsed_seconds=round(time.time()-started),gpus=[3,6],iterations=10000,**fields))
    def launch(command, log, gpu=None):
        child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
        with log.open('w') as stream:
            return subprocess.Popen(command, cwd=root, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
    def await_all(processes, phase):
        # A failed branch does not kill the other GPU's already-authorized work.
        while any(p.poll() is None for p in processes.values()):
            status(phase, workers={str(g):dict(pid=p.pid,returncode=p.poll()) for g,p in processes.items()})
            time.sleep(10)
        codes={g:p.returncode for g,p in processes.items()}
        if any(codes.values()):
            raise RuntimeError(f'{phase} failed: {codes}; inspect per-GPU logs.')
    try:
        processes={}
        runs = plan['runs']
        for job in plan['training']:
            gpu = job['gpu']
            processes[gpu]=launch(job['command'],output/f'training_gpu{gpu}.log',gpu)
        atomic_json(output/'runs.json',runs)
        await_all(processes,'training')
        processes={}
        for job in plan['evaluation']:
            gpu = job['gpu']
            processes[gpu]=launch(job['command'],output/f'evaluation_gpu{gpu}.log',gpu)
        await_all(processes,'metrics')
        if args.variant == 'ccw-v3':
            status('complete', results=str(evaluation), jobs_complete=40, selected_videos=0)
            return
        status('selected_videos',gpu=3)
        selection=json.loads((evaluation/'highlight_selection.json').read_text())
        for model,row in selection['rows'].items():
            command=[sys.executable,'scripts/evaluate_command_holdout.py','--gpu','3',
                '--manifest',str(evaluation/'manifest.json'),'--model',model,
                '--checkpoint',row['checkpoint'],'--phase','highlight']
            process=launch(command,output/f'highlight_{model}.log',3)
            if process.wait():
                raise RuntimeError(f'{model} selected render failed; numerical results are preserved.')
        status('complete',results=str(evaluation),jobs_complete=40,selected_videos=4)
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__': main()
