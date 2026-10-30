"""One-env MLP no-right audit, IsaacSim/FastSAC on physical GPU 3 only."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, 'src/holosoma/holosoma/train_agent.py',
        'exp:g1-29dof-mlp-no-right-fast-sac-v2', 'simulator:isaacsim', 'logger:disabled',
        '--training.seed', '43', '--training.num-envs', '1',
        '--training.name', 'mlp_no_right_env1_command_audit',
        '--logger.base-dir', str(output/'training'),
        '--algo.config.num-learning-iterations', '10000',
        '--algo.config.batch-size', str(args.batch_size), '--algo.config.command-audit', 'True',
        '--simulator.config.isaacsim.kit-args=--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0']
    started = time.time()
    metadata = dict(gpu=3, simulator='isaacsim', algorithm='FastSAC', num_envs=1,
        iterations=10000, seed=43, batch_size=args.batch_size, yaw_range=[-1., 1.],
        use_symmetry=False, command=command, started=started, checkpoint=None)
    (output/'command.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'command.txt').write_text('CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID '+shlex.join(command)+'\n')
    def status(phase, **fields):
        value = dict(metadata, phase=phase, elapsed_seconds=round(time.time()-started), **fields)
        target = output/'status.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n')
        temp.replace(target)
    run(root, output, command, status)


def run(root, output, command, status):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='1', DISPLAY='')
    status('starting')
    print(shlex.join(command), flush=True)
    with (output/'training_terminal.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        peak = 0
        while process.poll() is None:
            try:
                used = int(subprocess.check_output(['nvidia-smi', '-i', '3',
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5).strip())
                peak = max(peak, used)
            except (subprocess.SubprocessError, ValueError):
                pass
            audits = list((output/'training').rglob('command_audit_status.json'))
            audit = json.loads(audits[0].read_text()) if audits else None
            status('training', pid=process.pid, peak_gpu_total_used_mib=peak,
                   command_audit=audit, run_directory=str(audits[0].parent) if audits else None)
            time.sleep(5)
    audits = list((output/'training').rglob('command_audit_status.json'))
    audit = json.loads(audits[0].read_text()) if audits else None
    status('complete' if process.returncode == 0 else 'failed', returncode=process.returncode,
           peak_gpu_total_used_mib=peak, command_audit=audit,
           run_directory=str(audits[0].parent) if audits else None)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
