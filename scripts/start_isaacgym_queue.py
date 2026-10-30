"""Replace the paused IsaacSim queue with fresh IsaacGym training inside Docker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/holosoma')
PAUSE = ROOT / 'logs/bilateral_queue/20261028_101400/paused_state.json'
PID_FILE = Path('/tmp/holosoma_isaacgym_queue.pid')


def running(pid, script):
    try:
        return script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fastsac', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Run inside ycho-holosoma.')
    if PID_FILE.exists() and running(int(PID_FILE.read_text()), 'scripts/run_bilateral_queue.py'):
        if not args.fastsac:
            raise RuntimeError('IsaacGym queue is already running.')
        queue_pid = int(PID_FILE.read_text())
        children = []
        for path in Path('/proc').iterdir():
            if path.name.isdigit() and running(int(path.name), 'src/holosoma/holosoma/train_agent.py'):
                status = (path / 'status').read_text()
                if f'PPid:\t{queue_pid}\n' in status:
                    if b'exp:g1-29dof-fast-sac\0' in (path / 'cmdline').read_bytes():
                        print('FastSAC is already running.', flush=True)
                        return
                    children.append(int(path.name))
        os.kill(queue_pid, signal.SIGKILL)
        for pid in children:
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                if not running(pid, 'src/holosoma/holosoma/train_agent.py'):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
    setup = Path('/tmp/holosoma_isaacgym_setup.exit')
    if not setup.exists() or setup.read_text().strip() != '0':
        raise RuntimeError('IsaacGym installation has not completed successfully.')
    state = json.loads(PAUSE.read_text())
    for pid in state['container_pids']['queue']:
        if running(pid, 'scripts/run_bilateral_queue.py'):
            os.kill(pid, signal.SIGKILL)
    for pid in state['container_pids']['training']:
        script = 'src/holosoma/holosoma/train_agent.py'
        if running(pid, script):
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            for _ in range(60):
                if not running(pid, script):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
    state['superseded_by'] = 'Fresh IsaacGym queue; previous checkpoints preserved.'
    PAUSE.write_text(json.dumps(state, indent=2) + '\n')
    log = ROOT / 'logs/bilateral_queue' / ('gpu3_fastsac_launcher.log' if args.fastsac else 'gpu3_isaacgym_launcher.log')
    with log.open('ab') as stream:
        process = subprocess.Popen(
            ['bash', 'scripts/run_gpu3_fastsac.sh' if args.fastsac else 'scripts/run_gpu3_experiments.sh', '--num-envs', '4096', '--seed', '43'],
            cwd=ROOT / '.isaacgym/holosoma', stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='3', CUDA_DEVICE_ORDER='PCI_BUS_ID'),
        )
    PID_FILE.write_text(str(process.pid) + '\n')
    print(f'IsaacGym queue launcher PID: {process.pid}; log: {log}', flush=True)


if __name__ == '__main__':
    main()
