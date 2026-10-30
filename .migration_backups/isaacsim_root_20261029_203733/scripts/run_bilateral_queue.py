#!/usr/bin/env python3
"""Run inside the Holosoma Docker after activating hssim.

python scripts/run_bilateral_queue.py --gpu 3 --num-envs 4096 --seed 43
Add --iterations 2 for a short VRAM measurement before full training.
GPU ID is explicit: no auto-selection and no parallel training.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def gpu_memory(gpu):
    result = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    )
    return int(result.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--iterations", type=int, help="Optional PPO iteration limit for all three runs.")
    parser.add_argument("--disable-symmetry", action="store_true", help="Disable PPO mirroring for ALL runs, for a separate ablation.")
    args = parser.parse_args()
    if args.gpu < 0 or args.num_envs < 1 or (args.iterations is not None and args.iterations < 1):
        parser.error("GPU must be nonnegative; environment and iteration counts must be positive.")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    # This probe and every training child see only the explicitly selected CUDA GPU.
    subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.device_count() == 1"], env=env, check=True)
    output = root / "logs" / "bilateral_queue" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    presets = ("g1-29dof", "g1-29dof-bilateral-magnet", "g1-29dof-bilateral-no-right")
    for index, preset in enumerate(presets, 1):
        label = f"{index}_{preset}_seed{args.seed}"
        command = [
            sys.executable, "src/holosoma/holosoma/train_agent.py", f"exp:{preset}",
            "simulator:isaacsim", "logger:disabled", "--training.seed", str(args.seed),
            "--training.num-envs", str(args.num_envs), "--training.name", label, "--logger.base-dir", str(output / label),
            f"--simulator.config.isaacsim.kit-args=--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu={args.gpu}",
        ]
        if args.iterations is not None:
            command += ["--algo.config.num-learning-iterations", str(args.iterations)]
        if args.disable_symmetry:
            command += ["--algo.config.use-symmetry", "False"]
        before = gpu_memory(args.gpu)
        samples = [before]
        done = threading.Event()
        def sample():
            while not done.wait(1):
                try:
                    samples.append(gpu_memory(args.gpu))
                except (subprocess.SubprocessError, ValueError):
                    pass
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        started = time.monotonic()
        print(f"[{index}/3] GPU {args.gpu}: {preset}; log: {output / (label + '.log')}", flush=True)
        try:
            with (output / f"{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            done.set()
            monitor.join(timeout=6)
        report = dict(
            preset=preset, gpu=args.gpu, seed=args.seed, num_envs=args.num_envs, training_dir=str(output / label),
            command=command, returncode=result.returncode,
            elapsed_seconds=round(time.monotonic() - started, 1),
            baseline_gpu_used_mib=before, peak_gpu_used_mib=max(samples),
            peak_increase_mib=max(samples) - before,
            memory_note="Total memory on the selected GPU, including other processes; sampled every second.",
        )
        (output / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(f"Stopped after {preset} failed; see {output / (label + '.log')}")
    print(f"Three training runs finished. VRAM reports: {output}", flush=True)
    subprocess.run([sys.executable, "scripts/render_bilateral_checkpoints.py", "--queue-dir", str(output), "--gpu", str(args.gpu)], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
