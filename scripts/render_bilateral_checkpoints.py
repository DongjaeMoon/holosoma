"""Render every saved checkpoint on the queue GPU, after all three training runs."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

DIRECTIONS = {"forward": (0.6, 0, 0), "backward": (-0.6, 0, 0),
              "left": (0, 0.6, 0), "right": (0, -0.6, 0)}


def render_run(report_file, gpu, steps):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from holosoma.config_types.logger import DisabledLoggerConfig
    from holosoma.config_types.video import VideoConfig
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.sim_utils import setup_simulation_environment, simulation_resource_session
    from holosoma.utils.helpers import get_class

    report = json.loads(report_file.read_text())
    checkpoints = sorted(Path(report["training_dir"]).rglob("model_*.pt"), reverse=True)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints in {report['training_dir']}")
    output = report_file.parent / "videos" / report_file.stem
    output.mkdir(parents=True, exist_ok=True)
    saved, wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoints[0])))
    cfg = saved.get_eval_config()
    cfg = replace(cfg,
        training=replace(cfg.training, headless=True, num_envs=1, export_onnx=False),
        logger=DisabledLoggerConfig(base_dir=str(output / "eval_logs"), video=VideoConfig(
            enabled=True, interval=10**9, save_dir=str(output), upload_to_wandb=False)),
        algo=replace(cfg.algo, config=replace(cfg.algo.config, load_optimizer=False, eval_callbacks=None)),
        simulator=replace(cfg.simulator, config=replace(cfg.simulator.config,
            isaacsim=replace(cfg.simulator.config.isaacsim,
                kit_args=f"--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu={gpu}"))),
    )
    env, device, app = setup_simulation_environment(cfg)
    with simulation_resource_session(env, app):
        from holosoma.managers.observation.terms.locomotion import base_lin_vel
        algo = get_class(cfg.algo._target_)(env=env, config=cfg.algo.config,
            device=device, log_dir=str(output / "eval_logs"))
        algo.setup()
        algo.attach_checkpoint_metadata(saved, wandb_path)
        recorder = env.simulator.video_recorder
        if recorder is None:
            raise RuntimeError("Video recorder did not initialize")
        with SummaryWriter(str(output / "evaluation_metrics")) as writer:
            for checkpoint in checkpoints:
                algo.load(str(checkpoint))
                policy = algo.get_inference_policy()
                iteration = int(checkpoint.stem.rsplit("_", 1)[1])
                for direction, command in DIRECTIONS.items():
                    destination = output / checkpoint.stem / direction
                    metrics_file = destination / "metrics.json"
                    if metrics_file.exists() and list(destination.glob("*.mp4")):
                        continue
                    destination.mkdir(parents=True, exist_ok=True)
                    env.set_is_evaluating()
                    obs = env.reset_all()
                    commands = env.command_manager.commands
                    commands[:] = torch.tensor(command, device=commands.device, dtype=commands.dtype)
                    recorder.stop_recording()
                    recorder.config = replace(recorder.config, save_dir=str(destination))
                    recorder.start_recording(iteration)
                    velocities, rewards, terminated = [], [], False
                    with torch.inference_mode():
                        for _ in range(steps):
                            actions = policy({"actor_obs": torch.cat([obs[k] for k in algo.actor_obs_keys], dim=-1)})
                            obs, reward, done, _ = env.step({"actions": actions})
                            velocities.append(base_lin_vel(env)[0].detach().cpu())
                            rewards.append(float(reward.mean()))
                            if done.any():
                                terminated = True
                                break
                    recorder.stop_recording()
                    actual = torch.stack(velocities).mean(0).tolist()
                    metrics = dict(checkpoint=str(checkpoint), direction=direction, command=list(command),
                        mean_body_velocity=actual, mean_reward=sum(rewards)/len(rewards),
                        steps=len(rewards), terminated=terminated,
                        videos=[p.name for p in destination.glob("*.mp4")])
                    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
                    for axis, value in zip(("vx", "vy", "vz"), actual):
                        writer.add_scalar(f"{direction}/mean_body_{axis}", value, iteration)
                    writer.add_scalar(f"{direction}/terminated", int(terminated), iteration)
                    writer.add_scalar(f"{direction}/survived_steps", len(rewards), iteration)
                    writer.flush()
                    print(f"Rendered {checkpoint.name}: {direction} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for report in sorted(args.queue_dir.glob("*.json")):
        if "training_dir" in json.loads(report.read_text()):
            code = "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; from render_bilateral_checkpoints import render_run; render_run(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))"
            subprocess.run([sys.executable, "-c", code, str(report), str(args.gpu), str(args.steps)], check=True)


if __name__ == "__main__":
    main()
