from dataclasses import replace

from holosoma.config_types.algo import BilateralNetworkConfig

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacsim,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacsim,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

# Same MDP and PPO hyperparameters; only the actor/critic architecture changes.
g1_29dof_bilateral_magnet = replace(
    g1_29dof,
    algo=replace(
        g1_29dof.algo,
        _target_="holosoma.agents.ppo.bilateral_ppo.BilateralPPO",
        config=replace(g1_29dof.algo.config, bilateral=BilateralNetworkConfig()),
    ),
)

# Training commands: forward / backward / left. Right is body-frame vy < 0.
_locomotion_command = g1_29dof.command.setup_terms["locomotion_command"]
g1_29dof_bilateral_no_right = replace(
    g1_29dof_bilateral_magnet,
    command=replace(
        g1_29dof.command,
        setup_terms={
            **g1_29dof.command.setup_terms,
            "locomotion_command": replace(
                _locomotion_command,
                func="holosoma.managers.command.terms.three_direction:ThreeDirectionCommand",
            ),
        },
    ),
)

__all__ = ["g1_29dof", "g1_29dof_fast_sac", "g1_29dof_bilateral_magnet", "g1_29dof_bilateral_no_right"]


g1_29dof_bilateral_fast_sac = replace(
    g1_29dof_fast_sac,
    algo=replace(g1_29dof_fast_sac.algo,
        _target_="holosoma.agents.fast_sac.bilateral_fast_sac.BilateralFastSAC",
        config=replace(g1_29dof_fast_sac.algo.config, bilateral=BilateralNetworkConfig(), compile=False)),
)
g1_29dof_bilateral_no_right_fast_sac = replace(
    g1_29dof_bilateral_fast_sac, command=g1_29dof_bilateral_no_right.command,
    # Mirrored replay samples would reintroduce the held-out rightward commands.
    algo=replace(g1_29dof_bilateral_fast_sac.algo,
        config=replace(g1_29dof_bilateral_fast_sac.algo.config, use_symmetry=False, num_learning_iterations=10000)),
)

# Same three-direction command holdout for all four comparison conditions.
g1_29dof_ham_no_right_fast_sac = replace(
    g1_29dof_bilateral_no_right_fast_sac,
    algo=replace(g1_29dof_bilateral_no_right_fast_sac.algo,
        _target_="holosoma.agents.fast_sac.ham_fast_sac.HAMFastSAC",
        config=replace(g1_29dof_bilateral_no_right_fast_sac.algo.config,
            bilateral=replace(g1_29dof_bilateral_no_right_fast_sac.algo.config.bilateral,
                hidden_dims=(384, 576, 384), use_sym_param=False, checkpoint_layers=False))),
)
g1_29dof_ham_sym_aug_no_right_fast_sac = replace(
    g1_29dof_ham_no_right_fast_sac,
    algo=replace(g1_29dof_ham_no_right_fast_sac.algo,
        config=replace(g1_29dof_ham_no_right_fast_sac.algo.config, use_symmetry=True)),
)
g1_29dof_mlp_no_right_fast_sac = replace(
    g1_29dof_fast_sac, command=g1_29dof_bilateral_no_right.command,
    algo=replace(g1_29dof_fast_sac.algo,
        config=replace(g1_29dof_fast_sac.algo.config, use_symmetry=False, compile=False, num_learning_iterations=10000)),
)


def _corrected_io(experiment):
    """Versioned comparison: common domi phase and preprocessing for all four nets."""
    gait = "holosoma.managers.command.terms.domi_gait:"
    command = replace(experiment.command, setup_terms={
        **experiment.command.setup_terms,
        "locomotion_gait": replace(experiment.command.setup_terms["locomotion_gait"],
                                  func=gait + "DomiGait", params={"period_range": (.9, 1.1)}),
    })
    observation = replace(experiment.observation, groups={
        key: replace(group, terms={name: replace(term, func=gait + name)
                                   if name in {"sin_phase", "cos_phase"} else term
                                   for name, term in group.terms.items()})
        for key, group in experiment.observation.groups.items()
    })
    return replace(experiment, command=command, observation=observation,
                   algo=replace(experiment.algo, config=replace(experiment.algo.config, symmetry_compatible_io=True)))


g1_29dof_bilateral_no_right_fast_sac_v2 = _corrected_io(g1_29dof_bilateral_no_right_fast_sac)
g1_29dof_ham_sym_aug_no_right_fast_sac_v2 = _corrected_io(g1_29dof_ham_sym_aug_no_right_fast_sac)
g1_29dof_ham_no_right_fast_sac_v2 = _corrected_io(g1_29dof_ham_no_right_fast_sac)
g1_29dof_mlp_no_right_fast_sac_v2 = _corrected_io(g1_29dof_mlp_no_right_fast_sac)


def _counterclockwise_only(experiment):
    """Keep v2 intact; hold out clockwise (negative body-z yaw) commands."""
    term = experiment.command.setup_terms["locomotion_command"]
    params = {**term.params, "command_ranges": {
        **term.params["command_ranges"], "ang_vel_yaw": [0.0, 1.0],
    }}
    return replace(experiment, command=replace(experiment.command, setup_terms={
        **experiment.command.setup_terms, "locomotion_command": replace(term, params=params),
    }))


g1_29dof_bilateral_no_right_no_clockwise_fast_sac_v3 = _counterclockwise_only(g1_29dof_bilateral_no_right_fast_sac_v2)
g1_29dof_ham_sym_aug_no_right_no_clockwise_fast_sac_v3 = _counterclockwise_only(g1_29dof_ham_sym_aug_no_right_fast_sac_v2)
g1_29dof_ham_no_right_no_clockwise_fast_sac_v3 = _counterclockwise_only(g1_29dof_ham_no_right_fast_sac_v2)
g1_29dof_mlp_no_right_no_clockwise_fast_sac_v3 = _counterclockwise_only(g1_29dof_mlp_no_right_fast_sac_v2)

# Native FastSAC MLP baseline: retain empirical normalization, native gait,
# torch.compile, rewards, randomization and all other vanilla hyperparameters.
g1_29dof_mlp_no_right_vanilla_fast_sac_30k = replace(
    g1_29dof_fast_sac,
    command=g1_29dof_bilateral_no_right.command,
    algo=replace(g1_29dof_fast_sac.algo, config=replace(
        g1_29dof_fast_sac.algo.config, use_symmetry=False, num_learning_iterations=30000,
    )),
)

# Optional literal left-only control; the default baseline retains forward/backward.
_native_velocity = g1_29dof_fast_sac.command.setup_terms["locomotion_command"]
g1_29dof_mlp_left_only_vanilla_fast_sac_30k = replace(
    g1_29dof_mlp_no_right_vanilla_fast_sac_30k,
    command=replace(g1_29dof_fast_sac.command, setup_terms={
        **g1_29dof_fast_sac.command.setup_terms,
        "locomotion_command": replace(_native_velocity, params={
            **_native_velocity.params,
            "command_ranges": {**_native_velocity.params["command_ranges"],
                               "lin_vel_x": [0., 0.], "lin_vel_y": [0., 1.]},
        }),
    }),
)
