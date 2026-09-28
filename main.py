#!/usr/bin/env python3
"""
PCI-MAPPO Training Script

Example usage:
    python main.py --config configs/default.yaml
    python main.py --env simple_cooperative --num-agents 3 --use-wandb
    python main.py --config configs/small_scale.yaml --device cuda
"""

import os
import torch
from utils import parse_args, create_config_from_args, Logger
from utils.seed_utils import set_global_seed, ensure_reproducible_environment, log_reproducibility_info


def configure_jax_runtime(device: str):
    """Pin JAX backend before any JAX-dependent modules are imported."""
    if str(device).lower() != "cpu":
        return

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import jax
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_platform_name", "cpu")
    except Exception:
        # JAX may not be installed for non-JAX runs, or may be imported later.
        pass


def import_runtime_modules():
    """Import env/algorithm/trainer modules after runtime backend configuration."""
    global create_environment, get_available_environments
    global PCIMAPPOAlgorithm, MAPPOAlgorithm, rMAPPOAlgorithm, IPPOAlgorithm, rIPPOAlgorithm
    global X_PCIAlgorithm, COMAAlgorithm, rCOMAAlgorithm, rPCIAlgorithm, qPCIAlgorithm
    global rQMIXAlgorithm, JAXPPOAlgorithm, JAXQPCIAlgorithm, JAXRPCIAlgorithm, JAXQMIXAlgorithm
    global Trainer, JAXTrainer

    from envs import create_environment, get_available_environments
    from models import (
        rMAPPOAlgorithm,
        rIPPOAlgorithm,
        rPCIAlgorithm,
        rQMIXAlgorithm,
        JAXPPOAlgorithm,
        JAXRPCIAlgorithm,
        JAXQMIXAlgorithm,
    )
    from runner import Trainer, JAXTrainer


def sync_effective_environment_config(config, env):
    """Reflect environment-side normalizations in saved configs and wandb metadata."""
    env_config = getattr(config, 'environment', None)
    if env_config is None:
        return

    if env.__class__.__name__ not in ['SMAXWrapper', 'SMAXBatchWrapper']:
        return

    if hasattr(env, 'use_allowed_targets'):
        env_config.use_allowed_targets = bool(env.use_allowed_targets)


def create_algorithm(env, config):
    """Create algorithm based on config.algorithm.type."""
    network_config = config.network
    algorithm_config = config.algorithm

    # Get required_agents from environment if available (for StagHunt)
    env_required_agents = None
    if hasattr(env, 'required_agents'):
        env_required_agents = env.required_agents

    # Calculate effective observation dimension
    # If add_agent_id is enabled, trainer will add agent_id one-hot to observations
    base_obs_dim = env.get_obs_dim()
    add_agent_id = getattr(algorithm_config, 'add_agent_id', False)
    effective_obs_dim = base_obs_dim + env.num_agents if add_agent_id else base_obs_dim

    if add_agent_id:
        print(f"[INFO] add_agent_id enabled: base_obs_dim={base_obs_dim}, effective_obs_dim={effective_obs_dim}")

    # Normalize algorithm type string (hyphen to underscore, lower case)
    algorithm_type = getattr(algorithm_config, 'type', 'pci_mappo').lower().replace('-', '_')
    
    if algorithm_type == 'pci_mappo':
        algorithm = PCIMAPPOAlgorithm(
            env = env,
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            w_dim=base_obs_dim, # observation dim only (excluding previous reward and agent_id)
            hidden_dim=network_config.hidden_dim,
            lr_actor=network_config.lr_actor,
            lr_bridge=network_config.lr_bridge,
            lr_critic=network_config.lr_critic,
            causal_lambda=algorithm_config.causal_lambda,
            clip_delta=getattr(algorithm_config, 'clip_delta', 0.2), # [ADDED]
            causal_lambda_schedule=getattr(algorithm_config, 'causal_lambda_schedule', False),
            causal_lambda_adaptive_per_agent=getattr(algorithm_config, 'causal_lambda_adaptive_per_agent', False),
            causal_lambda_adaptive_alpha=getattr(algorithm_config, 'causal_lambda_adaptive_alpha', 0.9),
            causal_lambda_adaptive_min=getattr(algorithm_config, 'causal_lambda_adaptive_min', 0.1),
            causal_lambda_adaptive_max=getattr(algorithm_config, 'causal_lambda_adaptive_max', 1.0),
            causal_lambda_adaptive_error_threshold=getattr(algorithm_config, 'causal_lambda_adaptive_error_threshold', 0.5),
            causal_lambda_start=getattr(algorithm_config, 'causal_lambda_start', 0.0),
            causal_lambda_final=getattr(algorithm_config, 'causal_lambda_final', 0.2),
            causal_lambda_warmup_steps=getattr(algorithm_config, 'causal_lambda_warmup_steps', 10000),
            causal_lambda_startingpoint=getattr(algorithm_config, 'causal_lambda_startingpoint', 0),
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            # Buffer size limits for FIFO queue behavior
            max_buffer_size=getattr(algorithm_config, 'max_buffer_size', 10000),

            device=config.training.device,
            # RKHS Kernel parameters for analytical loss
            rkhs_gamma=getattr(network_config, 'rkhs_gamma', 1.0),
            # Environment required_agents for accurate scenario classification
            env_required_agents=env_required_agents,
            # Optimization parameter
            causal_mini_batch_size=getattr(algorithm_config, 'causal_mini_batch_size', None) or 256,
            # Value network parameter sharing
            use_value_parameter_sharing=getattr(algorithm_config, 'use_value_parameter_sharing', False),
        )
    elif algorithm_type in ['pci_rnn', 'r_pci']:
        algorithm = rPCIAlgorithm(
            env=env,
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            recurrent_N=getattr(network_config, 'recurrent_N', 1),
            rnn_chunk_length=getattr(algorithm_config, 'rnn_chunk_length', 16),
            lr_actor=network_config.lr_actor,
            lr_bridge=network_config.lr_bridge,
            lr_critic=network_config.lr_critic,
            causal_lambda=algorithm_config.causal_lambda,
            clip_delta=getattr(algorithm_config, 'clip_delta', 0.0),
            causal_lambda_schedule=getattr(algorithm_config, 'causal_lambda_schedule', False),
            causal_lambda_start=getattr(algorithm_config, 'causal_lambda_start', 0.0),
            causal_lambda_final=getattr(algorithm_config, 'causal_lambda_final', 0.2),
            causal_lambda_warmup_steps=getattr(algorithm_config, 'causal_lambda_warmup_steps', 10000),
            causal_lambda_startingpoint=getattr(algorithm_config, 'causal_lambda_startingpoint', 0),
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            max_buffer_size=getattr(algorithm_config, 'max_buffer_size', 10000),
            device=config.training.device,
            rkhs_gamma=getattr(network_config, 'rkhs_gamma', 1.0),
            env_required_agents=env_required_agents,
            causal_mini_batch_size=getattr(algorithm_config, 'causal_mini_batch_size', None),
            proxy_shuffle_mode=getattr(algorithm_config, 'proxy_shuffle_mode', 'none'),
            proxy_shuffle_prob=getattr(algorithm_config, 'proxy_shuffle_prob', 0.0),
            proxy_shuffle_seed=getattr(config.training, 'seed', 0),
            proxy_shuffle_strategy=getattr(algorithm_config, 'proxy_shuffle_strategy', 'episode_axis'),
            tikhonov_gamma=getattr(algorithm_config, 'tikhonov_gamma', 1e-4),
            stabilizer_lambda=getattr(algorithm_config, 'stabilizer_lambda', 1.0),
            use_value_parameter_sharing=getattr(algorithm_config, 'use_value_parameter_sharing', False),
            use_episode_buffer=getattr(algorithm_config, 'use_episode_buffer', False),
            use_centralized_critic=getattr(algorithm_config, 'use_centralized_critic', False),
            train_naivepredictor=getattr(algorithm_config, 'train_naivepredictor', False),
            use_recurrent_policy=getattr(network_config, 'use_recurrent_policy', True),
            global_reward=getattr(algorithm_config, 'global_reward', True),
        )
    elif algorithm_type == 'mappo':
        algorithm = MAPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
            global_reward=getattr(algorithm_config, 'global_reward', True),
            use_recurrent_policy=getattr(network_config, 'use_recurrent_policy', True),
        )
    
    elif algorithm_type == 'rippo':
        algorithm = IPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
        )
    
    elif algorithm_type in ['jax_ppo', 'jax_ippo', 'jax_mappo']:
        algorithm = JAXPPOAlgorithm(
            env=env,
            num_agents=env.num_agents,
            obs_dim=base_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            batch_size=algorithm_config.batch_size,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
            use_centralized_critic=(algorithm_type == 'jax_mappo') or getattr(algorithm_config, 'use_centralized_critic', False),
            add_agent_id=getattr(algorithm_config, 'add_agent_id', True),
            use_recurrent_policy=getattr(network_config, 'use_recurrent_policy', True),
            anneal_lr=getattr(algorithm_config, 'anneal_lr', False),
            total_iterations=getattr(config.training, 'max_iterations', 0),
            seed=getattr(config.training, 'seed', 42),
        )
    
    elif algorithm_type in ['jax_rpci', 'jax_rippo', 'jax_rmappo', 'jax_rppo', 'jax_pci_ippo', 'jax_pci_mappo']:
        algorithm = JAXRPCIAlgorithm(
            env=env,
            num_agents=env.num_agents,
            obs_dim=base_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            recurrent_N=getattr(network_config, 'recurrent_N', 1),
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            lr_bridge=getattr(network_config, 'lr_bridge', network_config.lr_critic),
            lr_delta_critic=getattr(network_config, 'lr_delta_critic', None),
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            batch_size=algorithm_config.batch_size,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
            global_reward=(
                True
                if algorithm_type in ['jax_rippo', 'jax_rmappo']
                else getattr(algorithm_config, 'global_reward', True)
            ),
            add_agent_id=getattr(algorithm_config, 'add_agent_id', True),
            use_recurrent_policy=getattr(network_config, 'use_recurrent_policy', True),
            use_centralized_critic=(
                algorithm_type in ['jax_rmappo', 'jax_pci_mappo']
            ) or (
                algorithm_type not in ['jax_pci_ippo'] and getattr(algorithm_config, 'use_centralized_critic', False)
            ),
            causal_lambda=(0.0 if algorithm_type in ['jax_rippo', 'jax_rmappo', 'jax_rppo'] else getattr(algorithm_config, 'causal_lambda', 1.0)),
            causal_lambda_schedule=getattr(algorithm_config, 'causal_lambda_schedule', False),
            causal_lambda_start=getattr(algorithm_config, 'causal_lambda_start', 0.0),
            causal_lambda_final=getattr(algorithm_config, 'causal_lambda_final', getattr(algorithm_config, 'causal_lambda', 0.1)),
            causal_lambda_warmup_steps=getattr(algorithm_config, 'causal_lambda_warmup_steps', 10000),
            causal_lambda_startingpoint=getattr(algorithm_config, 'causal_lambda_startingpoint', 0),
            bridge_epoch=getattr(algorithm_config, 'bridge_epoch', 4),
            causal_mini_batch_size=getattr(algorithm_config, 'causal_mini_batch_size', None) or 10000,
            normalize_returns=getattr(algorithm_config, 'normalize_returns', True),
            clip_delta=getattr(algorithm_config, 'clip_delta', 0.0),
            tikhonov_gamma=getattr(algorithm_config, 'tikhonov_gamma', 1e-4),
            stabilizer_lambda=getattr(algorithm_config, 'stabilizer_lambda', 1.0),
            print_bridge_predictions=getattr(algorithm_config, 'print_bridge_predictions', False),
            bridge_prediction_print_samples=getattr(algorithm_config, 'bridge_prediction_print_samples', 1),
            bridge_prediction_print_interval=getattr(algorithm_config, 'bridge_prediction_print_interval', 50),
            bridge_prediction_batch_size=getattr(algorithm_config, 'bridge_prediction_batch_size', 2048),
            train_naivepredictor=getattr(algorithm_config, 'train_naivepredictor', False),
            use_naive_predictor_for_delta=getattr(algorithm_config, 'use_naive_predictor_for_delta', False),
            normalize_pci_signal=getattr(algorithm_config, 'normalize_pci_signal', False),
            anneal_lr=getattr(algorithm_config, 'anneal_lr', False),
            total_iterations=getattr(config.training, 'max_iterations', 0),
            seed=getattr(config.training, 'seed', 42),
        )
    elif algorithm_type == 'coma':
        algorithm = COMAAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
        )
    elif algorithm_type in ['coma_rnn', 'r_coma']:
        algorithm = rCOMAAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            recurrent_N=getattr(network_config, 'recurrent_N', 1),
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            td_lambda=algorithm_config.gae_lambda,
            entropy_coef=algorithm_config.entropy_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            target_update_interval=getattr(algorithm_config, 'target_update_interval', 10),
            device=config.training.device,
            use_episode_buffer=getattr(algorithm_config, 'use_episode_buffer', True),
        )
    elif algorithm_type in ['mappo_rnn', 'r_mappo']:
        algorithm = rMAPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            recurrent_N=getattr(network_config, 'recurrent_N', 1),
            rnn_chunk_length=getattr(algorithm_config, 'rnn_chunk_length', 16),
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
            use_recurrent_policy=getattr(network_config, 'use_recurrent_policy', True),
        )
    elif algorithm_type in ['ippo_rnn', 'r_ippo']:
        algorithm = rIPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=effective_obs_dim,
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            recurrent_N=getattr(network_config, 'recurrent_N', 1),
            rnn_chunk_length=getattr(algorithm_config, 'rnn_chunk_length', 16),
            lr_actor=network_config.lr_actor,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            num_batches=algorithm_config.num_batches,
            device=config.training.device,
        )
    elif algorithm_type in ['qmix', 'r_qmix', 'jax_qmix', 'jax_r_qmix']:
        epsilon_anneal_time_config = getattr(algorithm_config, 'epsilon_anneal_time', None)

        if epsilon_anneal_time_config is None:
            # Calculate expected total environment steps.
            # NOTE: total_env_steps is independent of n_rollout_threads (counts actual env interactions)
            max_iterations = config.training.max_iterations
            episodes_per_iteration = getattr(config.training, 'episodes_per_iteration', None)

            if episodes_per_iteration is not None:
                avg_episode_length = getattr(config.environment, 'max_steps', 50)
                # Do NOT multiply by n_threads - total_env_steps counts actual steps, not parallelism
                expected_steps_per_iter = episodes_per_iteration * avg_episode_length
                total_expected_steps = expected_steps_per_iter * max_iterations
            else:
                max_steps_per_iteration = config.training.max_steps_per_iteration
                # Do NOT multiply by n_threads
                total_expected_steps = max_steps_per_iteration * max_iterations

            # Anneal over first 25% of training
            epsilon_anneal_time = int(total_expected_steps * 0.25)
            print(f"[QMIX] Auto-calculated epsilon_anneal_time: {epsilon_anneal_time:,} steps "
                  f"(25% of ~{total_expected_steps:,} total env-steps)")
        else:
            epsilon_anneal_time = epsilon_anneal_time_config
            print(f"[QMIX] Using configured epsilon_anneal_time: {epsilon_anneal_time} steps")

        if algorithm_type in ['jax_qmix', 'jax_r_qmix']:
            algorithm = JAXQMIXAlgorithm(
                env=env,
                num_agents=env.num_agents,
                obs_dim=base_obs_dim,
                action_dim=env.get_action_dim(),
                hidden_dim=network_config.hidden_dim,
                lr=getattr(network_config, 'lr', getattr(network_config, 'lr_actor', 5e-4)),
                gamma=algorithm_config.gamma,
                max_grad_norm=algorithm_config.max_grad_norm,
                target_update_interval=getattr(algorithm_config, 'target_update_interval', 200),
                epsilon_start=getattr(algorithm_config, 'epsilon_start', 1.0),
                epsilon_finish=getattr(algorithm_config, 'epsilon_finish', 0.05),
                epsilon_anneal_time=epsilon_anneal_time,
                double_q=getattr(algorithm_config, 'double_q', True),
                mixing_embed_dim=getattr(algorithm_config, 'mixing_embed_dim', 32),
                hypernet_layers=getattr(algorithm_config, 'hypernet_layers', 2),
                hypernet_embed=getattr(algorithm_config, 'hypernet_embed', 64),
                mixer_init_scale=getattr(algorithm_config, 'mixer_init_scale', 1.0e-3),
                reward_scale=getattr(algorithm_config, 'reward_scale', 1.0),
                replay_buffer_size=getattr(algorithm_config, 'replay_buffer_size', 5000),
                train_batch_size=getattr(algorithm_config, 'train_batch_size', 64),
                qmix_update_epochs=getattr(algorithm_config, 'qmix_update_epochs', None),
                learning_starts=getattr(algorithm_config, 'learning_starts', 0),
                device=config.training.device,
                add_agent_id=getattr(algorithm_config, 'add_agent_id', True),
                global_reward=getattr(algorithm_config, 'global_reward', True),
                seed=getattr(config.training, 'seed', 42),
            )
        else:
            algorithm = rQMIXAlgorithm(
                num_agents=env.num_agents,
                obs_dim=effective_obs_dim,
                action_dim=env.get_action_dim(),
                hidden_dim=network_config.hidden_dim,
                lr=getattr(network_config, 'lr', 5e-4),
                gamma=algorithm_config.gamma,
                max_grad_norm=algorithm_config.max_grad_norm,
                target_update_interval=getattr(algorithm_config, 'target_update_interval', 200),
                epsilon_start=getattr(algorithm_config, 'epsilon_start', 1.0),
                epsilon_finish=getattr(algorithm_config, 'epsilon_finish', 0.05),
                epsilon_anneal_time=epsilon_anneal_time,
                double_q=getattr(algorithm_config, 'double_q', True),
                mixing_embed_dim=getattr(algorithm_config, 'mixing_embed_dim', 32),
                hypernet_layers=getattr(algorithm_config, 'hypernet_layers', 2),
                hypernet_embed=getattr(algorithm_config, 'hypernet_embed', 64),
                mixer_init_scale=getattr(algorithm_config, 'mixer_init_scale', 1.0e-3),
                reward_scale=getattr(algorithm_config, 'reward_scale', 1.0),
                replay_buffer_size=getattr(algorithm_config, 'replay_buffer_size', 5000),
                train_batch_size=getattr(algorithm_config, 'train_batch_size', 64),
                qmix_update_epochs=getattr(algorithm_config, 'qmix_update_epochs', None),
                learning_starts=getattr(algorithm_config, 'learning_starts', 0),
                device=config.training.device,
                use_episode_buffer=True,
            )
    else:
        raise ValueError(f"Invalid algorithm type: {algorithm_type}. Must be 'pci_mappo', 'r_pci', 'mappo', 'r_mappo', 'ippo', 'r_ippo', 'jax_ppo', 'jax_ippo', 'jax_mappo', 'jax_rpci', 'jax_pci_ippo', 'jax_pci_mappo', 'jax_qmix', 'jax_rippo', 'jax_rmappo', 'x_pci', 'coma', 'r_coma', 'qmix', or 'r_qmix'")
    
    return algorithm

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    if args.eval and not args.checkpoint:
        raise ValueError("--eval requires --checkpoint")
    
    # Create config
    config = create_config_from_args(args)

    # Configure JAX backend before importing JAX-dependent runtime modules.
    configure_jax_runtime(config.training.device)
    import_runtime_modules()
    
    # Set up reproducibility
    ensure_reproducible_environment()
    actual_seed = set_global_seed(config.training.seed, config.training.deterministic)
    config.training.seed = actual_seed
    
    # Set device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config.training.device = device
    
    # Get algorithm type for logger name
    algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo').upper()
    
    # R-PCI: display name based on centralized critic setting
    display_algorithm_type = algorithm_type
    if algorithm_type in ['R_PCI', 'PCI_RNN']:
        if getattr(config.algorithm, 'use_centralized_critic', False):
            display_algorithm_type = 'PCI_rMAPPO'
        else:
            display_algorithm_type = 'PCI_rIPPO'
    elif algorithm_type == 'JAX_PCI_MAPPO':
        display_algorithm_type = 'JAX_PCI_MAPPO'
    elif algorithm_type == 'JAX_PCI_IPPO':
        display_algorithm_type = 'JAX_PCI_IPPO'
    elif algorithm_type in ['JAX_QPCI', 'JAX_Q_PCI']:
        display_algorithm_type = 'JAX_qPCI'
    
    # Create logger
    logger = Logger(name=display_algorithm_type)
    logger.info(f"Start {display_algorithm_type} training")
    
    # Log reproducibility information
    log_reproducibility_info(actual_seed, logger)
    # logger.info(f"Configuration: {config.to_dict()}") # print out all config settings
    
    # Create environment
    env = None
    try:
        logger.info("Creating environment...")
        # Add seed to environment config
        env_config = config.environment
        if hasattr(env_config, '__dict__'):
            env_config.seed = actual_seed
        else:
            env_config['seed'] = actual_seed

        env_name = getattr(env_config, 'env_name', '').upper()
        n_rollout_threads = getattr(config.training, 'n_rollout_threads', 1)
        algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo').lower().replace('-', '_')

        # For SMAX batched runs and the new JAX learning path, create the vmap wrapper directly.
        # This avoids instantiating a single SMAXWrapper first, which would
        # trigger an extra warm-up/JIT compile pass before the real batch env.
        if env_name == 'SMAX' and (n_rollout_threads > 1 or algorithm_type.startswith('jax_')):
            from envs import SMAXBatchWrapper
            env = SMAXBatchWrapper(
                n_envs=max(1, n_rollout_threads),
                map_name=getattr(env_config, 'map_name', None),
                num_allies=getattr(env_config, 'num_allies', 5),
                num_enemies=getattr(env_config, 'num_enemies', 5),
                seed=actual_seed,
                max_steps=getattr(env_config, 'max_steps', 100),
                use_individual_rewards=getattr(env_config, 'use_individual_rewards', False),
                enemy_shoots=getattr(env_config, 'enemy_shoots', True),
                attack_mode=getattr(env_config, 'attack_mode', 'closest'),
                map_width=getattr(env_config, 'map_width', 32),
                map_height=getattr(env_config, 'map_height', 32),
                world_steps_per_env_step=getattr(env_config, 'world_steps_per_env_step', 8),
                time_per_step=getattr(env_config, 'time_per_step', 1.0 / 16),
                see_enemy_actions=getattr(env_config, 'see_enemy_actions', True),
                won_battle_bonus=getattr(env_config, 'won_battle_bonus', 1.0),
                walls_cause_death=getattr(env_config, 'walls_cause_death', True),
                observation_type=getattr(env_config, 'observation_type', 'unit_list'),
                smacv2_position_generation=getattr(env_config, 'smacv2_position_generation', False),
                smacv2_unit_type_generation=getattr(env_config, 'smacv2_unit_type_generation', False),
                use_allowed_targets=getattr(env_config, 'use_allowed_targets', False),
                allowed_target_window=getattr(env_config, 'allowed_target_window', None),
                allowed_targets=getattr(env_config, 'allowed_targets', None),
                use_partial_damage_restore=getattr(env_config, 'use_partial_damage_restore', False),
                use_cooperative_kill_bonus=getattr(env_config, 'use_cooperative_kill_bonus', False),
                cooperative_bonus_mode=getattr(env_config, 'cooperative_bonus_mode', 'kill'),
                cooperative_size=getattr(env_config, 'cooperative_size', 2),
                cooperative_kill_window=getattr(env_config, 'cooperative_kill_window', 2),
                cooperative_kill_bonus=getattr(env_config, 'cooperative_kill_bonus', 2.0),
                cooperative_kill_owner_ratio=getattr(env_config, 'cooperative_kill_owner_ratio', 0.7),
            )
        elif env_name in ['JUMANJILBF', 'JAXLBF', 'JAX_LBF', 'JUMANJI_LBF'] and (n_rollout_threads > 1 or algorithm_type.startswith('jax_')):
            from envs import JumanjiLBFBatchWrapper
            env = JumanjiLBFBatchWrapper(
                n_envs=max(1, n_rollout_threads),
                field_size=getattr(env_config, 'field_size', 10),
                num_agents=getattr(env_config, 'num_agents', 3),
                max_food=getattr(env_config, 'max_food', 3),
                player_level=getattr(env_config, 'player_level', (1, 2)),
                sight=getattr(env_config, 'sight', 2),
                max_episode_steps=getattr(env_config, 'max_episode_steps', 100),
                force_coop=getattr(env_config, 'force_coop', False),
                seed=actual_seed,
                grid_observation=getattr(env_config, 'grid_observation', True),
                normalize_reward=getattr(env_config, 'normalize_reward', True),
                penalty=getattr(env_config, 'penalty', 0.0),
            )
        elif env_name in ['VMASNAVIGATION', 'VMASDISPERSION'] and n_rollout_threads > 1:
            from envs import VMASNavigationEnv, VMASDispersionEnv
            wrapper_cls = VMASNavigationEnv if env_name == 'VMASNAVIGATION' else VMASDispersionEnv
            env = wrapper_cls(
                num_agents=getattr(env_config, 'num_agents', 4),
                n_envs=max(1, n_rollout_threads),
                max_steps=getattr(env_config, 'max_steps', 100),
                seed=actual_seed,
                device=device,
                continuous_actions=getattr(env_config, 'continuous_actions', False),
                **getattr(env_config, 'env_args', {})
            )
        else:
            env = create_environment(env_config)

        sync_effective_environment_config(config, env)

        logger.info(f"Environment created: {env.__class__.__name__}")
        logger.info(f"Number of agents: {env.num_agents}")
        logger.info(f"Observation dimension: {env.get_obs_dim()}")
        logger.info(f"Action dimension: {env.get_action_dim()}")

        if hasattr(env, 'print_env_setting'):
            logger.info(env.print_env_setting())
            
        # Auto-generate wandb run_name based on environment if not explicitly set
        algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo')
        
        if not config.wandb.run_name:
            env_class_name = env.__class__.__name__
            # For LBForagingWrapper, use env_id
            if env_class_name == 'LBForagingWrapper' and hasattr(env, 'env_id'):
                env_id = env.env_id
                # Remove -v3 suffix from env_id if present
                if env_id.endswith('-v3'):
                    env_id = env_id[:-3]
                config.wandb.run_name = env_id
            # For SMAC, use smac_{map_name}
            elif env_class_name in ['SMACEnv', 'SMACWrapper']:
                map_name = getattr(config.environment, 'map_name', 'unknown')
                config.wandb.run_name = f"{map_name}"
            # For SMACv2
            elif env_class_name in ['SMACv2Env', 'SMACv2Wrapper']:
                map_name = getattr(config.environment, 'map_name', 'unknown')
                races = ['protoss', 'zerg', 'terran'] # find race from the map_name, [protoss, zerg, terran]
                race = next((r for r in races if r in map_name), 'unknown')
                n_units = config.environment.capability_config["n_units"]
                n_enemies = config.environment.capability_config["n_enemies"] 
                config.wandb.run_name = f"{race}_{n_units}_vs_{n_enemies}"
            # For SMAX (JaxMARL)
            elif env_class_name in ['SMAXWrapper', 'SMAXBatchWrapper']:
                map_name = getattr(config.environment, 'map_name', None)
                post_parts = []
                if getattr(config.environment, 'use_allowed_targets', False) and getattr(config.environment, 'allowed_target_window', False):
                    post_parts.append(f"tar_w={config.environment.allowed_target_window}")
                if getattr(config.environment, 'use_cooperative_kill_bonus', False):
                    post_parts.append('c.b.')
                post = '_'.join(post_parts)
                if map_name:
                    config.wandb.run_name = f"smax_{map_name}" + (f"_{post}" if post else "")
                else:
                    n_allies = env.num_agents
                    n_enemies = env.num_enemies
                    config.wandb.run_name = f"smax_{n_allies}v{n_enemies}" + (f"_{post}" if post else "")
            elif env_name in ['JUMANJILBF', 'JAXLBF', 'JAX_LBF', 'JUMANJI_LBF']:
                config.wandb.run_name = getattr(env, 'env_id', env.__class__.__name__.lower())
            else:
                # For other environments, use environment class name
                env_name = env_class_name.lower()
                config.wandb.run_name = env_name

        # Log available environments
        available_envs = get_available_environments()
        logger.info(f"Available environments: {list(available_envs.keys())}")
        
        # Create algorithm
        logger.info("Creating algorithm...")
        algorithm = create_algorithm(env, config)
        logger.info("Algorithm created")
        
        # Create trainer
        logger.info("Creating trainer...")
        
        # Test 모드인 경우 checkpoint 저장 비활성화
        is_test_mode = hasattr(args, 'test') and args.test
        trainer_save_dir = None if is_test_mode else config.training.save_dir
        
        if is_test_mode:
            logger.info("Test mode: Model checkpoint saving disabled")
        
        # wandb run name에 알고리즘 타입 추가
        algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo')
        
        # set wandb logging name
        wandb_project_name = config.wandb.project
        wandb_run_name = config.wandb.run_name
        wandb_algo_name = algorithm_type
        if algorithm_type.lower().replace('-', '_') in ['r_pci', 'pci_rnn']:
            if getattr(config.algorithm, 'use_centralized_critic', False):
                wandb_algo_name = 'PCI_rmappo'
            else:
                wandb_algo_name = 'PCI_rippo'
        elif algorithm_type.lower().replace('-', '_') in ['jax_qpci', 'jax_q_pci']:
            wandb_algo_name = 'JAX_qpci'
        if not getattr(config.network, 'use_recurrent_policy', True):
            wandb_algo_name = f"{wandb_algo_name}_FFAgent"
        
        if wandb_run_name:
            wandb_run_name = f"{wandb_algo_name}_{wandb_run_name}_seed{actual_seed}"
        else:
            wandb_run_name = wandb_algo_name

        # Create env_fn for parallel environments
        n_rollout_threads = getattr(config.training, 'n_rollout_threads', 1)

        env_name = getattr(config.environment, 'env_name', '').upper()

        # Native batched envs (SMAX / VMAS) use in-environment vectorization instead of forked workers.
        if env_name == 'SMAX' and n_rollout_threads > 1:
            print(f"[INFO] SMAX + n_rollout_threads={n_rollout_threads}: SMAXBatchWrapper (vmap) 사용")
            env_fn = None
            _trainer_env = env
        elif env_name in ['JUMANJILBF', 'JAXLBF', 'JAX_LBF', 'JUMANJI_LBF'] and n_rollout_threads > 1:
            print(f"[INFO] {env_name} + n_rollout_threads={n_rollout_threads}: JumanjiLBFBatchWrapper native batching 사용")
            env_fn = None
            _trainer_env = env
        elif env_name in ['VMASNAVIGATION', 'VMASDISPERSION'] and n_rollout_threads > 1:
            print(f"[INFO] {env_name} + n_rollout_threads={n_rollout_threads}: VMASWrapper native batching 사용")
            # env is already created above as SMAXBatchWrapper, so reuse it.
            env_fn = None
            _trainer_env = env
        else:
            # 기존 로직: fork 기반 또는 단일 환경
            if env_name == 'SMAX':
                pass  # n_rollout_threads=1일 때는 단일 SMAXWrapper 그대로 사용
            _trainer_env = env
            env_fn = None
            if n_rollout_threads > 1:
                # Create a function that creates a new environment instance
                def env_fn(rank=0):
                    env_config_copy = config.environment
                    # Add rank to seed for diversity
                    seed = actual_seed + rank * 1000

                    if hasattr(env_config_copy, '__dict__'):
                        env_config_copy.seed = seed
                    else:
                        env_config_copy['seed'] = seed
                    return create_environment(env_config_copy)

        trainer_cls = JAXTrainer if getattr(algorithm, 'requires_jax_trainer', False) else Trainer
        trainer = trainer_cls(
            config=config,
            env=_trainer_env,
            algorithm=algorithm,
            max_iterations=config.training.max_iterations,
            max_steps_per_iteration=config.training.max_steps_per_iteration,
            episodes_per_iteration=getattr(config.training, 'episodes_per_iteration', None),
            log_interval=config.training.log_interval,
            save_interval=config.training.save_interval,
            eval_interval=config.training.eval_interval,
            eval_episodes=config.training.eval_episodes,
            save_dir=trainer_save_dir,
            use_wandb=False if is_test_mode else config.wandb.use_wandb,
            wandb_project=wandb_project_name,
            wandb_run_name=wandb_run_name,
            device=device,
            n_rollout_threads=n_rollout_threads,
            env_fn=env_fn
        )
        # Check for evaluation or test mode
        if hasattr(args, 'eval') and args.eval:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
            
            logger.info("Running evaluation...")
            eval_results = trainer.evaluate_policy()
            logger.info(f"Evaluation results: {eval_results}")
            return

        # Resume from checkpoint if provided
        if hasattr(args, 'checkpoint') and args.checkpoint:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            trainer.save_checkpoint()
            logger.info("Checkpoint saved before exit")
                    
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    except:
        raise
    finally:
        if env is not None:
            logger.info("Closing environment...")
            if hasattr(env, 'close'):
                env.close()
            logger.info("Environment closed")

if __name__ == "__main__":
    main()
