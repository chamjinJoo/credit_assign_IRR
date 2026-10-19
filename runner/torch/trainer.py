import os
import time
import numpy as np
import torch
import wandb
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from tqdm import tqdm
from datetime import datetime

from models.torch import rMAPPOAlgorithm, rIPPOAlgorithm, rPCIAlgorithm
from models.torch.common import resolve_step_episode_done
from envs import MultiAgentEnv, dict_to_tensor, create_environment
from utils import Logger, MetricsTracker
from typing import Union

class Trainer:
    """
    Main trainer for PCI-MAPPO, MAPPO, and IPPO algorithms.
    Implements the training loop following the pseudocode.
    """
    
    def __init__(
        self,
        config: Dict,
        env: MultiAgentEnv,
        algorithm: Union[rMAPPOAlgorithm, rIPPOAlgorithm, rPCIAlgorithm],
        max_iterations: int = 1000,
        max_steps_per_iteration: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        eval_episodes: int = 32,
        save_dir: str = "./runs/checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "pci-mappo",
        wandb_run_name: Optional[str] = None,
        device: str = "cpu",
        n_rollout_threads: int = 1,  # Number of parallel environments
        env_fn: Optional[callable] = None,  # Function to create environment (for parallel)
        episodes_per_iteration: Optional[int] = None  # Number of episodes to collect per iteration
    ):
        self.config = config
        self.n_rollout_threads = n_rollout_threads
        self.episodes_per_iteration = episodes_per_iteration
        self._eval_env_build_error = None
        
        # Setup environment (single or parallel)
        if n_rollout_threads > 1:
            if env_fn is not None:
                from envs.parallel_env import ParallelEnvWrapper
                self.env = ParallelEnvWrapper(env_fn, n_rollout_threads)
                self.is_parallel = True
                self.eval_env = env
            elif hasattr(env, 'n_envs') and getattr(env, 'n_envs', 1) > 1:
                self.env = env
                self.is_parallel = True
                self.eval_env = self._build_eval_env(env)
            else:
                raise ValueError(
                    "n_rollout_threads > 1 requires env_fn or SMAXBatchWrapper."
                )
        else:
            # Use single environment
            self.env = env
            self.eval_env = env
            self.is_parallel = False
        
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.max_steps_per_iteration = max_steps_per_iteration
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.base_save_dir = save_dir
        self.use_wandb = use_wandb
        self.device = device

        # Store reproducibility info
        self.seed = getattr(config.training, 'seed', None) if hasattr(config, 'training') else None
        self.deterministic = getattr(config.training, 'deterministic', True) if hasattr(config, 'training') else True

        # Agent ID feature configuration
        self.add_agent_id = getattr(config.algorithm, 'add_agent_id', False) if hasattr(config, 'algorithm') else False
        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else len(env.agents)
        
        # Initialize logging first with algorithm-specific name
        algorithm_type = type(algorithm).__name__.replace('Algorithm', '').upper()
        self.logger = Logger(name=algorithm_type)
        self.metrics_tracker = MetricsTracker()
        if self.eval_env is not self.env:
            self.logger.info(
                f"Created dedicated evaluation environment: {self.eval_env.__class__.__name__}"
            )
        elif self._eval_env_build_error is not None:
            self.logger.warning(
                f"Falling back to shared evaluation environment: {self._eval_env_build_error}"
            )
        
        # Initialize causal effect evaluator (only for PCI-MAPPO)
        # For other algorithms, these will be None
        self.causal_evaluator = None
        self.advanced_causal_evaluator = None
       
        # Create structured save directory for this simulation (only if save_dir is provided)
        if save_dir is not None:
            self.save_dir = self._create_simulation_directory()
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save simulation metadata
            self._save_simulation_metadata()
        else:
            self.save_dir = None
        
        # Initialize wandb if requested
        if use_wandb:
            serialized_config = self._get_serialized_config()
            # Get algorithm type for config
            algorithm_type = type(self.algorithm).__name__.replace('Algorithm', '').lower()
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=serialized_config
            )
            
            # Define metrics for better visualization
            wandb.define_metric("total_steps")
            wandb.define_metric("causal_effect/*", step_metric="total_steps")
            wandb.define_metric("rollout/*", step_metric="total_steps")
            wandb.define_metric("update/*", step_metric="total_steps")
            wandb.define_metric("eval/*", step_metric="total_steps")
            
            # Add summary information for causal effect metrics
            wandb.summary.update({
                "causal_effect_description": "Delta values for different reward scenarios in StagHunt environment",
                "reward_scenarios": {
                    "no_reward": "No hare hunt occurred and the focal agent did not participate in its assigned stag hunt",
                    "focal_forage_reward": "The focal agent successfully hunted a hare",
                    "forage_reward": "Another agent successfully hunted a hare",
                    "stag_reward": "The focal agent participated in its assigned stag hunt"
                },
                "reproducibility": {
                    "seed": getattr(self, 'seed', 'unknown'),
                    "deterministic": getattr(self, 'deterministic', 'unknown')
                }
            })
        
        # Training state
        self.iteration = 0
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
        # Episode statistics
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.recent_rewards = deque(maxlen=100)

    def _build_eval_env(self, env: MultiAgentEnv) -> MultiAgentEnv:
        """Prefer a dedicated single-environment evaluator for SMAX batched runs."""
        if env.__class__.__name__ != "SMAXBatchWrapper":
            return env

        env_config = getattr(self.config, 'environment', None)
        if env_config is None:
            return env

        try:
            return create_environment(env_config)
        except Exception as exc:
            self._eval_env_build_error = str(exc)
            return env

    def _get_serialized_config(self) -> Dict[str, Any]:
        """Return config data safe for metadata persistence and wandb logging."""
        if hasattr(self.config, 'to_dict'):
            try:
                return self.config.to_dict(filter_environment=True)
            except TypeError:
                config_dict = self.config.to_dict()
        elif isinstance(self.config, dict):
            config_dict = dict(self.config)
        else:
            return self.config

        env_config = getattr(self.config, 'environment', None)
        if env_config is not None and hasattr(env_config, 'get_relevant_config'):
            config_dict['environment'] = env_config.get_relevant_config()

        return config_dict

    def _get_eval_episode_limit(self) -> int:
        """Infer a reasonable evaluation horizon from the environment."""
        max_steps = getattr(self.eval_env, 'max_steps', None)
        if max_steps is not None:
            return max(1, int(max_steps))

        get_env_info = getattr(self.eval_env, 'get_env_info', None)
        if callable(get_env_info):
            try:
                env_info = get_env_info()
                episode_limit = env_info.get('episode_limit')
                if episode_limit is not None:
                    return max(1, int(episode_limit))
            except Exception:
                pass

        max_episode_steps = getattr(self.eval_env, 'max_episode_steps', None)
        if max_episode_steps is not None:
            return max(1, int(max_episode_steps))

        return 1000

    @staticmethod
    def _is_eval_only_advantage_metric(key: str) -> bool:
        """PCI advantage diagnostics are logged only alongside evaluation snapshots."""
        advantage_prefixes = (
            "pci_raw_adv_",
            "pci_res_adv_",
            "pci_bonus_abs_",
        )
        advantage_keys = {
            "env_adv_std",
            "pci_adv_res_std",
            "pci_bonus_std",
            "combined_adv_std",
        }
        return key in advantage_keys or key.startswith(advantage_prefixes)

    def _get_eval_only_advantage_metrics(self, update_info: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not update_info:
            return {}
        return {
            key: value
            for key, value in update_info.items()
            if self._is_eval_only_advantage_metric(key)
        }

    def _log_update_metric_line(self, key: str, value: float, update_info: Dict[str, float], indent: str = "    "):
        """Pretty-print update metrics, attaching counts to paired mean values when available."""
        if key.endswith('_count'):
            return

        if key.startswith('delta_') and key.endswith('_mean'):
            scenario = key.replace('delta_', '').replace('_mean', '')
            count_key = f'delta_{scenario}_count'
            if count_key in update_info:
                count = update_info[count_key]
                self.logger.info(f"{indent}{key}: {value:.6f} (n={count:g})")
                return

        if self._is_eval_only_advantage_metric(key) and key.endswith('_mean'):
            count_key = key.replace('_mean', '_count')
            if count_key in update_info:
                count = update_info[count_key]
                self.logger.info(f"{indent}{key}: {value:.6f} (n={count:g})")
                return

        self.logger.info(f"{indent}{key}: {value:.6f}")
    
    def _create_simulation_directory(self) -> str:
        """Create structured directory for this simulation."""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add seed to directory name if available
        seed_str = f"_seed{self.seed}" if self.seed is not None else ""
        
        # Get environment name
        env_name = self.eval_env.__class__.__name__.lower()
        
        # Create directory structure: base_dir/timestamp_seed/env_name/checkpoints
        sim_dir = os.path.join(
            self.base_save_dir,
            f"{timestamp}{seed_str}",
            env_name,
            "checkpoints"
        )
        
        return sim_dir
    
    def _save_simulation_metadata(self):
        """Save simulation configuration and metadata."""
        import json
        def convert_to_python_type(obj):
            """Convert numpy types and other non-JSON-serializable types to Python types."""
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_python_type(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python_type(item) for item in obj]
            else:
                return obj
        
        # Create metadata directory
        metadata_dir = os.path.dirname(self.save_dir)  # Remove 'checkpoints' from path
        
        # Prepare full config with filtered environment config
        config_dict = self._get_serialized_config()
             
        metadata = {
            "simulation_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.eval_env.__class__.__name__,
                "num_agents": self.env.num_agents,
                "obs_dim": self.env.get_obs_dim(),
                "action_dim": self.env.get_action_dim()
            },
            
            "config": config_dict,
            
            # "directories": {
            #     "base_save_dir": self.base_save_dir,
            #     "simulation_dir": self.save_dir,
            #     "checkpoint_dir": self.save_dir
            # }
        }
        
        # Save metadata
        metadata = convert_to_python_type(metadata)
        metadata_path = os.path.join(metadata_dir, "simulation_config.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple readme
        readme_path = os.path.join(metadata_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"PCI-MAPPO Training Simulation\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {self.eval_env.__class__.__name__}\n")
            f.write(f"Agents: {self.env.num_agents}\n")
            f.write(f"Max Iterations: {self.max_iterations}\n")
            f.write(f"Steps per Iteration: {self.max_steps_per_iteration}\n")
            f.write(f"Total Steps: {self.max_iterations * self.max_steps_per_iteration:,}\n\n")
            f.write(f"Directory Structure:\n")
            f.write(f"  checkpoints/     - Model checkpoints\n")
            f.write(f"  logs/           - Training logs (if enabled)\n")
            f.write(f"  results/        - Evaluation results (if enabled)\n")
        self.logger.info(f"Simulation metadata saved to: {metadata_dir}")
        self.logger.info(f"Checkpoint directory: {self.save_dir}")
    
    def _save_training_summary(self, total_time: float, final_eval: Dict[str, float]):
        """Save final training summary and results."""
        import json
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(self.save_dir), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Training summary
        summary = {
            "training_completed": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "iterations_completed": self.max_iterations,
            "total_steps": self.max_iterations * self.max_steps_per_iteration,
            "average_time_per_iteration": total_time / self.max_iterations,
            "best_eval_reward": self.best_eval_reward,
            "final_evaluation": final_eval
        }
        
        # Save summary
        summary_path = os.path.join(results_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save training metrics (if available)
        if hasattr(self, 'metrics_tracker') and hasattr(self.metrics_tracker, 'get_all_metrics'):
            metrics_path = os.path.join(results_dir, "training_metrics.json")
            try:
                metrics = self.metrics_tracker.get_all_metrics()
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            except:
                pass  # Skip if metrics not available
        
        self.logger.info(f"Training summary saved to: {results_dir}")
        self.logger.info(f"Final best reward: {self.best_eval_reward:.4f}")
        
    def train(self):
        """Main training loop following PCI-MAPPO pseudocode."""
        # Print training configuration
        algorithm_type = type(self.algorithm).__name__.replace('Algorithm', '').upper()
        self.logger.info("=" * 60)
        self.logger.info(f"{algorithm_type} Training Configuration")
        self.logger.info("=" * 60)
        self.logger.info(f"Environment: {self.eval_env.__class__.__name__}")
        self.logger.info(f"Number of agents: {self.env.num_agents}")
        self.logger.info(f"Observation dimension: {self.env.get_obs_dim()}")
        self.logger.info(f"Action dimension: {self.env.get_action_dim()}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        self.logger.info(f"Steps per iteration: {self.max_steps_per_iteration}")
        self.logger.info(f"Total steps: {self.max_iterations * self.max_steps_per_iteration:,}")
        self.logger.info(f"Log interval: {self.log_interval}")
        self.logger.info(f"Eval interval: {self.eval_interval}")
        self.logger.info(f"Save interval: {self.save_interval}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Wandb: {self.use_wandb}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use tqdm for progress tracking
        with tqdm(total=self.max_iterations, desc="Training Progress", unit="iter") as pbar:
            for iteration in range(self.max_iterations):
                self.iteration = iteration
                # Data Collection Phase
                rollout_data = self.collect_rollouts()
                self.total_steps += rollout_data['step_count']

                # Training Phase
                update_info = self.algorithm.update(
                    num_episodes_collected=rollout_data.get('episode_count', 1)
                )
                
                # Log metrics (only for logging, not displayed in tqdm)
                if iteration % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    self.log_training_metrics(rollout_data, update_info)
                
                # Evaluate policy
                if iteration % self.eval_interval == 0 or iteration == self.max_iterations - 1:
                    eval_info = self.evaluate_policy(num_episodes=self.eval_episodes)
                    self.log_evaluation_metrics(eval_info, update_info=update_info)
                
                # Save checkpoint
                if iteration % self.save_interval == 0 or iteration == self.max_iterations - 1:
                    self.save_checkpoint()
                
                # Update progress bar
                pbar.update(1)
        
        total_time = time.time() - start_time
        
        # Print final results
        self.logger.info("=" * 60)
        self.logger.info("Training Completed!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        self.logger.info(f"Average time per iteration: {total_time/self.max_iterations:.2f} seconds")
        self.logger.info(f"Total iterations completed: {self.max_iterations}")
        self.logger.info(f"Total steps collected: {self.max_iterations * self.max_steps_per_iteration:,}")
        
        # Final evaluation and save
        self.logger.info("Running final evaluation...")
        final_eval = self.evaluate_policy(num_episodes=self.eval_episodes)
        self.log_evaluation_metrics(final_eval, title="Final Evaluation Results")
        self.save_checkpoint(is_final=True)
        
        # Save final training summary
        self._save_training_summary(total_time, final_eval)
        
        if self.use_wandb:
            wandb.finish()

    def _add_agent_id_to_obs(self, observations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Add agent_id one-hot vector to each agent's observation.

        Args:
            observations: Dictionary mapping agent_id to observation tensor

        Returns:
            Dictionary mapping agent_id to observation with agent_id one-hot concatenated
        """
        if not self.add_agent_id:
            return observations

        processed_obs = {}
        for agent_id, obs in observations.items():
            # Create one-hot encoding for this agent
            leading_shape = obs.shape[:-1]
            agent_id_onehot = torch.zeros(*leading_shape, self.num_agents, device=obs.device, dtype=obs.dtype)
            agent_id_onehot[..., agent_id] = 1.0
            # Concatenate to observation
            processed_obs[agent_id] = torch.cat([obs, agent_id_onehot], dim=-1)

        return processed_obs

    def _add_agent_id_to_obs_batch(self, observations_batch: List[Dict[int, torch.Tensor]]) -> List[Dict[int, torch.Tensor]]:
        """
        Add agent_id one-hot vector to each agent's observation in batch.

        Args:
            observations_batch: List of dictionaries mapping agent_id to observation tensor

        Returns:
            List of dictionaries mapping agent_id to observation with agent_id one-hot concatenated
        """
        if not self.add_agent_id:
            return observations_batch

        return [self._add_agent_id_to_obs(obs_dict) for obs_dict in observations_batch]

    def collect_rollouts(self) -> Dict[str, Any]:
        """
        Collect rollouts for one iteration.
        
        Returns:
            Dictionary containing rollout statistics
        """
        if self.is_parallel:
            return self._collect_rollouts_parallel()
        else:
            return self._collect_rollouts_single()
    
    def _collect_rollouts_single(self) -> Dict[str, Any]:
        """
        Collect rollouts from a single environment (original implementation).
        
        Returns:
            Dictionary containing rollout statistics
        """
        step_count = 0
        episode_count = 0
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        
        # Determine loop condition
        use_ep_buffer = getattr(self.algorithm, 'use_episode_buffer', False)
        use_ep_limit = use_ep_buffer and self.episodes_per_iteration is not None
        
        while (episode_count < self.episodes_per_iteration if use_ep_limit else step_count < self.max_steps_per_iteration):
            # Reset environment and agents
            observations = self.env.reset()
            self.algorithm.reset_agents()
            # Convert observations to tensors
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            # if enabled, add agent ID to observations
            observations = self._add_agent_id_to_obs(observations)

            # Initialize previous components (for first step, use zero vector since no previous observation exists)
            prev_observations = {agent_id: torch.zeros_like(obs) for agent_id, obs in observations.items()}
            prev_rewards = {agent_id: 0.0 for agent_id in observations.keys()}
            prev_actions = {agent_id: 0 for agent_id in observations.keys()}

            episode_reward = defaultdict(float)
            episode_length = 0
            done = False

            # Inner loop: Run until episode is done.
            # If NOT using episode limit (i.e. using step limit), also break if max steps reached.
            while not done and (use_ep_limit or step_count < self.max_steps_per_iteration):
                # Get available actions
                available_actions = None
                if hasattr(self.env, "get_avail_actions"):
                    avail = self.env.get_avail_actions()
                    # SMAC returns List[List[int]] where index is agent_id
                    available_actions = {i: avail[i] for i in range(len(avail))}

                # Get actions from agents
                actions = self.algorithm.get_actions(observations, available_actions)
                
                # Convert actions for environment
                env_actions = {agent_id: action for agent_id, (action, _) in actions.items()}
                
                # Take step in environment
                next_observations, rewards, dones, info = self.env.step(env_actions)

                # Convert to tensors
                next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)
                # Add agent ID to observations if enabled
                next_observations = self._add_agent_id_to_obs(next_observations)

                # Store transition with proper causal structure (o_{t-1}, o_t, a_t, r_t, o_{t+1})
                self.algorithm.store_transition(
                    prev_observations=prev_observations,  # o_{t-1}
                    prev_rewards=prev_rewards,            # r_{t-1} (z_t)
                    prev_actions=prev_actions,            # a_{t-1} (z_t)
                    observations=observations,            # o_t (w_t)
                    actions=env_actions,                  # a_t
                    rewards=rewards,                      # r_t
                    next_observations=next_observations,  # o_{t+1}
                    dones=dones,
                    info=info,  # Environment info (e.g., stag_hunts, hare_hunts)
                    available_actions=available_actions # [ADDED]
                )
                
                # Update episode statistics
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                
                # Update state: prev_observations = current observations, observations = next_observations
                prev_observations = observations
                prev_rewards = rewards
                prev_actions = env_actions
                observations = next_observations
                episode_length += 1
                step_count += 1
                
                # Check if episode is done
                done = any(dones.values()) or episode_length >= 1000
            
            if use_ep_buffer and done:
                self.algorithm.finish_episode()
            # else: Incomplete episode remains in current_episode for next iteration
            
            # Store episode statistics
            for agent_id, total_reward in episode_reward.items():
                episode_rewards[agent_id].append(total_reward)
            
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Check success condition for StagHunt
            is_success = 0.0
            if self.env.__class__.__name__ == "StagHuntEnv":
                stag_alive = info.get('stag_alive', {})
                hare_alive = info.get('hare_alive', {})
                # Success if all objects (stags and hares) are hunted (not alive)
                if not any(stag_alive.values()) and not any(hare_alive.values()):
                    is_success = 1.0
            episode_successes.append(is_success)
            
            # Track recent rewards for logging
            mean_episode_reward = np.mean(list(episode_reward.values()))
            self.recent_rewards.append(mean_episode_reward)
        
        # Compile rollout data
        rollout_data = {
            'step_count': step_count,
            'episode_count': episode_count,
            'episode_rewards': dict(episode_rewards),
            'episode_lengths': episode_lengths,
            'mean_episode_reward': np.mean([np.mean(rewards) for rewards in episode_rewards.values()]),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_success_rate': np.mean(episode_successes) if episode_successes else 0.0
        }
        
        return rollout_data
    
    def _collect_rollouts_parallel(self) -> Dict[str, Any]:
        """
        Collect rollouts from parallel environments.
        
        Returns:
            Dictionary containing rollout statistics
        """
        import torch
        import numpy as np
        from envs import dict_to_tensor
        
        def _batch_dict_to_tensor(dict_list, device):
            """
            Optimized helper to batch CPU->GPU transfers across parallel environments.
            Minimizes the number of GPU transfers by stacking all data first.
            """
            if not dict_list:
                return []

            tensor_dict_list = [{} for _ in range(len(dict_list))]

            # Collect all unique keys
            all_keys = set()
            for d in dict_list:
                all_keys.update(d.keys())

            # Process each key
            for key in all_keys:
                values = []
                valid_indices = []

                # Collect values for this key
                for i, d in enumerate(dict_list):
                    if key in d:
                        valid_indices.append(i)
                        values.append(d[key])

                if not values:
                    continue

                # Optimize: Single CPU→GPU transfer per key
                if isinstance(values[0], np.ndarray):
                    # Stack on CPU first, then single GPU transfer
                    stacked = np.stack(values)
                    tensor_batch = torch.from_numpy(stacked).float().to(device, non_blocking=True)
                elif isinstance(values[0], (int, float)):
                    # Direct tensor creation on GPU
                    tensor_batch = torch.tensor(values, dtype=torch.float32, device=device)
                else:
                    # Fallback for other types
                    try:
                        tensor_batch = torch.tensor(values, device=device)
                    except (ValueError, TypeError):
                        # Last resort: individual transfers (slower)
                        tensor_batch = [torch.tensor(v, device=device) for v in values]

                # Distribute to dictionary list
                if isinstance(tensor_batch, list):
                    for idx, tensor_val in zip(valid_indices, tensor_batch):
                        tensor_dict_list[idx][key] = tensor_val
                else:
                    for i, idx in enumerate(valid_indices):
                        tensor_dict_list[idx][key] = tensor_batch[i]

            return tensor_dict_list
        
        step_count = 0
        episode_count = 0
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        
        # Reset all environments
        # [OPTIMIZED] reset() now also returns avail_actions, eliminating a separate IPC call
        observations_batch, available_actions_batch = self.env.reset()
        self.algorithm.reset_agents_batch(self.n_rollout_threads)

        observations_batch = _batch_dict_to_tensor(observations_batch, self.algorithm.device)
        # if enabled, Add agent ID to observations
        observations_batch = self._add_agent_id_to_obs_batch(observations_batch)

        # Initialize previous 1) obs, 2) rewards, 3) actions for each environment
        prev_observations_batch = [
            {agent_id: torch.zeros_like(obs) for agent_id, obs in obs_dict.items()}
            for obs_dict in observations_batch
        ]
        prev_rewards_batch = [
            {agent_id: 0.0 for agent_id in obs_dict.keys()}
            for obs_dict in observations_batch
        ]
        prev_actions_batch = [
            {agent_id: 0 for agent_id in obs_dict.keys()}
            for obs_dict in observations_batch
        ]
        
        # Track episode state for each environment
        env_episode_rewards = [{agent_id: 0.0 for agent_id in range(self.env.num_agents)} for _ in range(self.n_rollout_threads)]
        env_episode_lengths = [0] * self.n_rollout_threads
        env_dones = [False] * self.n_rollout_threads
        
        # Determine loop condition
        use_ep_buffer = getattr(self.algorithm, 'use_episode_buffer', False)
        use_ep_limit = use_ep_buffer and self.episodes_per_iteration is not None

        while (episode_count < self.episodes_per_iteration if use_ep_limit else step_count < self.max_steps_per_iteration):
            # [OPTIMIZED] avail_actions now come from the previous step()/reset() result —
            # no separate get_avail_actions_batch() IPC call needed here.
            # Get actions for all environments
            actions_batch = self.algorithm.get_actions_batch(
                observations_batch,
                available_actions_batch
            )
            
            # Convert actions for environment (remove log_probs)
            # actions_batch is List[Dict[agent_id, (action, log_prob)]]
            env_actions_batch = []
            for env_idx in range(self.n_rollout_threads):
                env_actions = {}
                for agent_id in range(self.env.num_agents):
                    # actions_batch[env_idx] is Dict[agent_id, (action, log_prob)]
                    action_tuple = actions_batch[env_idx][agent_id]
                    if isinstance(action_tuple, tuple):
                        action, log_prob = action_tuple
                    else: # Fallback if not a tuple
                        action = action_tuple
                    env_actions[agent_id] = int(action)  # Ensure it's an int
                env_actions_batch.append(env_actions)
            
            # Step all environments
            # [OPTIMIZED] step() now returns avail_actions for the NEXT step as the 5th element
            next_observations_batch, rewards_batch, dones_batch, infos_batch, next_available_actions_batch = self.env.step(env_actions_batch)
            next_observations_batch = _batch_dict_to_tensor(next_observations_batch, self.algorithm.device)
            # Add agent ID to observations if enabled
            next_observations_batch = self._add_agent_id_to_obs_batch(next_observations_batch)

            # Store transitions for each environment
            for env_idx in range(self.n_rollout_threads):
                if not env_dones[env_idx]:
                    # Store transition
                    self.algorithm.store_transition_from_env(
                        env_idx=env_idx,
                        prev_observations=prev_observations_batch[env_idx],
                        prev_rewards=prev_rewards_batch[env_idx],
                        prev_actions=prev_actions_batch[env_idx],
                        observations=observations_batch[env_idx],
                        actions=actions_batch[env_idx],
                        rewards=rewards_batch[env_idx],
                        next_observations=next_observations_batch[env_idx],
                        dones=dones_batch[env_idx],
                        info=infos_batch[env_idx],
                        available_actions=available_actions_batch[env_idx] if available_actions_batch[env_idx] is not None else None
                    )
                    
                    # Update episode statistics
                    for agent_id, reward in rewards_batch[env_idx].items():
                        env_episode_rewards[env_idx][agent_id] += reward
                    
                    env_episode_lengths[env_idx] += 1
                    step_count += 1
                    
                    # Check if episode is done
                    if any(dones_batch[env_idx].values()):
                        env_dones[env_idx] = True
                        
                        # Store episode statistics
                        for agent_id, total_reward in env_episode_rewards[env_idx].items():
                            episode_rewards[agent_id].append(total_reward)
                        
                        episode_lengths.append(env_episode_lengths[env_idx])
                        episode_count += 1
                        
                        # Check success
                        is_success = 0.0
                        info_ = infos_batch[env_idx]
                        # for staghunt env
                        if 'stag_alive' in info_:
                            stag_alive = info_.get('stag_alive', {})
                            hare_alive = info_.get('hare_alive', {})
                            if not any(stag_alive.values()) and not any(hare_alive.values()):
                                is_success = 1.0
                        # for SMACs
                        elif info_.get('battle_won', False):
                            is_success = 1.0
                        episode_successes.append(is_success)
                        
                        # Track recent rewards
                        mean_episode_reward = np.mean(list(env_episode_rewards[env_idx].values()))
                        self.recent_rewards.append(mean_episode_reward)
                        
                        # Reset episode tracking for this environment
                        env_episode_rewards[env_idx] = {agent_id: 0.0 for agent_id in range(self.env.num_agents)}
                        env_episode_lengths[env_idx] = 0
                        env_dones[env_idx] = False
            
            # Update states
            for env_idx in range(self.n_rollout_threads):
                episode_done = any(dones_batch[env_idx].values())
                if episode_done:
                    prev_observations_batch[env_idx] = {
                        agent_id: torch.zeros_like(next_observations_batch[env_idx][agent_id])
                        for agent_id in next_observations_batch[env_idx].keys()
                    }
                    prev_rewards_batch[env_idx] = {agent_id: 0.0 for agent_id in next_observations_batch[env_idx].keys()}
                    prev_actions_batch[env_idx] = {agent_id: 0 for agent_id in next_observations_batch[env_idx].keys()}
                else:
                    prev_observations_batch[env_idx] = observations_batch[env_idx]
                    prev_rewards_batch[env_idx] = rewards_batch[env_idx]
                    prev_actions_batch[env_idx] = env_actions_batch[env_idx]

            observations_batch = next_observations_batch
            # [OPTIMIZED] carry forward the avail_actions from this step
            available_actions_batch = next_available_actions_batch
        
        # Compile rollout data
        rollout_data = {
            'step_count': step_count,
            'episode_count': episode_count,
            'episode_rewards': dict(episode_rewards),
            'episode_lengths': episode_lengths,
            'mean_episode_reward': np.mean([np.mean(rewards) for rewards in episode_rewards.values()]) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'mean_success_rate': np.mean(episode_successes) if episode_successes else 0.0
        }
        
        return rollout_data
 
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy for multiple episodes.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        eval_rewards = defaultdict(list)
        eval_lengths = []
        eval_successes = []
        episode_limit = self._get_eval_episode_limit()

        is_batch_env = hasattr(self.eval_env, 'n_envs') and getattr(self.eval_env, 'n_envs', 1) > 1

        for episode in range(num_episodes):
            # Reset environment and agents (use eval_env for evaluation)
            observations = self.eval_env.reset()

            if is_batch_env:
                obs_list, avail_list_reset = observations
                observations = obs_list[0]           # Dict[int, np.ndarray]
                _eval_avail_cache = avail_list_reset[0]  # Dict[int, List[int]]
            else:
                _eval_avail_cache = None

            self.algorithm.reset_agents()
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            # Add agent ID to observations if enabled
            observations = self._add_agent_id_to_obs(observations)

            episode_reward = defaultdict(float)
            episode_success = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < episode_limit:
                # Get availability of actions
                available_actions = None
                if is_batch_env:
                    # SMAXBatchWrapper: use cached availability from the first env.
                    if _eval_avail_cache is not None:
                        available_actions = _eval_avail_cache
                elif hasattr(self.eval_env, "get_avail_actions"):
                    avail = self.eval_env.get_avail_actions()
                    # Convert to Dict[int, List[int]]
                    # SMAC returns List[List[int]] where index is agent_id
                    if avail is not None:
                        available_actions = {i: avail[i] for i in range(len(avail))}

                # Get actions (no exploration for evaluation)
                with torch.no_grad():
                    actions = {}
                    for agent_id, obs in observations.items():
                        # Handle available actions for agent
                        avail_actions_tensor = None
                        if available_actions is not None and agent_id in available_actions:
                            avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.algorithm.device)
                            
                        if hasattr(self.algorithm.agents[agent_id], 'get_action'):
                             if avail_actions_tensor is not None:
                                 action, _, = self.algorithm.agents[agent_id].get_action(obs, available_actions=avail_actions_tensor)
                             else:
                                 action, _, = self.algorithm.agents[agent_id].get_action(obs)
                        else:
                             # Fallback
                             action, _, = self.algorithm.agents[agent_id].get_action(obs)

                        actions[agent_id] = action
                
                # Take step
                if is_batch_env:
                    # SMAXBatchWrapper: send actions to the first env.
                    actions_batch = [actions] + [
                        {i: 0 for i in range(self.eval_env.num_agents)}
                        for _ in range(self.eval_env.n_envs - 1)
                    ]
                    step_result = self.eval_env.step(actions_batch)
                    next_obs_list, rew_list, done_list, info_list, avail_list_next = step_result
                    next_observations = next_obs_list[0]
                    rewards = rew_list[0]
                    dones = done_list[0]
                    info = info_list[0]
                    _eval_avail_cache = avail_list_next[0]
                else:
                    next_observations, rewards, dones, info = self.eval_env.step(actions)

                next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)
                # Add agent ID to observations if enabled
                next_observations = self._add_agent_id_to_obs(next_observations)

                # Check success (Evaluate)
                if self.eval_env.__class__.__name__ in ("StagHuntEnv",):
                    stag_alive = info.get('stag_alive', {})
                    hare_alive = info.get('hare_alive', {})
                    if not any(stag_alive.values()) and not any(hare_alive.values()):
                        episode_success = 1.0
                        
                if info.get('battle_won', False): # for SMAC, SMACv2, SMAX
                    episode_success = 1.0

                # Update statistics
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                
                observations = next_observations
                episode_length += 1
                done = resolve_step_episode_done(dones, info)

            if not done and episode_length >= episode_limit:
                self.logger.warning(
                    f"Evaluation episode {episode + 1}/{num_episodes} reached the "
                    f"episode limit ({episode_limit}) without an explicit done signal."
                )
            
            # Store episode data
            for agent_id, total_reward in episode_reward.items():
                eval_rewards[agent_id].append(total_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(episode_success)
        
        # Compile evaluation metrics
        eval_info = {
            'eval_success_rate': np.mean(eval_successes) if eval_successes else 0.0,
            'eval_mean_reward': np.mean([np.mean(rewards) for rewards in eval_rewards.values()]),
            'eval_std_reward': np.std([np.mean(rewards) for rewards in eval_rewards.values()]),
            'eval_mean_length': np.mean(eval_lengths),
            'eval_min_reward': np.min([np.mean(rewards) for rewards in eval_rewards.values()]),
            'eval_max_reward': np.max([np.mean(rewards) for rewards in eval_rewards.values()])
        }
        
        # Update best reward
        if eval_info['eval_mean_reward'] > self.best_eval_reward:
            self.best_eval_reward = eval_info['eval_mean_reward']
            eval_info['is_best'] = True
        else:
            eval_info['is_best'] = False
        
        # Perform causal effect evaluation (only for PCI-MAPPO)
        if self.causal_evaluator is not None:
            try:
                causal_stats = self.causal_evaluator.evaluate_causal_effects(num_eval_steps=1000)
                eval_info['causal_effects'] = causal_stats
                
                # Log causal effect statistics
                self.logger.info("Causal Effect Statistics:")
                for scenario, stats in causal_stats.items():
                    self.logger.info(f"  {scenario}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to perform causal effect evaluation: {e}")
                eval_info['causal_effects'] = None
        else:
            eval_info['causal_effects'] = None
        
        return eval_info
    
    def log_training_metrics(self, rollout_data: Dict[str, Any], update_info: Dict[str, float]):
        """Log training metrics."""
        # Log to console
        self.logger.info(f"Iteration {self.iteration}")
        
        # Calculate steps to show
        steps_val = rollout_data['step_count']
        if getattr(self.algorithm, 'use_episode_buffer', False):
            # For EpisodeBuffer, show effective steps (sum of lengths) as primary
            effective_steps = sum(rollout_data['episode_lengths'])
            
            # effective: step data reflected on training (excluding zero-padded data)  
            # env steps: step data containing zero-padding
            self.logger.info(f"  Effective Steps: {effective_steps} (Full Steps: {steps_val})")
        else:
            self.logger.info(f"  Steps: {steps_val}")
        self.logger.info(f"  Episodes: {rollout_data['episode_count']}")
        self.logger.info(f"  Mean Episode Reward: {rollout_data['mean_episode_reward']:.3f}")
        self.logger.info(f"  Mean Episode Length: {rollout_data['mean_episode_length']:.1f}")
        if 'mean_success_rate' in rollout_data:
            self.logger.info(f"  Mean Success Rate: {rollout_data['mean_success_rate']:.3f}")
        
        if update_info:
            self.logger.info("  Update Info:")
            # Log epsilon first for QMIX (for easy visibility)
            if 'epsilon' in update_info:
                self.logger.info(f"    epsilon: {update_info['epsilon']:.4f}")
            if 'total_env_steps' in update_info:
                self.logger.info(f"    total_env_steps: {update_info['total_env_steps']}")
            if 'replay_buffer_episodes' in update_info:
                self.logger.info(f"    replay_buffer_episodes: {update_info['replay_buffer_episodes']}")

            for key, value in update_info.items():
                # Skip already logged keys
                if key in ['epsilon', 'total_env_steps', 'replay_buffer_episodes']:
                    continue
                if self._is_eval_only_advantage_metric(key):
                    continue
                self._log_update_metric_line(key, value, update_info)

        bridge_print_interval = max(
            1,
            int(getattr(self.algorithm, "bridge_prediction_print_interval", 50)),
        )
        if (
            hasattr(self.algorithm, "consume_bridge_prediction_debug_lines")
            and self.iteration % bridge_print_interval == 0
        ):
            bridge_debug_lines = self.algorithm.consume_bridge_prediction_debug_lines()
            if bridge_debug_lines:
                self.logger.info("  Bridge Prediction Debug:")
                for line in bridge_debug_lines:
                    self.logger.info(f"    {line}")
        
        # Log to wandb
        if self.use_wandb:
            wandb_data = {
                'iteration': self.iteration,
                'total_steps': self.total_steps,
                'rollout/mean_episode_reward': rollout_data['mean_episode_reward'],
                'rollout/mean_episode_length': rollout_data['mean_episode_length'],
                'rollout/mean_success_rate': rollout_data.get('mean_success_rate', 0.0),
                'rollout/episodes': rollout_data['episode_count']
            }
            
            # Add agent-specific rewards
            for agent_id, rewards in rollout_data['episode_rewards'].items():
                wandb_data[f'rollout/agent_{agent_id}_mean_reward'] = np.mean(rewards)
            
            # Add update info
            for key, value in update_info.items():
                if self._is_eval_only_advantage_metric(key):
                    continue
                wandb_data[f'update/{key}'] = value
            
            # Add delta statistics for causal reward shaping
            # Group by agent and scenario for better visualization
            delta_scenarios = ['no_reward', 'forage_reward', 'focal_forage_reward', 'stag_reward']
            num_agents = getattr(self.algorithm, 'num_agents', 0)
            
            for agent_id in range(num_agents):
                for scenario in delta_scenarios:
                    mean_key = f'delta_agent_{agent_id}_{scenario}_mean'
                    std_key = f'delta_agent_{agent_id}_{scenario}_std'
                    count_key = f'delta_agent_{agent_id}_{scenario}_count'
                    
                    if mean_key in update_info:
                        # Log with agent-specific naming
                        wandb_data[f'causal_effect/agent_{agent_id}/delta_{scenario}'] = update_info[mean_key]
                        # Also log std and count separately for reference
                        if std_key in update_info:
                            wandb_data[f'causal_effect/agent_{agent_id}/delta_{scenario}_std'] = update_info[std_key]
                        if count_key in update_info:
                            wandb_data[f'causal_effect/agent_{agent_id}/delta_{scenario}_count'] = update_info[count_key]
            
            wandb.log(wandb_data, step=self.iteration)
    
    def log_evaluation_metrics(
        self,
        eval_info: Dict[str, float],
        update_info: Optional[Dict[str, float]] = None,
        title: str = "Evaluation Results:",
    ):
        """Log evaluation metrics."""
        self.logger.info(title)
        self.logger.info(f"  Success Rate: {eval_info['eval_success_rate']:.3f}")
        self.logger.info(f"  Mean Reward: {eval_info['eval_mean_reward']:.3f} ± {eval_info['eval_std_reward']:.3f}")
        self.logger.info(f"  Mean Length: {eval_info['eval_mean_length']:.1f}")
        self.logger.info(f"  Min/Max Reward: {eval_info['eval_min_reward']:.3f} / {eval_info['eval_max_reward']:.3f}")
        
        if eval_info['is_best']:
            self.logger.info("  *** New best performance! ***")

        eval_only_adv_metrics = self._get_eval_only_advantage_metrics(update_info)
        if eval_only_adv_metrics:
            self.logger.info("  PCI Advantage Snapshot:")
            for key, value in eval_only_adv_metrics.items():
                self._log_update_metric_line(key, value, eval_only_adv_metrics, indent="    ")

        gram_spectrum_metrics = {}
        if hasattr(self.algorithm, 'compute_gram_spectrum_metrics'):
            try:
                gram_spectrum_metrics = self.algorithm.compute_gram_spectrum_metrics()
            except Exception as exc:
                self.logger.warning(f"Failed to compute eval Gram spectrum metrics: {exc}")

        if gram_spectrum_metrics:
            self.logger.info("  Gram Spectrum Snapshot:")
            for key, value in gram_spectrum_metrics.items():
                self.logger.info(f"    {key}: {value:.6f}")
        
        if self.use_wandb:
            wandb_data = {
                'iteration': self.iteration,
                'total_steps': self.total_steps,
                'eval/success_rate': eval_info['eval_success_rate'],
                'eval/mean_reward': eval_info['eval_mean_reward'],
                'eval/std_reward': eval_info['eval_std_reward'],
                'eval/mean_length': eval_info['eval_mean_length'],
                'eval/min_reward': eval_info['eval_min_reward'],
                'eval/max_reward': eval_info['eval_max_reward'],
                'eval/best_reward': self.best_eval_reward
            }

            for key, value in eval_only_adv_metrics.items():
                wandb_data[f'eval/{key}'] = value
            for key, value in gram_spectrum_metrics.items():
                wandb_data[f'eval/{key}'] = value
            
            # Add causal effect metrics to wandb (only for PCI-MAPPO)
            if self.causal_evaluator is not None and 'causal_effects' in eval_info and eval_info['causal_effects'] is not None:
                causal_stats = eval_info['causal_effects']
                for scenario, stats in causal_stats.items():
                    wandb_data[f'eval/causal_effect_{scenario}_mean'] = stats['mean']
                    wandb_data[f'eval/causal_effect_{scenario}_std'] = stats['std']
                    wandb_data[f'eval/causal_effect_{scenario}_count'] = stats['count']
                
                # Create and log causal effect plot to wandb
                try:
                    import matplotlib.pyplot as plt
                    fig = self.causal_evaluator.plot_causal_effects(
                        causal_stats, 
                        return_figure=True
                    )
                    if fig is not None:
                        wandb_data[f'eval/causal_effect_plot'] = wandb.Image(fig)
                        plt.close(fig)  # Clean up memory
                except Exception as e:
                    self.logger.warning(f"Failed to create causal effect plot for wandb: {e}")
            
            wandb.log(wandb_data, step=self.iteration)
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        # Skip saving if save_dir is None (testing mode)
        if self.save_dir is None:
            return
            
        checkpoint = {
            'iteration': self.iteration,
            'total_steps': self.total_steps,
            'best_eval_reward': self.best_eval_reward,
            'algorithm_state': self.get_algorithm_state(),
        }
        
        # Save checkpoint
        if is_final:
            checkpoint_path = os.path.join(self.save_dir, 'final_checkpoint.pt')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{self.iteration}.pt')
        
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if is_final or (hasattr(self, 'best_eval_reward') and self.best_eval_reward > float('-inf')):
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
            
        # Save the observational-regressor checkpoint separately.
        if getattr(self.algorithm, "train_naivepredictor", False) and hasattr(self.algorithm, "naive_predictor"):
            naive_dir = self.save_dir + "_naivepredictor"
            os.makedirs(naive_dir, exist_ok=True)
            naive_checkpoint = {
                'iteration': self.iteration,
                'naive_predictor_state': self.algorithm.naive_predictor.state_dict(),
                'naive_optimizer_state': self.algorithm.naive_optimizer.state_dict()
            }
            if is_final:
                naive_path = os.path.join(naive_dir, 'final_checkpoint.pt')
            else:
                naive_path = os.path.join(naive_dir, f'checkpoint_{self.iteration}.pt')
            torch.save(naive_checkpoint, naive_path, _use_new_zipfile_serialization=False)
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        if self.save_dir is None or not os.path.exists(self.save_dir):
            return None
        
        # Check for final checkpoint first
        final_checkpoint = os.path.join(self.save_dir, 'final_checkpoint.pt')
        if os.path.exists(final_checkpoint):
            return final_checkpoint
        
        # Check for best model
        best_checkpoint = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(best_checkpoint):
            return best_checkpoint
        
        # Find the latest iteration checkpoint
        checkpoint_files = [f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if checkpoint_files:
            # Sort by iteration number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = os.path.join(self.save_dir, checkpoint_files[-1])
            return latest_checkpoint
        
        return None
    
    def get_algorithm_state(self) -> Dict[str, Any]:
        """Get algorithm state for saving."""
        state = {
            'agents': {},
            'algorithm_type': type(self.algorithm).__name__
        }

        if hasattr(self.algorithm, 'agents'):
            for agent_id, agent in self.algorithm.agents.items():
                agent_state = {}
                if hasattr(agent, 'policy_net'):
                    agent_state['policy_net'] = agent.policy_net.state_dict()
                if hasattr(agent, 'optimizer'):
                    agent_state['optimizer'] = agent.optimizer.state_dict()
                if hasattr(agent, 'value_net'):
                    agent_state['value_net'] = agent.value_net.state_dict()
                if hasattr(agent, 'value_optimizer'):
                    agent_state['value_optimizer'] = agent.value_optimizer.state_dict()
                state['agents'][agent_id] = agent_state
        
        # Save algorithm-specific states
        if isinstance(self.algorithm, rPCIAlgorithm):
            # PCI-MAPPO specific states
            state['outcome_bridge_net'] = self.algorithm.outcome_bridge_net.state_dict()
            state['outcome_bridge_optimizer'] = self.algorithm.outcome_bridge_optimizer.state_dict()
            if hasattr(self.algorithm, 'critic_kernel_w'):
                state['critic_kernel_w'] = self.algorithm.critic_kernel_w.state_dict()
            if hasattr(self.algorithm, 'prev_action_embedding'):
                state['prev_action_embedding'] = self.algorithm.prev_action_embedding.state_dict()
            if hasattr(self.algorithm, 'action_embedding_for_kernel'):
                state['action_embedding_for_kernel'] = self.algorithm.action_embedding_for_kernel.state_dict()
            if hasattr(self.algorithm, 'kernel_action_embed_optimizer'):
                state['kernel_action_embed_optimizer'] = self.algorithm.kernel_action_embed_optimizer.state_dict()
            if getattr(self.algorithm, 'delta_value_net', None) is not None:
                state['delta_value_net'] = self.algorithm.delta_value_net.state_dict()
            if getattr(self.algorithm, 'delta_value_optimizer', None) is not None:
                state['delta_value_optimizer'] = self.algorithm.delta_value_optimizer.state_dict()
            # W Predictor (rQPCI only)
            if hasattr(self.algorithm, 'w_predictor'):
                state['w_predictor'] = self.algorithm.w_predictor.state_dict()
                state['w_predictor_optimizer'] = self.algorithm.w_predictor_optimizer.state_dict()
            
            # Save value networks and optimizers (only for algorithms with value networks)
            if hasattr(self.algorithm, 'value_nets') or hasattr(self.algorithm, 'value_net'):
                from utils.util import is_value_network_shared
                
                if not is_value_network_shared(self.algorithm):
                    # Individual value networks
                    state['value_nets'] = {}
                    state['value_optimizers'] = {}
                    for agent_id, value_net in self.algorithm.value_nets.items():
                        state['value_nets'][agent_id] = value_net.state_dict()
                    
                    for agent_id, value_optimizer in self.algorithm.value_optimizers.items():
                        state['value_optimizers'][agent_id] = value_optimizer.state_dict()
                else:
                    # Parameter sharing: save single value_net
                    if hasattr(self.algorithm, 'value_net'):
                        state['value_net'] = self.algorithm.value_net.state_dict()
                    if hasattr(self.algorithm, 'value_optimizer'):
                        state['value_optimizer'] = self.algorithm.value_optimizer.state_dict()
        else:
            if hasattr(self.algorithm, 'value_net'):
                state['value_net'] = self.algorithm.value_net.state_dict()
            if hasattr(self.algorithm, 'value_optimizer'):
                state['value_optimizer'] = self.algorithm.value_optimizer.state_dict()
        
        return state
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.iteration = checkpoint['iteration']
        self.total_steps = checkpoint['total_steps']
        self.best_eval_reward = checkpoint['best_eval_reward']
        
        # Load algorithm state
        algorithm_state = checkpoint['algorithm_state']

        agent_states = algorithm_state.get('agents', {})
        if hasattr(self.algorithm, 'agents'):
            for agent_id, agent in self.algorithm.agents.items():
                saved_agent_state = agent_states.get(agent_id, agent_states.get(str(agent_id)))
                if not saved_agent_state:
                    continue
                if 'policy_net' in saved_agent_state and hasattr(agent, 'policy_net'):
                    agent.policy_net.load_state_dict(saved_agent_state['policy_net'])
                if 'optimizer' in saved_agent_state and hasattr(agent, 'optimizer'):
                    agent.optimizer.load_state_dict(saved_agent_state['optimizer'])
                if 'value_net' in saved_agent_state and hasattr(agent, 'value_net'):
                    agent.value_net.load_state_dict(saved_agent_state['value_net'])
                if 'value_optimizer' in saved_agent_state and hasattr(agent, 'value_optimizer'):
                    agent.value_optimizer.load_state_dict(saved_agent_state['value_optimizer'])

        # Load algorithm-specific states
        if isinstance(self.algorithm, rPCIAlgorithm):
            # PCI-MAPPO specific states
            if 'outcome_bridge_net' in algorithm_state:
                self.algorithm.outcome_bridge_net.load_state_dict(algorithm_state['outcome_bridge_net'])
            if 'action_bridge_net' in algorithm_state:
                self.algorithm.action_bridge_net.load_state_dict(algorithm_state['action_bridge_net'])
            if 'critic_kernel_z' in algorithm_state:
                self.algorithm.critic_kernel_z.load_state_dict(algorithm_state['critic_kernel_z'])
            if 'critic_kernel_w' in algorithm_state:
                self.algorithm.critic_kernel_w.load_state_dict(algorithm_state['critic_kernel_w'])
            if 'prev_action_embedding' in algorithm_state and hasattr(self.algorithm, 'prev_action_embedding'):
                self.algorithm.prev_action_embedding.load_state_dict(algorithm_state['prev_action_embedding'])
            if 'action_embedding_for_kernel' in algorithm_state and hasattr(self.algorithm, 'action_embedding_for_kernel'):
                self.algorithm.action_embedding_for_kernel.load_state_dict(algorithm_state['action_embedding_for_kernel'])
            if 'outcome_bridge_optimizer' in algorithm_state:
                self.algorithm.outcome_bridge_optimizer.load_state_dict(algorithm_state['outcome_bridge_optimizer'])
            if 'kernel_action_embed_optimizer' in algorithm_state:
                self.algorithm.kernel_action_embed_optimizer.load_state_dict(algorithm_state['kernel_action_embed_optimizer'])
            if 'delta_value_net' in algorithm_state and getattr(self.algorithm, 'delta_value_net', None) is not None:
                self.algorithm.delta_value_net.load_state_dict(algorithm_state['delta_value_net'])
            if 'delta_value_optimizer' in algorithm_state and getattr(self.algorithm, 'delta_value_optimizer', None) is not None:
                self.algorithm.delta_value_optimizer.load_state_dict(algorithm_state['delta_value_optimizer'])
     
        # Load shared value network (for all algorithms)
        if 'value_net' in algorithm_state:
            if hasattr(self.algorithm, 'value_net'):
                self.algorithm.value_net.load_state_dict(algorithm_state['value_net'])
                # If using individual networks, copy to all agents
                if hasattr(self.algorithm, 'value_nets'):
                    for agent_id in self.algorithm.value_nets.keys():
                        self.algorithm.value_nets[agent_id].load_state_dict(algorithm_state['value_net'])
            else:
                self.logger.warning("Checkpoint contains shared value network but algorithm uses individual networks")
        
        if 'value_optimizer' in algorithm_state:
            if hasattr(self.algorithm, 'value_optimizer'):
                self.algorithm.value_optimizer.load_state_dict(algorithm_state['value_optimizer'])
                # If using individual optimizers, copy to all agents
                if hasattr(self.algorithm, 'value_optimizers'):
                    for agent_id in self.algorithm.value_optimizers.keys():
                        self.algorithm.value_optimizers[agent_id].load_state_dict(algorithm_state['value_optimizer'])
            else:
                self.logger.warning("Checkpoint contains shared value optimizer but algorithm uses individual optimizers")
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from iteration {self.iteration}")
