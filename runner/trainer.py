import os
import time
import numpy as np
import torch
import wandb
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import logging
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from algorithms import PCIMAPPOAlgorithm, MAPPOAlgorithm, rMAPPOAlgorithm, IPPOAlgorithm, rIPPOAlgorithm, X_PCIAlgorithm, COMAAlgorithm, rCOMAAlgorithm, rPCIAlgorithm, rQPCIAlgorithm
from envs import MultiAgentEnv, dict_to_tensor
from utils import Logger, MetricsTracker, get_causal_evaluator, get_advanced_causal_evaluator
from typing import Union


def get_network_architectures() -> Dict[str, str]:
    """
    네트워크 아키텍처 정보를 추출합니다 (rnn.py 제외).
    
    Returns:
        Dict[str, str]: 네트워크 파일명을 키로 하고 소스 코드를 값으로 하는 딕셔너리
    """
    import os
    networks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms", "networks")
    architectures = {}
    
    # 제외할 파일
    exclude_files = {"rnn.py", "__init__.py", "__pycache__"}
    
    # 네트워크 파일들 읽기
    for filename in os.listdir(networks_dir):
        if filename.endswith(".py") and filename not in exclude_files:
            filepath = os.path.join(networks_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    architectures[filename] = f.read()
            except Exception as e:
                architectures[filename] = f"Error reading file: {str(e)}"
    
    return architectures


class Trainer:
    """
    Main trainer for PCI-MAPPO, MAPPO, and IPPO algorithms.
    Implements the training loop following the pseudocode.
    """
    
    def __init__(
        self,
        config: Dict,
        env: MultiAgentEnv,
        algorithm: Union[PCIMAPPOAlgorithm, MAPPOAlgorithm, IPPOAlgorithm, COMAAlgorithm, rCOMAAlgorithm, rPCIAlgorithm, rMAPPOAlgorithm, rIPPOAlgorithm, rQPCIAlgorithm],
        max_iterations: int = 1000,
        max_steps_per_iteration: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        save_dir: str = "./save_checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "pci-mappo",
        wandb_run_name: Optional[str] = None,
        device: str = "cpu",
        n_rollout_threads: int = 1,  # [ADDED] Number of parallel environments
        env_fn: Optional[callable] = None,  # [ADDED] Function to create environment (for parallel)
        episodes_per_iteration: Optional[int] = None  # [ADDED] Number of episodes to collect per iteration
    ):
        self.config = config
        self.n_rollout_threads = n_rollout_threads
        self.episodes_per_iteration = episodes_per_iteration
        
        # Setup environment (single or parallel)
        if n_rollout_threads > 1:
            # Use parallel environment wrapper for training
            if env_fn is None:
                raise ValueError("env_fn must be provided when n_rollout_threads > 1")
            from envs.parallel_env import ParallelEnvWrapper
            self.env = ParallelEnvWrapper(env_fn, n_rollout_threads)
            self.is_parallel = True
            # Keep a single environment for evaluation
            self.eval_env = env
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
        self.seed = getattr(self, 'seed', None)
        self.deterministic = getattr(self, 'deterministic', True)
        
        # Initialize logging first with algorithm-specific name
        algorithm_type = type(algorithm).__name__.replace('Algorithm', '').upper()
        self.logger = Logger(name=algorithm_type)
        self.metrics_tracker = MetricsTracker()
        
        # Initialize causal effect evaluator (only for PCI-MAPPO)
        # For other algorithms, these will be None
        self.causal_evaluator = None
        self.advanced_causal_evaluator = None

        # Initialize causal effect evaluator (only for PCI-MAPPO and StagHuntEnv)
        if isinstance(algorithm, (PCIMAPPOAlgorithm, X_PCIAlgorithm)) and env.__class__.__name__ == "StagHuntEnv":
            CausalEffectEvaluator = get_causal_evaluator()
            self.causal_evaluator = CausalEffectEvaluator(
                algorithm=algorithm,
                env=env,
                num_eval_steps=5000,  # 기본값으로 5000 스텝 설정
                device=device
            )
            
            # Initialize advanced causal effect evaluator
            AdvancedCausalEvaluator = get_advanced_causal_evaluator()
            self.advanced_causal_evaluator = AdvancedCausalEvaluator(
                algorithm=algorithm,
                env=env,
                num_eval_steps=5000,  # 기본값으로 5000 스텝 설정
                device=device
            )
        
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
            # Get algorithm type for config
            algorithm_type = type(self.algorithm).__name__.replace('Algorithm', '').lower()
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config
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
                    "no_reward": "Other agent receives no reward (0.0)",
                    "forage_reward": "Other agent receives forage reward (3.0)", 
                    "stag_reward": "Other agent receives stag reward (5.0)"
                },
                "reproducibility": {
                    "seed": getattr(self, 'seed', 'unknown'),
                    "deterministic": getattr(self, 'deterministic', 'unknown')
                }
            })
            
            # 네트워크 아키텍처 저장 (rnn.py 제외)
            network_architectures = get_network_architectures()
            for filename, source_code in network_architectures.items():
                wandb.summary[f"network_architecture/{filename}"] = wandb.Html(
                    f"<pre style='white-space: pre-wrap;'>{source_code}</pre>"
                )
        
        # Training state
        self.iteration = 0
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
        # Episode statistics
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.recent_rewards = deque(maxlen=100)
    
    def _create_simulation_directory(self) -> str:
        """Create structured directory for this simulation."""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get environment name
        env_name = self.eval_env.__class__.__name__.lower()
        
        # Create directory structure: base_dir/timestamp/env_name/checkpoints
        sim_dir = os.path.join(
            self.base_save_dir,
            timestamp,
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
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        if hasattr(self.config.environment, 'get_relevant_config'):
             config_dict['environment'] = self.config.environment.get_relevant_config()
             
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
                update_info = self.algorithm.update()
                
                # Log metrics (only for logging, not displayed in tqdm)
                if iteration % self.log_interval == 0:
                    self.log_training_metrics(rollout_data, update_info)
                
                # Evaluate policy
                if iteration % self.eval_interval == 0:
                    eval_info = self.evaluate_policy()
                    self.log_evaluation_metrics(eval_info)
                
                # Save checkpoint
                if iteration % self.save_interval == 0:
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
        final_eval = self.evaluate_policy()
        mean_reward = final_eval.get('eval_mean_reward', 'N/A')
        if isinstance(mean_reward, (int, float)):
            self.logger.info(f"Final evaluation reward: {mean_reward:.4f}")
        else:
            self.logger.info(f"Final evaluation reward: {mean_reward}")
        self.save_checkpoint(is_final=True)
        
        # Save final training summary
        self._save_training_summary(total_time, final_eval)
        
        if self.use_wandb:
            wandb.finish()
    
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
            
            # Initialize previous observations (for first step, use zero vector since no previous observation exists)
            prev_observations = {agent_id: torch.zeros_like(obs) for agent_id, obs in observations.items()}
            # Initialize previous rewards (for first step, use 0.0)
            prev_rewards = {agent_id: 0.0 for agent_id in observations.keys()}
            # Initialize previous actions (for first step, use 0 for simplicity, or -1 if 0 is meaningful)
            # Assuming 0 is a valid action, but since it's the first step, it shouldn't matter much as long as it's consistent dimension-wise.
            # Ideally, we could use a special token or just 0.
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
                    # Convert to Dict[int, List[int]]
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
                self.total_steps += 1
                
                # Check if episode is done
                done = any(dones.values()) or episode_length >= 1000
            
            # For COMA, finish the episode in the buffer only if episode actually completed
            # If episode was cut short due to max_steps_per_iteration, keep it in current_episode
            # for the next iteration to continue collecting
            if isinstance(self.algorithm, (COMAAlgorithm, rCOMAAlgorithm)):
                if done:  # Episode actually completed
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
            elif "SMAC" in str(self.env):
                if info.get('battle_won', False):
                    is_success = 1.0
            elif "LBForaging" in str(self.env):
                if sum(episode_reward.values()) >= 0.99:
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
        from envs import dict_to_tensor
        
        step_count = 0
        episode_count = 0
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        
        # Reset all environments
        observations_batch = self.env.reset()
        self.algorithm.reset_agents_batch(self.n_rollout_threads)
        
        # Convert to tensors
        observations_batch = [
            dict_to_tensor(obs, device=self.algorithm.device)
            for obs in observations_batch
        ]
        
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
            # Get available actions
            available_actions_batch = self.env.get_avail_actions_batch()
            
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
            next_observations_batch, rewards_batch, dones_batch, infos_batch = self.env.step(env_actions_batch)
            
            # Convert to tensors
            next_observations_batch = [
                dict_to_tensor(obs, device=self.algorithm.device)
                for obs in next_observations_batch
            ]
            
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
                    step_count += 1 # actually, increment=n_rollout_threads
                    
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
 
    def evaluate_policy(self) -> Dict[str, float]:
        """
        Evaluate current policy for multiple episodes.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        eval_rewards = defaultdict(list)
        eval_lengths = []
        eval_successes = []
        
        for episode in range(self.eval_episodes):
            # Reset environment and agents (use eval_env for evaluation)
            observations = self.eval_env.reset()
            self.algorithm.reset_agents()
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            
            episode_reward = defaultdict(float)
            episode_success = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                # Get availability of actions
                available_actions = None
                if hasattr(self.eval_env, "get_avail_actions"):
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
                next_observations, rewards, dones, info = self.eval_env.step(actions)
                next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)
                
                # Check success (Evaluate)
                if self.eval_env.__class__.__name__ == "StagHuntEnv":
                    stag_alive = info.get('stag_alive', {})
                    hare_alive = info.get('hare_alive', {})
                    if not any(stag_alive.values()) and not any(hare_alive.values()):
                        episode_success = 1.0
                        
                if info.get('battle_won', False): # for SMAC and SMACv2
                    episode_success = 1.0

                # Update statistics
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                
                observations = next_observations
                episode_length += 1
                done = any(dones.values())
            
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
            for key, value in update_info.items():
                if key.startswith('delta_'):
                    # Format delta statistics nicely
                    if key.endswith('_mean'):
                        scenario = key.replace('delta_', '').replace('_mean', '')
                        count_key = f'delta_{scenario}_count'
                        if count_key in update_info:
                            count = update_info[count_key]
                            self.logger.info(f"    {key}: {value:.6f} (n={count})")
                        else:
                            self.logger.info(f"    {key}: {value:.6f}")
                    elif key.endswith('_std'):
                        scenario = key.replace('delta_', '').replace('_std', '')
                        self.logger.info(f"    {key}: {value:.6f}")
                    elif key.endswith('_count'):
                        continue  # Skip count, already shown with mean
                    else:
                        self.logger.info(f"    {key}: {value:.6f}")
                else:
                    self.logger.info(f"    {key}: {value:.6f}")
        
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
                wandb_data[f'update/{key}'] = value
            
            # Add delta statistics for causal reward shaping
            # Group by agent and scenario for better visualization
            delta_scenarios = ['no_reward', 'forage_reward', 'stag_reward']
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
    
    def log_evaluation_metrics(self, eval_info: Dict[str, float]):
        """Log evaluation metrics."""
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Mean Reward: {eval_info['eval_mean_reward']:.3f} ± {eval_info['eval_std_reward']:.3f}")
        self.logger.info(f"  Mean Length: {eval_info['eval_mean_length']:.1f}")
        self.logger.info(f"  Min/Max Reward: {eval_info['eval_min_reward']:.3f} / {eval_info['eval_max_reward']:.3f}")
        
        if eval_info['is_best']:
            self.logger.info("  *** New best performance! ***")
        
        if self.use_wandb:
            wandb_data = {
                'eval/success_rate': eval_info['eval_success_rate'],
                'eval/mean_reward': eval_info['eval_mean_reward'],
                'eval/std_reward': eval_info['eval_std_reward'],
                'eval/mean_length': eval_info['eval_mean_length'],
                'eval/min_reward': eval_info['eval_min_reward'],
                'eval/max_reward': eval_info['eval_max_reward'],
                'eval/best_reward': self.best_eval_reward
            }
            
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
        
        # Save agent states
        if isinstance(self.algorithm, rCOMAAlgorithm):
            # COMA RNN uses shared policy — save once, no per-agent optimizer
            state['shared_policy'] = self.algorithm.shared_policy.state_dict()
            state['policy_optimizer'] = self.algorithm.policy_optimizer.state_dict()
            for agent_id, agent in self.algorithm.agents.items():
                state['agents'][agent_id] = {
                    'policy_net': agent.policy_net.state_dict(),
                }
        else:
            for agent_id, agent in self.algorithm.agents.items():
                state['agents'][agent_id] = {
                    'policy_net': agent.policy_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                }
        
        # Save algorithm-specific states
        if isinstance(self.algorithm, (PCIMAPPOAlgorithm, rPCIAlgorithm, rQPCIAlgorithm)):
            # PCI-MAPPO specific states
            state['outcome_bridge_net'] = self.algorithm.outcome_bridge_net.state_dict()
            state['action_bridge_net'] = self.algorithm.action_bridge_net.state_dict()
            state['critic_kernel_z'] = self.algorithm.critic_kernel_z.state_dict()
            state['critic_kernel_w'] = self.algorithm.critic_kernel_w.state_dict()
            state['action_embedding_for_kernel'] = self.algorithm.action_embedding_for_kernel.state_dict()
            state['outcome_bridge_optimizer'] = self.algorithm.outcome_bridge_optimizer.state_dict()
            state['action_bridge_optimizer'] = self.algorithm.action_bridge_optimizer.state_dict()
            state['kernel_action_embed_optimizer'] = self.algorithm.kernel_action_embed_optimizer.state_dict()
            # W Predictor (rQPCI only)
            if hasattr(self.algorithm, 'w_predictor'):
                state['w_predictor'] = self.algorithm.w_predictor.state_dict()
                state['w_predictor_optimizer'] = self.algorithm.w_predictor_optimizer.state_dict()
            
            # Save value networks and optimizers (only for algorithms with value networks)
            if hasattr(self.algorithm, 'value_nets') or hasattr(self.algorithm, 'value_net'):
                from utils.value_network_utils import is_value_network_shared
                
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
                    
        elif isinstance(self.algorithm, X_PCIAlgorithm):
            # X-PCI specific states
            state['reward_predictor'] = self.algorithm.reward_predictor.state_dict()
            state['reward_predictor_optimizer'] = self.algorithm.reward_predictor_optimizer.state_dict()

            from utils.value_network_utils import is_value_network_shared
            
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
        
        elif isinstance(self.algorithm, (COMAAlgorithm, rCOMAAlgorithm)):
            # COMA specific states
            state['critic'] = self.algorithm.critic.state_dict()
            state['target_critic'] = self.algorithm.target_critic.state_dict()
            state['critic_optimizer'] = self.algorithm.critic_optimizer.state_dict()
        else:
            # MAPPO and IPPO: save shared value network
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
    
        # Load agent states
        if isinstance(self.algorithm, rCOMAAlgorithm):
            # COMA RNN: load shared policy and optimizer
            if 'shared_policy' in algorithm_state:
                self.algorithm.shared_policy.load_state_dict(algorithm_state['shared_policy'])
            if 'policy_optimizer' in algorithm_state:
                self.algorithm.policy_optimizer.load_state_dict(algorithm_state['policy_optimizer'])
        else:
            for agent_id, agent_state in algorithm_state['agents'].items():
                agent = self.algorithm.agents[int(agent_id)]
                agent.policy_net.load_state_dict(agent_state['policy_net'])
                agent.optimizer.load_state_dict(agent_state['optimizer'])
        
        # Load algorithm-specific states
        if isinstance(self.algorithm, (PCIMAPPOAlgorithm, rPCIAlgorithm, rQPCIAlgorithm)):
            # PCI-MAPPO specific states
            if 'outcome_bridge_net' in algorithm_state:
                self.algorithm.outcome_bridge_net.load_state_dict(algorithm_state['outcome_bridge_net'])
            if 'action_bridge_net' in algorithm_state:
                self.algorithm.action_bridge_net.load_state_dict(algorithm_state['action_bridge_net'])
            if 'critic_kernel_z' in algorithm_state:
                self.algorithm.critic_kernel_z.load_state_dict(algorithm_state['critic_kernel_z'])
            if 'critic_kernel_w' in algorithm_state:
                self.algorithm.critic_kernel_w.load_state_dict(algorithm_state['critic_kernel_w'])
            if 'action_embedding_for_kernel' in algorithm_state:
                self.algorithm.action_embedding_for_kernel.load_state_dict(algorithm_state['action_embedding_for_kernel'])
            if 'outcome_bridge_optimizer' in algorithm_state:
                self.algorithm.outcome_bridge_optimizer.load_state_dict(algorithm_state['outcome_bridge_optimizer'])
            if 'action_bridge_optimizer' in algorithm_state:
                self.algorithm.action_bridge_optimizer.load_state_dict(algorithm_state['action_bridge_optimizer'])
            if 'kernel_action_embed_optimizer' in algorithm_state:
                self.algorithm.kernel_action_embed_optimizer.load_state_dict(algorithm_state['kernel_action_embed_optimizer'])
            # W Predictor (rQPCI only)
            if 'w_predictor' in algorithm_state and hasattr(self.algorithm, 'w_predictor'):
                self.algorithm.w_predictor.load_state_dict(algorithm_state['w_predictor'])
            if 'w_predictor_optimizer' in algorithm_state and hasattr(self.algorithm, 'w_predictor_optimizer'):
                self.algorithm.w_predictor_optimizer.load_state_dict(algorithm_state['w_predictor_optimizer'])

        elif isinstance(self.algorithm, X_PCIAlgorithm):
            # X-PCI specific states
            if 'reward_predictor' in algorithm_state:
                self.algorithm.reward_predictor.load_state_dict(algorithm_state['reward_predictor'])
            if 'reward_predictor_optimizer' in algorithm_state:
                self.algorithm.reward_predictor_optimizer.load_state_dict(algorithm_state['reward_predictor_optimizer'])
        
        # Load value networks and optimizers for algorithms that support individual value networks
        if isinstance(self.algorithm, (PCIMAPPOAlgorithm, X_PCIAlgorithm)):
            if 'value_nets' in algorithm_state:
                # Individual value networks
                if hasattr(self.algorithm, 'value_nets'):
                    for agent_id, value_net_state in algorithm_state['value_nets'].items():
                        self.algorithm.value_nets[int(agent_id)].load_state_dict(value_net_state)
                else:
                    self.logger.warning("Checkpoint contains individual value networks but algorithm uses parameter sharing")
            
            if 'value_optimizers' in algorithm_state:
                # Individual value optimizers
                if hasattr(self.algorithm, 'value_optimizers'):
                    for agent_id, value_optimizer_state in algorithm_state['value_optimizers'].items():
                        self.algorithm.value_optimizers[int(agent_id)].load_state_dict(value_optimizer_state)
                else:
                    self.logger.warning("Checkpoint contains individual value optimizers but algorithm uses parameter sharing")
        
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
    
    def test_policy(
        self, 
        num_episodes: int = 10, 
        render: bool = False, 
        save_animation: bool = False, 
        animation_dir: str = "./save_animations",
        save_replay: bool = True,
        evaluate_advanced_causal: bool = False,
        advanced_causal_steps: int = 5000,
        evaluate_policy_quality: bool = True
    ):
        """
        Test the trained policy.
        
        Args:
            num_episodes: Number of episodes for basic policy testing
            render: Whether to render the environment during testing
            save_animation: Whether to save animations
            animation_dir: Directory to save animations
            evaluate_advanced_causal: Whether to run advanced causal effect evaluation
            advanced_causal_steps: Number of env steps (interactions) for advanced causal evaluation (approximate)
            evaluate_policy_quality: Whether to evaluate policy quality metrics for stag-hunt environment
        """
        self.logger.info(f"Testing policy for {num_episodes} episodes...")
        
        # Create animation directory if needed
        if save_animation:
            os.makedirs(animation_dir, exist_ok=True)
            self.logger.info(f"Animation will be saved to: {animation_dir}")
        
        test_rewards = defaultdict(list)
        test_lengths = []
        
        # Policy quality metrics for stag-hunt environment (agent-specific)
        policy_metrics = {
            'agents': {},  # agent별 지표 저장
            'total_episodes': 0,
            'total_steps': 0  # 총 스텝 수
        }
        
        # Initialize agent-specific metrics (will be initialized after first reset)
        agents_initialized = False
        # Determine data collection method
        if evaluate_policy_quality:
            # Use step-based collection for policy quality evaluation
            self.logger.info(f"Collecting policy quality data for {advanced_causal_steps} steps...")
            step_count = 0
            episode_count = 0
            
            # Reset policy bias buffers in evaluator (case-by-case and aggregated) - only for PCI-MAPPO
            if self.advanced_causal_evaluator is not None:
                try:
                    from collections import defaultdict as _dd
                    # Initialize all data structures used for case-by-case study
                    self.advanced_causal_evaluator.object_choice_data = _dd(lambda: _dd(list))
                    if hasattr(self.advanced_causal_evaluator, 'object_choice_cases'):
                        self.advanced_causal_evaluator.object_choice_cases = []
                    if hasattr(self.advanced_causal_evaluator, 'case_attempt_data'):
                        self.advanced_causal_evaluator.case_attempt_data = _dd(lambda: _dd(list))
                    # Also reset detailed_causal_effects for clean statistics
                    if hasattr(self.advanced_causal_evaluator, 'detailed_causal_effects'):
                        self.advanced_causal_evaluator.detailed_causal_effects = _dd(lambda: _dd(list))
                except Exception as e:
                    self.logger.debug(f"Failed to reset evaluator data: {e}")

            while step_count < advanced_causal_steps:
                episode_count += 1
                observations = self.env.reset()
                self.algorithm.reset_agents()
                observations = dict_to_tensor(observations, device=self.algorithm.device)
                
                # Initialize agent-specific metrics on first reset
                if not agents_initialized:
                    for agent_id in range(self.env.num_agents):
                        policy_metrics['agents'][agent_id] = {
                            'my_stag_hunting_utilization': 0.0,
                            'coop_stag_hunting_utilization': 0.0,
                            'other_stag_hunting_utilization': 0.0,
                            'hare_hunting_utilization': 0.0,
                            'my_stag_hunt_attempts': 0,
                            'coop_stag_hunt_attempts': 0,
                            'other_stag_hunt_attempts': 0,
                            'hare_hunt_attempts': 0,
                            'my_stag_opportunities': 0,
                            'coop_stag_opportunities': 0,
                            'other_stag_opportunities': 0,
                            'hare_opportunities': 0
                        }
                    agents_initialized = True
                
                episode_reward = defaultdict(float)
                episode_length = 0
                done = False
                
                # metrics for policy quality evaluation (agent-specific)
                agent_metrics = {}
                for agent_id in range(self.env.num_agents):
                    agent_metrics[agent_id] = {
                        'my_stag_hunt_attempts': 0,
                        'coop_stag_hunt_attempts': 0,
                        'other_stag_hunt_attempts': 0,
                        'hare_hunt_attempts': 0,
                        'my_stag_opportunities': 0,
                        'coop_stag_opportunities': 0,
                        'other_stag_opportunities': 0,
                        'hare_opportunities': 0
                    }
                
                # Animation frames storage
                frames = []
                
                while not done and step_count < advanced_causal_steps:
                    # Render and capture frame for animation
                    if (render or save_animation) and hasattr(self.env, 'render'):
                        if save_animation:
                            # Capture frame for animation
                            frame = self.env.render(mode='rgb_array')
                            if frame is not None:
                                frames.append(frame.copy())
                                self.logger.debug(f"Frame {len(frames)} captured for episode {episode_count}")
                            else:
                                self.logger.warning(f"Failed to capture frame for episode {episode_count}")
                        else:
                            # Regular rendering
                            self.env.render()
                    
                    # Get availability of actions
                    available_actions = None
                    if hasattr(self.env, "get_avail_actions"):
                        avail = self.env.get_avail_actions()
                        # Convert to Dict[int, List[int]]
                        # SMAC returns List[List[int]] where index is agent_id
                        available_actions = {i: avail[i] for i in range(len(avail))}

                    # Get actions (deterministic for testing)
                    with torch.no_grad():
                        actions = {}
                        for agent_id, obs in observations.items():
                            # Handle available actions for agent
                            avail_actions_tensor = None
                            if available_actions is not None and agent_id in available_actions:
                                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.algorithm.device)

                            if avail_actions_tensor is not None:
                                 action, _, = self.algorithm.agents[agent_id].get_action(obs, available_actions=avail_actions_tensor)
                            else:
                                 action, _, = self.algorithm.agents[agent_id].get_action(obs)

                            actions[agent_id] = action
                    
                    # Take step
                    next_observations, rewards, dones, info = self.env.step(actions)
                    next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)

                    # Update statistics
                    for agent_id, reward in rewards.items():
                        episode_reward[agent_id] += reward
                    
                    # Collect policy quality metrics if enabled 
                    if evaluate_policy_quality and 'huntable_objects' in info:
                        huntable_objects = info['huntable_objects']
                        agents_with_hare_opportunity = set()
                        
                        # Process stag opportunities (my_stag, coop_stag)
                        if 'huntable_stags' in huntable_objects:
                            for stag_id, stag_info in huntable_objects['huntable_stags'].items():
                                required_agents = stag_info.get('required_agents', [])
                                
                                # 각 agent별로 my_stag vs coop_stag 구분
                                for agent_id in range(self.env.num_agents):
                                    if agent_id in required_agents:
                                        # stag_id를 사냥할 때 reward를 받는 agent인지 확인
                                        if agent_id == stag_id:
                                            # my_stag: 내가 reward를 받을 stag
                                            agent_metrics[agent_id]['my_stag_opportunities'] += 1
                                        else:   
                                            # coop_stag: 타인이 reward를 받는 stag를 도와줄 수 있는 기회
                                            agent_metrics[agent_id]['coop_stag_opportunities'] += 1
                        
                        # Process other_stag opportunities (각 agent별로 자신이 required_agents에 포함되지 않은 stag)
                        if 'other_stags_by_agent' in huntable_objects:
                            for agent_id, other_stags in huntable_objects['other_stags_by_agent'].items():
                                # 각 other_stag 기회마다 카운트
                                for stag_id in other_stags.keys():
                                    agent_metrics[agent_id]['other_stag_opportunities'] += 1
                            
                        # Process stag hunting attempts
                        if 'stag_attempts' in info:
                            for stag_id, attempting_agents in info['stag_attempts'].items():
                                if not attempting_agents:
                                    continue
                                
                                # huntable_stags에 있는 경우 (my_stag, coop_stag)
                                if 'huntable_stags' in huntable_objects and stag_id in huntable_objects['huntable_stags']:
                                    stag_info = huntable_objects['huntable_stags'][stag_id]
                                    required_agents = stag_info.get('required_agents', [])
                                    # 각 attempting agent별로 분류
                                    for agent_id in attempting_agents:
                                        if agent_id in required_agents:
                                            if agent_id == stag_id:
                                                agent_metrics[agent_id]['my_stag_hunt_attempts'] += 1
                                            else:
                                                agent_metrics[agent_id]['coop_stag_hunt_attempts'] += 1
                                
                                # other_stags_by_agent에 있는 경우 (other_stag)
                                if 'other_stags_by_agent' in huntable_objects:
                                    for agent_id in attempting_agents:
                                        # 해당 agent의 other_stags에 이 stag_id가 있는지 확인
                                        agent_other_stags = huntable_objects['other_stags_by_agent'].get(agent_id, {})
                                        if stag_id in agent_other_stags:
                                            agent_metrics[agent_id]['other_stag_hunt_attempts'] += 1
                        
                        # Process hare opportunities and attempts
                        if 'huntable_hares' in huntable_objects:
                            for hare_info in huntable_objects['huntable_hares']:
                                hunting_agents = hare_info.get('hunting_agents', [])
                                for agent_id in hunting_agents:
                                    agents_with_hare_opportunity.add(agent_id)
                                    agent_metrics[agent_id]['hare_opportunities'] += 1
                        
                        # Process hare hunting attempts (agent-specific)
                        if 'hare_attempts' in info and 'huntable_hares' in huntable_objects:
                            for agent_id in info['hare_attempts']:
                                if agent_id in agents_with_hare_opportunity:
                                    agent_metrics[agent_id]['hare_hunt_attempts'] += 1

                        # Collect policy bias data for advanced evaluator (case-by-case) - only for PCI-MAPPO
                        if self.advanced_causal_evaluator is not None:
                            try:
                                # Convert current observations (before step update) back to numpy
                                obs_np = {aid: obs.detach().cpu().numpy() for aid, obs in observations.items()}
                                self.advanced_causal_evaluator._collect_policy_bias_data(
                                    observations=obs_np,
                                    actions=actions,
                                    info=info
                                )
                            except Exception as e:
                                self.logger.debug(f"Policy bias data collection failed: {e}")
                    
                    observations = next_observations
                    episode_length += 1
                    step_count += 1
                    done = any(dones.values())
                
                # Save animation if requested
                if save_animation:
                    if frames:
                        self.logger.info(f"Saving animation for episode {episode_count} with {len(frames)} frames")
                        self._save_episode_animation(frames, episode_count - 1, animation_dir)
                    else:
                        self.logger.warning(f"No frames captured for episode {episode_count}")
                
                # Store episode data
                for agent_id, total_reward in episode_reward.items():
                    test_rewards[agent_id].append(total_reward)
                test_lengths.append(episode_length)
                
                # Accumulate policy quality metrics (agent-specific)
                if evaluate_policy_quality:
                    for agent_id, agent_metrics in agent_metrics.items():
                        policy_metrics['agents'][agent_id]['my_stag_hunt_attempts'] += agent_metrics['my_stag_hunt_attempts']
                        policy_metrics['agents'][agent_id]['coop_stag_hunt_attempts'] += agent_metrics['coop_stag_hunt_attempts']
                        policy_metrics['agents'][agent_id]['other_stag_hunt_attempts'] += agent_metrics['other_stag_hunt_attempts']
                        policy_metrics['agents'][agent_id]['hare_hunt_attempts'] += agent_metrics['hare_hunt_attempts']
                        policy_metrics['agents'][agent_id]['my_stag_opportunities'] += agent_metrics['my_stag_opportunities']
                        policy_metrics['agents'][agent_id]['coop_stag_opportunities'] += agent_metrics['coop_stag_opportunities']
                        policy_metrics['agents'][agent_id]['other_stag_opportunities'] += agent_metrics['other_stag_opportunities']
                        policy_metrics['agents'][agent_id]['hare_opportunities'] += agent_metrics['hare_opportunities']
                    policy_metrics['total_episodes'] += 1
                    policy_metrics['total_steps'] = step_count
                
                mean_reward = np.mean(list(episode_reward.values()))
                self.logger.info(f"Episode {episode_count}: Reward = {mean_reward:.3f}, Length = {episode_length}, Steps = {step_count}/{advanced_causal_steps}")
        
        else:
            # Use episode-based collection for basic testing
            for episode in range(num_episodes):
                observations = self.env.reset()
                self.algorithm.reset_agents()
                observations = dict_to_tensor(observations, device=self.algorithm.device)
                
                episode_reward = defaultdict(float)
                episode_length = 0
                done = False
                
                # Animation frames storage
                frames = []
                
                while not done and episode_length < 1000:
                    # Render and capture frame for animation
                    if (render or save_animation) and hasattr(self.env, 'render'):
                        if save_animation:
                            # Capture frame for animation
                            frame = self.env.render(mode='rgb_array')
                            if frame is not None:
                                frames.append(frame.copy())
                                self.logger.debug(f"Frame {len(frames)} captured for episode {episode + 1}")
                            else:
                                self.logger.warning(f"Failed to capture frame for episode {episode + 1}")
                        else:
                            # Regular rendering
                            self.env.render()
                    
                    # Get availability of actions
                    available_actions = None
                    if hasattr(self.env, "get_avail_actions"):
                        avail = self.env.get_avail_actions()
                        available_actions = {i: avail[i] for i in range(len(avail))}

                    # Get actions (deterministic for testing)
                    with torch.no_grad():
                        actions = {}
                        for agent_id, obs in observations.items():
                            # Handle available actions
                            avail_actions_tensor = None
                            if available_actions is not None and agent_id in available_actions:
                                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.algorithm.device)

                            if avail_actions_tensor is not None:
                                action, _, = self.algorithm.agents[agent_id].get_action(obs, available_actions=avail_actions_tensor)
                            else:
                                action, _, = self.algorithm.agents[agent_id].get_action(obs)
                            
                            actions[agent_id] = action
                    
                    # Take step
                    next_observations, rewards, dones, info = self.env.step(actions)
                    next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)
                    
                    # Update statistics
                    for agent_id, reward in rewards.items():
                        episode_reward[agent_id] += reward
                    
                    observations = next_observations
                    episode_length += 1
                    done = any(dones.values())
                
                if save_replay and hasattr(self.eval_env, 'save_replay'):
                    self.logger.info(f"Saving replay for episode {episode + 1}")
                    self.eval_env.save_replay(dir=animation_dir)

                # Save animation if requested
                if save_animation:
                    if frames:
                        self.logger.info(f"Saving animation for episode {episode + 1} with {len(frames)} frames")
                        self._save_episode_animation(frames, episode, animation_dir)
                    else:
                        self.logger.warning(f"No frames captured for episode {episode + 1}")
                
                # Store episode data
                for agent_id, total_reward in episode_reward.items():
                    test_rewards[agent_id].append(total_reward)
                test_lengths.append(episode_length)
                
                mean_reward = np.mean(list(episode_reward.values()))
                self.logger.info(f"Episode {episode + 1}: Reward = {mean_reward:.3f}, Length = {episode_length}")
        
        # Log final statistics
        overall_mean = np.mean([np.mean(rewards) for rewards in test_rewards.values()])
        overall_std = np.std([np.mean(rewards) for rewards in test_rewards.values()])
        mean_length = np.mean(test_lengths)
        
        self.logger.info("Test Results:")
        self.logger.info(f"  Mean Reward: {overall_mean:.3f} ± {overall_std:.3f}")
        self.logger.info(f"  Mean Length: {mean_length:.1f}")
        
        for agent_id, rewards in test_rewards.items():
            agent_mean = np.mean(rewards)
            agent_std = np.std(rewards)
            self.logger.info(f"  Agent {agent_id}: {agent_mean:.3f} ± {agent_std:.3f}")
        
        # Calculate and log policy quality metrics (agent-specific)
        if evaluate_policy_quality and policy_metrics['total_episodes'] > 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("Policy Quality Evaluation (Stag-Hunt Environment)")
            self.logger.info("="*60)
            
            # Calculate derived metrics for each agent
            for agent_id, agent_metrics in policy_metrics['agents'].items():
                # my_stag hunting utilization
                if agent_metrics['my_stag_opportunities'] > 0:
                    my_stag_util = agent_metrics['my_stag_hunt_attempts'] / agent_metrics['my_stag_opportunities']
                else:
                    my_stag_util = 0.0
                
                # coop_stag hunting utilization
                if agent_metrics['coop_stag_opportunities'] > 0:
                    coop_stag_util = agent_metrics['coop_stag_hunt_attempts'] / agent_metrics['coop_stag_opportunities']
                else:
                    coop_stag_util = 0.0
                
                # other_stag hunting utilization
                if agent_metrics['other_stag_opportunities'] > 0:
                    other_stag_util = agent_metrics['other_stag_hunt_attempts'] / agent_metrics['other_stag_opportunities']
                else:
                    other_stag_util = 0.0
                
                # hare hunting utilization
                if agent_metrics['hare_opportunities'] > 0:
                    hare_util = agent_metrics['hare_hunt_attempts'] / agent_metrics['hare_opportunities']
                else:
                    hare_util = 0.0
                
                # Store calculated metrics
                agent_metrics.update({
                    'my_stag_hunting_utilization': my_stag_util,
                    'coop_stag_hunting_utilization': coop_stag_util,
                    'other_stag_hunting_utilization': other_stag_util,
                    'hare_hunting_utilization': hare_util
                })
                
                # Log metrics
                self.logger.info(f"\nAgent {agent_id}:")
                self.logger.info(f"  My Stag Utilization: {my_stag_util:.3f} ({agent_metrics['my_stag_hunt_attempts']}/{agent_metrics['my_stag_opportunities']})")
                self.logger.info(f"  Coop Stag Utilization: {coop_stag_util:.3f} ({agent_metrics['coop_stag_hunt_attempts']}/{agent_metrics['coop_stag_opportunities']})")
                self.logger.info(f"  Other Stag Utilization: {other_stag_util:.3f} ({agent_metrics['other_stag_hunt_attempts']}/{agent_metrics['other_stag_opportunities']})")
                self.logger.info(f"  Hare Utilization: {hare_util:.3f} ({agent_metrics['hare_hunt_attempts']}/{agent_metrics['hare_opportunities']})")
            
            self.logger.info(f"\nTotal Steps: {policy_metrics['total_steps']}, Episodes: {policy_metrics['total_episodes']}")
            self.logger.info("="*60)
            
            # Generate and save policy quality plots
            if self.save_dir is not None:
                self._plot_policy_quality_metrics(policy_metrics, self.save_dir)

            # Additionally, generate policy bias plot using advanced evaluator's collected data - only for PCI-MAPPO
            if self.advanced_causal_evaluator is not None:
                try:
                    advanced_eval_dir = os.path.join(self.save_dir, 'advanced_causal_evaluation') if self.save_dir is not None else None
                    if advanced_eval_dir is not None:
                        os.makedirs(advanced_eval_dir, exist_ok=True)
                        policy_bias_path = os.path.join(advanced_eval_dir, 'policy_bias.png')
                        stats_for_bias = self.advanced_causal_evaluator._compute_detailed_statistics()
                        self.advanced_causal_evaluator.plot_policy_bias(
                            statistics=stats_for_bias,
                            save_path=policy_bias_path,
                            return_figure=False
                        )
                        self.logger.info(f"Policy bias plot saved: {policy_bias_path}")
                        if self.use_wandb:
                            wandb.log({"policy/policy_bias": wandb.Image(policy_bias_path)}, step=self.iteration)
                except Exception as e:
                    self.logger.warning(f"Failed to generate policy bias plot: {e}")
        
        # Causal effect evaluation if requested (tsne and detailed_causal_effects.png only) - only for PCI-MAPPO
        causal_results = None
        if evaluate_advanced_causal and self.causal_evaluator is not None and self.advanced_causal_evaluator is not None:
            self.logger.info("\n" + "="*80)
            self.logger.info("Starting Causal Effect Evaluation")
            self.logger.info("="*80)

            # 1. Basic causal effect evaluation (for t-SNE)
            self.logger.info("\n[1/3] Running basic causal effect evaluation...")
            self.logger.info(f"Causal evaluation: target steps={advanced_causal_steps}")
            basic_causal_stats = self.causal_evaluator.evaluate_causal_effects(num_eval_steps=advanced_causal_steps)
            
            # 2. Advanced causal effect evaluation (for detailed_causal_effects.png)
            self.logger.info("\n[2/3] Running advanced causal effect evaluation...")
            self.advanced_causal_evaluator.num_eval_steps = advanced_causal_steps
            advanced_causal_stats = self.advanced_causal_evaluator.evaluate_detailed_causal_effects()
            
            # Save results if save_dir exists
            if self.save_dir is not None:
                advanced_eval_dir = os.path.join(self.save_dir, 'advanced_causal_evaluation')
                os.makedirs(advanced_eval_dir, exist_ok=True)
                
                # Save statistics
                self.causal_evaluator.save_results(basic_causal_stats, advanced_eval_dir)
                self.advanced_causal_evaluator.save_results(advanced_causal_stats, advanced_eval_dir)
                
                # Generate and save plots
                self.logger.info("\n[3/3] Generating visualizations...")
                
                # Detailed causal effect plot
                causal_plot_path = os.path.join(advanced_eval_dir, 'detailed_causal_effects.png')
                self.advanced_causal_evaluator.plot_detailed_causal_effects(
                    advanced_causal_stats,
                    save_path=causal_plot_path
                )
                
                # t-SNE analysis
                tsne_plot_path = os.path.join(advanced_eval_dir, 'observation_tsne.png')
                self.logger.info("Generating optimized t-SNE visualizations (single embedding for all plots)...")
                from ..evaluation.observation_analysis import create_tsne_plots
                create_tsne_plots(
                    self.causal_evaluator,
                    self.env,
                    save_path=tsne_plot_path,
                    perplexity=60,
                    use_huntable_cases=True,
                    n_clusters=None
                )
                
                self.logger.info(f"\nCausal evaluation results saved to: {advanced_eval_dir}")
                
                # Log to wandb if enabled
                if self.use_wandb:
                    self.logger.info("Logging causal evaluation to wandb...")
                    
                    # Log causal effect statistics
                    causal_stats = advanced_causal_stats.get('causal_effects', {})
                    for agent_key, agent_data in causal_stats.items():
                        for scenario, stats in agent_data.items():
                            wandb.log({
                                f"causal/{agent_key}/{scenario}/mean": stats['mean'],
                                f"causal/{agent_key}/{scenario}/std": stats['std'],
                                f"causal/{agent_key}/{scenario}/count": stats['count']
                            })
                    
                    # Log images
                    wandb.log({
                        "causal/detailed_effects": wandb.Image(causal_plot_path),
                        "causal/observation_tsne": wandb.Image(tsne_plot_path)
                    })
            
            causal_results = {
                'causal_effects': advanced_causal_stats
            }
            
            # 결과 요약 출력
            self.logger.info("="*80)
            self.logger.info("Causal Effect Evaluation Complete")
            self.logger.info("="*80)
        
        return {
            'mean_reward': overall_mean,
            'std_reward': overall_std,
            'mean_length': mean_length,
            'agent_rewards': {agent_id: np.mean(rewards) for agent_id, rewards in test_rewards.items()},
            'causal_evaluation': causal_results,
            'policy_quality_metrics': policy_metrics if evaluate_policy_quality else None
        }
    
    def _save_episode_animation(self, frames, episode, animation_dir):
        """Save episode animation as MP4 file."""
        if not frames:
            return
        # Check if ffmpeg is available
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            use_ffmpeg = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            use_ffmpeg = False
            self.logger.info("ffmpeg not available, using Pillow writer")
        
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-0.5, frames[0].shape[1] - 0.5)
            ax.set_ylim(-0.5, frames[0].shape[0] - 0.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Create animation
            def animate(frame_idx):
                ax.clear()
                ax.set_xlim(-0.5, frames[0].shape[1] - 0.5)
                ax.set_ylim(-0.5, frames[0].shape[0] - 0.5)
                ax.set_aspect('equal')
                ax.axis('off')
                # Flip the image vertically to correct orientation
                flipped_frame = np.flipud(frames[frame_idx])
                ax.imshow(flipped_frame, origin='lower')
                ax.set_title(f'Episode {episode + 1} - Step {frame_idx + 1}')
            
            # Create animation with faster timestep
            anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                         interval=100, repeat=False)
            
            if use_ffmpeg:
                # Try MP4 first
                try:
                    animation_path = os.path.join(animation_dir, f'episode_{episode + 1:03d}.mp4')
                    anim.save(animation_path, writer='ffmpeg', fps=5, bitrate=1800)
                    self.logger.info(f"Animation saved as MP4: {animation_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save as MP4: {e}")
                    raise e
            else:
                # Use GIF directly with faster speed
                animation_path = os.path.join(animation_dir, f'episode_{episode + 1:03d}.gif')
                anim.save(animation_path, writer='pillow', fps=10)
                self.logger.info(f"Animation saved as GIF: {animation_path}")
            
            # Close figure to free memory
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"Failed to save animation for episode {episode + 1}: {e}")
            # Final fallback: save individual frames as images
            try:
                frames_dir = os.path.join(animation_dir, f'episode_{episode + 1:03d}_frames')
                os.makedirs(frames_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
                    plt.imsave(frame_path, frame)
                self.logger.info(f"Frames saved as images: {frames_dir}")
            except Exception as e2:
                self.logger.error(f"Failed to save frames as images: {e2}")
    
    def _plot_policy_quality_metrics(self, policy_metrics, save_dir):
        """
        Generate and save policy quality visualization plots (agent-specific).
        Note: Other stag hunting opportunities is quite different from my stag and coop stag hunting opportunities.
              It is counted when an agent can reach to the other stag's position.
              Howvever, my/coop stag hunting opportunities is counted 
              when all required agents can reach to the my/coop stag's position.
        """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'policy_quality_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        num_agents = self.env.num_agents
        # Create figure with subplots: one row per agent
        fig, axes = plt.subplots(num_agents, 2, figsize=(15, 6 * num_agents))
        if num_agents == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Policy Quality Metrics (Agent-Specific)', fontsize=16, fontweight='bold')
        
        for agent_id in range(num_agents):
            agent_metrics = policy_metrics['agents'][agent_id]
            
            # Hunting utilization rates
            my_stag_util = agent_metrics['my_stag_hunting_utilization']
            coop_stag_util = agent_metrics['coop_stag_hunting_utilization']
            other_stag_util = agent_metrics['other_stag_hunting_utilization']
            hare_util = agent_metrics['hare_hunting_utilization']
            
            hunting_rates = [my_stag_util, coop_stag_util, other_stag_util, hare_util]
            hunting_labels = ['My Stag', 'Coop Stag', 'Other Stag', 'Hare']
            bars = axes[agent_id, 0].bar(hunting_labels, hunting_rates, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'], alpha=0.7)
            axes[agent_id, 0].set_title(f'Agent {agent_id}: Hunting Opportunity Utilization')
            axes[agent_id, 0].set_ylim(0, 1)
            axes[agent_id, 0].set_ylabel('Utilization Rate')
            
            # Add data count labels on bars
            attempts_list = [
                agent_metrics['my_stag_hunt_attempts'],
                agent_metrics['coop_stag_hunt_attempts'],
                agent_metrics['other_stag_hunt_attempts'],
                agent_metrics['hare_hunt_attempts']
            ]
            opps_list = [
                agent_metrics['my_stag_opportunities'],
                agent_metrics['coop_stag_opportunities'],
                agent_metrics['other_stag_opportunities'],
                agent_metrics['hare_opportunities']
            ]
            
            for bar, rate, attempts, opps in zip(bars, hunting_rates, attempts_list, opps_list):
                height = bar.get_height()
                axes[agent_id, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{attempts}/{opps}\n{rate:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Opportunities vs attempts
            opportunities = [agent_metrics['my_stag_opportunities'], 
                           agent_metrics['coop_stag_opportunities'],
                           agent_metrics['other_stag_opportunities'],
                           agent_metrics['hare_opportunities']]
            attempts = [agent_metrics['my_stag_hunt_attempts'],
                       agent_metrics['coop_stag_hunt_attempts'],
                       agent_metrics['other_stag_hunt_attempts'],
                       agent_metrics['hare_hunt_attempts']]
            
            x = np.arange(len(hunting_labels))
            width = 0.35
            
            bars1 = axes[agent_id, 1].bar(x - width/2, opportunities, width, label='Opportunities', alpha=0.8, color='#95a5a6')
            bars2 = axes[agent_id, 1].bar(x + width/2, attempts, width, label='Attempts', alpha=0.8, color='#34495e')
            axes[agent_id, 1].set_title(f'Agent {agent_id}: Opportunities vs Attempts')
            axes[agent_id, 1].set_ylabel('Count')
            axes[agent_id, 1].set_xticks(x)
            axes[agent_id, 1].set_xticklabels(hunting_labels)
            axes[agent_id, 1].legend()
            
            # Add count labels on bars
            for bars, values in [(bars1, opportunities), (bars2, attempts)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if height > 0:  # Only label if value > 0
                        axes[agent_id, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                       f'{value}', ha='center', va='bottom', fontsize=9)
        
        # Add overall data summary
        total_steps = policy_metrics.get('total_steps', 0)
        total_episodes = policy_metrics.get('total_episodes', 0)
        fig.text(0.5, 0.02, f'Data Collection: {total_steps} steps across {total_episodes} episodes', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'policy_quality_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Policy quality plots saved to: {plots_dir}")