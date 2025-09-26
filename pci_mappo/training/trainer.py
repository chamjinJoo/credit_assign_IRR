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

from ..algorithms import PCIMAPPOAlgorithm
from ..envs import MultiAgentEnv, dict_to_tensor
from ..utils import Logger, MetricsTracker, get_causal_evaluator


class PCIMAPPOTrainer:
    """
    Main trainer for PCI-MAPPO algorithm.
    Implements the training loop following the pseudocode.
    """
    
    def __init__(
        self,
        env: MultiAgentEnv,
        algorithm: PCIMAPPOAlgorithm,
        max_iterations: int = 1000,
        max_steps_per_iteration: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        save_dir: str = "./checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "pci-mappo",
        wandb_run_name: Optional[str] = None,
        device: str = "cpu"
    ):
        self.env = env
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
        
        # Initialize logging first
        self.logger = Logger()
        self.metrics_tracker = MetricsTracker()
        
        # Initialize causal effect evaluator
        CausalEffectEvaluator = get_causal_evaluator()
        self.causal_evaluator = CausalEffectEvaluator(
            algorithm=algorithm,
            env=env,
            num_eval_episodes=eval_episodes,
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
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "num_agents": env.num_agents,
                    "obs_dim": env.get_obs_dim(),
                    "action_dim": env.get_action_dim(),
                    "max_iterations": max_iterations,
                    "max_steps_per_iteration": max_steps_per_iteration,
                }
            )
            
            # Define metrics for better visualization
            wandb.define_metric("iteration")
            wandb.define_metric("causal_effect/*", step_metric="iteration")
            wandb.define_metric("rollout/*", step_metric="iteration")
            wandb.define_metric("update/*", step_metric="iteration")
            wandb.define_metric("eval/*", step_metric="iteration")
            
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
        env_name = self.env.__class__.__name__.lower()
        
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
        
        # Create metadata directory
        metadata_dir = os.path.dirname(self.save_dir)  # Remove 'checkpoints' from path
        
        metadata = {
            "simulation_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.env.__class__.__name__,
                "num_agents": self.env.num_agents,
                "obs_dim": self.env.get_obs_dim(),
                "action_dim": self.env.get_action_dim()
            },
            "training_config": {
                "max_iterations": self.max_iterations,
                "max_steps_per_iteration": self.max_steps_per_iteration,
                "log_interval": self.log_interval,
                "save_interval": self.save_interval,
                "eval_interval": self.eval_interval,
                "eval_episodes": self.eval_episodes,
                "device": self.device
            },
            "directories": {
                "base_save_dir": self.base_save_dir,
                "simulation_dir": self.save_dir,
                "checkpoint_dir": self.save_dir
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(metadata_dir, "simulation_config.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple readme
        readme_path = os.path.join(metadata_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"PCI-MAPPO Training Simulation\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {self.env.__class__.__name__}\n")
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
        self.logger.info("=" * 60)
        self.logger.info("PCI-MAPPO Training Configuration")
        self.logger.info("=" * 60)
        self.logger.info(f"Environment: {self.env.__class__.__name__}")
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
        step_count = 0
        episode_count = 0
        episode_rewards = defaultdict(list)
        episode_lengths = []
        
        while step_count < self.max_steps_per_iteration:
            # Reset environment and agents
            observations = self.env.reset()
            self.algorithm.reset_agents()
            # Convert observations to tensors
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            
            # Initialize previous observations (for first step, use zero vector since no previous observation exists)
            prev_observations = {agent_id: torch.zeros_like(obs) for agent_id, obs in observations.items()}
            
            episode_reward = defaultdict(float)
            episode_length = 0
            done = False
            
            while not done and step_count < self.max_steps_per_iteration:
                # Get actions from agents
                actions = self.algorithm.get_actions(observations)
                
                # Convert actions for environment
                env_actions = {agent_id: action for agent_id, (action, _) in actions.items()}
                
                # Take step in environment
                next_observations, rewards, dones, info = self.env.step(env_actions)
                
                # Convert to tensors
                next_observations = dict_to_tensor(next_observations, device=self.algorithm.device)
                
                # Store transition with proper causal structure (o_{t-1}, o_t, a_t, r_t, o_{t+1})
                self.algorithm.store_transition(
                    prev_observations=prev_observations,  # o_{t-1} (z_t)
                    observations=observations,            # o_t (w_t)
                    actions=env_actions,                  # a_t
                    rewards=rewards,                      # r_t
                    next_observations=next_observations,  # o_{t+1}
                    dones=dones
                )
                
                # Update episode statistics
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                
                # Update state: prev_observations = current observations, observations = next_observations
                prev_observations = observations
                observations = next_observations
                episode_length += 1
                step_count += 1
                self.total_steps += 1
                
                # Check if episode is done
                done = any(dones.values()) or episode_length >= 1000
            
            # Store episode statistics
            for agent_id, total_reward in episode_reward.items():
                episode_rewards[agent_id].append(total_reward)
            
            episode_lengths.append(episode_length)
            episode_count += 1
            
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
            'mean_episode_length': np.mean(episode_lengths)
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
        
        for episode in range(self.eval_episodes):
            # Reset environment and agents
            observations = self.env.reset()
            self.algorithm.reset_agents()
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            
            episode_reward = defaultdict(float)
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                # Get actions (no exploration for evaluation)
                with torch.no_grad():
                    actions = {}
                    for agent_id, obs in observations.items():
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
            
            # Store episode data
            for agent_id, total_reward in episode_reward.items():
                eval_rewards[agent_id].append(total_reward)
            eval_lengths.append(episode_length)
        
        # Compile evaluation metrics
        eval_info = {
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
        
        # Perform causal effect evaluation
        try:
            causal_stats = self.causal_evaluator.evaluate_causal_effects()
            eval_info['causal_effects'] = causal_stats
            
            # Log causal effect statistics
            self.logger.info("Causal Effect Statistics:")
            for scenario, stats in causal_stats.items():
                self.logger.info(f"  {scenario}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
                
        except Exception as e:
            self.logger.warning(f"Failed to perform causal effect evaluation: {e}")
            eval_info['causal_effects'] = None
        
        return eval_info
    
    def log_training_metrics(self, rollout_data: Dict[str, Any], update_info: Dict[str, float]):
        """Log training metrics."""
        # Log to console
        self.logger.info(f"Iteration {self.iteration}")
        self.logger.info(f"  Steps: {rollout_data['step_count']}")
        self.logger.info(f"  Episodes: {rollout_data['episode_count']}")
        self.logger.info(f"  Mean Episode Reward: {rollout_data['mean_episode_reward']:.3f}")
        self.logger.info(f"  Mean Episode Length: {rollout_data['mean_episode_length']:.1f}")
        
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
                'rollout/episodes': rollout_data['episode_count']
            }
            
            # Add agent-specific rewards
            for agent_id, rewards in rollout_data['episode_rewards'].items():
                wandb_data[f'rollout/agent_{agent_id}_mean_reward'] = np.mean(rewards)
            
            # Add update info
            for key, value in update_info.items():
                wandb_data[f'update/{key}'] = value
            
            # Add delta statistics for causal reward shaping
            # Group by scenario for better visualization
            delta_scenarios = ['no_reward', 'forage_reward', 'stag_reward']
            for scenario in delta_scenarios:
                mean_key = f'delta_{scenario}_mean'
                std_key = f'delta_{scenario}_std'
                count_key = f'delta_{scenario}_count'
                
                if mean_key in update_info:
                    # Use consistent naming for wandb to group them in one plot
                    wandb_data[f'causal_effect/delta_{scenario}'] = update_info[mean_key]
                    # Also log std and count separately for reference
                    if std_key in update_info:
                        wandb_data[f'causal_effect/delta_{scenario}_std'] = update_info[std_key]
                    if count_key in update_info:
                        wandb_data[f'causal_effect/delta_{scenario}_count'] = update_info[count_key]
            
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
                'eval/mean_reward': eval_info['eval_mean_reward'],
                'eval/std_reward': eval_info['eval_std_reward'],
                'eval/mean_length': eval_info['eval_mean_length'],
                'eval/min_reward': eval_info['eval_min_reward'],
                'eval/max_reward': eval_info['eval_max_reward'],
                'eval/best_reward': self.best_eval_reward
            }
            
            # Add causal effect metrics to wandb
            if 'causal_effects' in eval_info and eval_info['causal_effects'] is not None:
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
            'outcome_bridge_net': self.algorithm.outcome_bridge_net.state_dict(),
            'action_bridge_net': self.algorithm.action_bridge_net.state_dict(),
            'critic_kernel_z': self.algorithm.critic_kernel_z.state_dict(),
            'critic_kernel_w': self.algorithm.critic_kernel_w.state_dict(),
            'action_embedding_for_kernel': self.algorithm.action_embedding_for_kernel.state_dict(),
            'outcome_bridge_optimizer': self.algorithm.outcome_bridge_optimizer.state_dict(),
            'action_bridge_optimizer': self.algorithm.action_bridge_optimizer.state_dict(),
            'kernel_action_embed_optimizer': self.algorithm.kernel_action_embed_optimizer.state_dict(),
        }
        
        # Save agent states
        for agent_id, agent in self.algorithm.agents.items():
            state['agents'][agent_id] = {
                'policy_net': agent.policy_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
            }
        
        return state
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.iteration = checkpoint['iteration']
        self.total_steps = checkpoint['total_steps']
        self.best_eval_reward = checkpoint['best_eval_reward']
        
        # Load algorithm state
        algorithm_state = checkpoint['algorithm_state']
        
        # Load network states
        self.algorithm.outcome_bridge_net.load_state_dict(algorithm_state['outcome_bridge_net'])
        self.algorithm.action_bridge_net.load_state_dict(algorithm_state['action_bridge_net'])
        self.algorithm.critic_kernel_z.load_state_dict(algorithm_state['critic_kernel_z'])
        self.algorithm.critic_kernel_w.load_state_dict(algorithm_state['critic_kernel_w'])
        self.algorithm.action_embedding_for_kernel.load_state_dict(algorithm_state['action_embedding_for_kernel'])
        self.algorithm.outcome_bridge_optimizer.load_state_dict(algorithm_state['outcome_bridge_optimizer'])
        self.algorithm.action_bridge_optimizer.load_state_dict(algorithm_state['action_bridge_optimizer'])
        self.algorithm.kernel_action_embed_optimizer.load_state_dict(algorithm_state['kernel_action_embed_optimizer'])
        
        # Load agent states
        for agent_id, agent_state in algorithm_state['agents'].items():
            agent = self.algorithm.agents[int(agent_id)]
            agent.policy_net.load_state_dict(agent_state['policy_net'])
            agent.optimizer.load_state_dict(agent_state['optimizer'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from iteration {self.iteration}")
    
    def test_policy(self, num_episodes: int = 10, render: bool = False):
        """Test the trained policy."""
        self.logger.info(f"Testing policy for {num_episodes} episodes...")
        
        test_rewards = defaultdict(list)
        test_lengths = []
        
        for episode in range(num_episodes):
            observations = self.env.reset()
            self.algorithm.reset_agents()
            observations = dict_to_tensor(observations, device=self.algorithm.device)
            
            episode_reward = defaultdict(float)
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                if render and hasattr(self.env, 'render'):
                    self.env.render()
                
                # Get actions (deterministic for testing)
                with torch.no_grad():
                    actions = {}
                    for agent_id, obs in observations.items():
                        action, _, _ = self.algorithm.agents[agent_id].get_action(obs)
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
        
        return {
            'mean_reward': overall_mean,
            'std_reward': overall_std,
            'mean_length': mean_length,
            'agent_rewards': {agent_id: np.mean(rewards) for agent_id, rewards in test_rewards.items()}
        }
