"""
Parallel Environment Wrapper for Multi-Agent RL

This module provides wrappers for parallel rollout collection:
1. ParallelEnv: Simple in-process parallel environments (existing)
2. ParallelEnvWrapper: Multiprocessing-based parallel environments (new)

The multiprocessing version is based on PyMARL's parallel_runner.py:
https://github.com/oxwhirl/pymarl/blob/master/src/runners/parallel_runner.py
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from multiprocessing import Pipe, Process
import cloudpickle


class ParallelEnv:
    """
    Wrapper for managing n independent environments in parallel (in-process).
    
    Each environment runs independently with its own state, episode lifecycle,
    and reset conditions. This enables efficient data collection from multiple
    environments simultaneously.
    
    Args:
        env_fn: Callable that creates a new environment instance
        n_envs: Number of parallel environments to create
    """
    
    def __init__(self, env_fn: Callable, n_envs: int):
        """Initialize parallel environments."""
        self.n_envs = n_envs
        self.envs = [env_fn() for _ in range(n_envs)]
        
        # Get environment properties from first environment
        self.num_agents = self.envs[0].num_agents
        self.obs_dim = self.envs[0].get_obs_dim()
        self.action_dim = self.envs[0].get_action_dim()
        
    def reset(self, env_idx: Optional[int] = None) -> Any:
        """
        Reset environment(s).
        
        Args:
            env_idx: If None, reset all environments. Otherwise, reset specific environment.
            
        Returns:
            If env_idx is None: List of initial observations from all environments
            Otherwise: Initial observations from specified environment
        """
        if env_idx is None:
            # Reset all environments
            return [env.reset() for env in self.envs]
        else:
            # Reset specific environment
            return self.envs[env_idx].reset()
    
    def step(self, actions_list: List[Dict[int, int]]) -> List[Tuple[Dict, Dict, Dict, Dict]]:
        """
        Execute one step in all environments.
        
        Args:
            actions_list: List of action dictionaries, one per environment
                         Each dict maps agent_id to action
        
        Returns:
            List of (observations, rewards, dones, info) tuples, one per environment
        """
        if len(actions_list) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} action dicts, got {len(actions_list)}")
        
        results = []
        for env, actions in zip(self.envs, actions_list):
            obs, rewards, dones, info = env.step(actions)
            results.append((obs, rewards, dones, info))
        
        return results
    
    def get_avail_actions(self) -> List[Optional[List[List[int]]]]:
        """
        Get available actions for all environments.
        
        Returns:
            List of available actions per environment. Each element is either:
            - None if environment doesn't support action masking
            - List[List[int]] where outer list is per agent, inner list is available actions
        """
        avail_actions_list = []
        for env in self.envs:
            if hasattr(env, 'get_avail_actions'):
                avail_actions_list.append(env.get_avail_actions())
            else:
                avail_actions_list.append(None)
        
        return avail_actions_list
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Multiprocessing-based Parallel Environment Wrapper
# ============================================================================

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).
    From OpenAI Baselines:
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
    """
    def __init__(self, x):
        self.x = x
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def env_worker(remote, env_fn_wrapper):
    """
    Worker process for running a single environment.
    
    Args:
        remote: Pipe connection for communication with parent process
        env_fn_wrapper: CloudpickleWrapper containing environment creation function
    """
    # Create environment
    env = env_fn_wrapper.x()
    
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
            
        if cmd == "step":
            # Take a step in the environment
            actions = data
            next_observations, rewards, dones, info = env.step(actions)
            
            # Check if episode is done
            episode_done = any(dones.values())

            # [OPTIMIZED] Fetch avail_actions here to avoid a separate IPC round-trip
            if episode_done:
                reset_obs = env.reset()
                # After reset, get avail_actions for the NEW episode
                if hasattr(env, "get_avail_actions"):
                    avail = env.get_avail_actions()
                    avail_actions = {i: avail[i] for i in range(len(avail))}
                else:
                    avail_actions = None
                remote.send({
                    "observations": next_observations,
                    "rewards": rewards,
                    "dones": dones,
                    "info": info,
                    "episode_done": True,
                    "reset_observations": reset_obs,
                    "avail_actions": avail_actions,         # avail_actions after reset
                })
            else:
                if hasattr(env, "get_avail_actions"):
                    avail = env.get_avail_actions()
                    avail_actions = {i: avail[i] for i in range(len(avail))}
                else:
                    avail_actions = None
                remote.send({
                    "observations": next_observations,
                    "rewards": rewards,
                    "dones": dones,
                    "info": info,
                    "episode_done": False,
                    "avail_actions": avail_actions,
                })
                
        elif cmd == "reset":
            observations = env.reset()
            # [OPTIMIZED] Also return avail_actions after reset
            if hasattr(env, "get_avail_actions"):
                avail = env.get_avail_actions()
                avail_actions = {i: avail[i] for i in range(len(avail))}
            else:
                avail_actions = None
            remote.send({"observations": observations, "avail_actions": avail_actions})
            
        elif cmd == "get_avail_actions":
            # For SMAC environments
            if hasattr(env, "get_avail_actions"):
                avail_actions = env.get_avail_actions()
                # Convert List[List[int]] to Dict[int, List[int]]
                avail_dict = {i: avail_actions[i] for i in range(len(avail_actions))}
                remote.send(avail_dict)
            else:
                remote.send(None)
                
        elif cmd == "get_env_info":
            env_info = {
                "num_agents": env.num_agents,
                "obs_dim": env.get_obs_dim(),
                "action_dim": env.get_action_dim()
            }
            remote.send(env_info)
            
        elif cmd == "close":
            remote.close()
            break
            
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class ParallelEnvWrapper:
    """
    Wrapper for running multiple environments in parallel using multiprocessing.
    
    This wrapper creates n_rollout_threads separate processes, each running an independent
    environment. Communication is done via Pipes.
    
    Based on PyMARL's parallel_runner.py.
    """
    
    def __init__(self, env_fn: Callable, n_rollout_threads: int = 1):
        """
        Initialize parallel environment wrapper.
        
        Args:
            env_fn: Function that creates an environment instance
            n_rollout_threads: Number of parallel environments to run
        """
        self.n_rollout_threads = n_rollout_threads
        self.env_fn = env_fn
        
        # Create pipes for communication
        self.parent_conns, self.worker_conns = zip(*[
            Pipe() for _ in range(n_rollout_threads)
        ])
        
        # Create worker processes
        self.processes = []
        for rank, worker_conn in enumerate(self.worker_conns):
            # Create a wrapper function that sets the seed based on rank
            def seeded_env_fn(r=rank):
                return env_fn(r)
                
            process = Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(seeded_env_fn))
            )
            process.daemon = True
            process.start()
            self.processes.append(process)
        
        # Get environment info from first environment
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        
        self.num_agents = self.env_info["num_agents"]
        self.obs_dim = self.env_info["obs_dim"]
        self.action_dim = self.env_info["action_dim"]
    
    def reset(self) -> Tuple[List[Dict[int, np.ndarray]], List[Optional[Dict[int, List[int]]]]]:
        """
        Reset all environments.

        Returns:
            observations_batch: List of observation dicts, length = n_rollout_threads
            avail_actions_batch: List of avail_actions dicts (or None), length = n_rollout_threads
        """
        # Send reset command to all environments
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        
        # Collect observations + avail_actions from all environments
        observations_batch = []
        avail_actions_batch = []
        for parent_conn in self.parent_conns:
            result = parent_conn.recv()
            observations_batch.append(result["observations"])
            avail_actions_batch.append(result["avail_actions"])
        
        return observations_batch, avail_actions_batch
    
    def step(
        self,
        actions_batch: List[Dict[int, int]]
    ) -> Tuple[List[Dict[int, np.ndarray]], List[Dict[int, float]],
               List[Dict[int, bool]], List[Dict],
               List[Optional[Dict[int, List[int]]]]]:
        """
        Take a step in all environments.

        [OPTIMIZED] avail_actions are now fetched inside the worker process
        alongside the step, eliminating a separate IPC round-trip per step.

        Args:
            actions_batch: List of action dicts, length = n_rollout_threads
                          Each element is Dict[agent_id, action]

        Returns:
            observations_batch:  List[Dict[int, np.ndarray]]
            rewards_batch:       List[Dict[int, float]]
            dones_batch:         List[Dict[int, bool]]
            infos_batch:         List[Dict]
            avail_actions_batch: List[Optional[Dict[int, List[int]]]]
        """
        # Send step command to all environments
        for parent_conn, actions in zip(self.parent_conns, actions_batch):
            parent_conn.send(("step", actions))
        
        # Collect results from all environments
        observations_batch = []
        rewards_batch = []
        dones_batch = []
        infos_batch = []
        avail_actions_batch = []
        
        for parent_conn in self.parent_conns:
            result = parent_conn.recv()
            
            # If episode is done, use reset observations for next step
            if result["episode_done"]:
                observations_batch.append(result["reset_observations"])
            else:
                observations_batch.append(result["observations"])
            
            rewards_batch.append(result["rewards"])
            dones_batch.append(result["dones"])
            infos_batch.append(result["info"])
            avail_actions_batch.append(result["avail_actions"])
        
        return observations_batch, rewards_batch, dones_batch, infos_batch, avail_actions_batch
    
    def get_avail_actions_batch(self) -> List[Optional[Dict[int, List[int]]]]:
        """
        Get available actions from all environments (explicit IPC call).

        NOTE: This method is kept for backward compatibility but should be
        avoided in the main training loop. After the optimization, avail_actions
        are returned directly from step() and reset(), so this method is only
        needed for the very first step before any step() has been called AND
        reset() has not been called yet.

        Returns:
            List of available actions dicts, length = n_rollout_threads
            Each element is Dict[agent_id, List[int]] or None if not applicable
        """
        # Send command to all environments
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_avail_actions", None))
        
        # Collect results
        avail_actions_batch = []
        for parent_conn in self.parent_conns:
            avail = parent_conn.recv()
            avail_actions_batch.append(avail)
        
        return avail_actions_batch
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim
    
    def close(self):
        """Close all environment processes."""
        # Send close command to all environments
        for parent_conn in self.parent_conns:
            try:
                parent_conn.send(("close", None))
            except:
                pass
        
        # Wait for all processes to finish
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        
        # Close all connections
        for parent_conn in self.parent_conns:
            parent_conn.close()
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()
