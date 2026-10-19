import torch
from typing import Dict, List, Optional
from collections import defaultdict


class TransitionBuffer:
    """
    Transition buffer with FIFO queue behavior for storing transitions.
    Stores individual transitions for batch-based training algorithms.
    """
    
    def __init__(
        self,
        max_buffer_size: int = 10000
    ):
        self.max_buffer_size = max_buffer_size
        
        # Main buffer for policy learning
        self.buffer = {'transitions': []}
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],  # o_{t-1}
        prev_rewards: Dict[int, float],              # r_{t-1} (z_t for reward proxy)
        prev_actions: Dict[int, int],                # a_{t-1} (z_t for action proxy)
        observations: Dict[int, torch.Tensor],       # o_t
        actions: Dict[int, int],                     # a_t
        rewards: Dict[int, float],                   # r_t
        next_observations: Dict[int, torch.Tensor],  # o_{t+1}
        dones: Dict[int, bool],
        masks: Dict[int, float],
        hidden_states: Dict[int, torch.Tensor],
        available_actions: Optional[Dict[int, List[int]]] = None,
        log_probs: Optional[Dict[int, float]] = None,
        info: Dict = None
    ):
        """
        Store transition in buffer with FIFO queue behavior.
        Store (o_{t-1}, r_{t-1}, a_{t-1}, o_t, a_t, r_t, o_{t+1}) tuple to maintain proper causal relationships.
        
        Args:
            prev_observations: Dict mapping agent_id to previous observation tensor (o_{t-1})
            prev_rewards: Dict mapping agent_id to previous reward (r_{t-1})
            prev_actions: Dict mapping agent_id to previous action (a_{t-1})
            observations: Dict mapping agent_id to current observation tensor (o_t)
            actions: Dict mapping agent_id to action (a_t)
            rewards: Dict mapping agent_id to reward (r_t)
            next_observations: Dict mapping agent_id to next observation tensor (o_{t+1})
            dones: Dict mapping agent_id to done flag
            masks: Dict mapping agent_id to mask value (1.0 for valid, 0.0 for done/reset)
            hidden_states: Dict mapping agent_id to hidden state tensor
            available_actions: Dict mapping agent_id to available actions (optional)
            info: Additional information from environment (e.g., cooperation info)
        """
        # Store complete system state with proper causal structure
        transition = {
            'bold_o_t_minus_1': prev_observations,  # {o^1_{t-1}, ..., o^n_{t-1}}
            'bold_r_t_minus_1': prev_rewards,       # {r^1_{t-1}, ..., r^n_{t-1}} 
            'bold_a_t_minus_1': prev_actions,       # {a^1_{t-1}, ..., a^n_{t-1}} 
            'bold_o_t': observations,                # {o^1_t, ..., o^n_t} (w_t)
            'bold_a_t': actions,                     # {a^1_t, ..., a^n_t}  
            'bold_r_t': rewards,                     # {r^1_t, ..., r^n_t}
            'next_observations': next_observations,  # {o^1_{t+1}, ..., o^n_{t+1}}
            'dones': dones,
            'masks': masks,
            'hidden_states': hidden_states,
            'available_actions': available_actions,
            'log_probs': log_probs,
            'info': info if info is not None else {},  # Environment info (e.g., stag_hunts, hare_hunts)
        }
        
        # Store in main buffer with FIFO queue behavior
        self.buffer['transitions'].append(transition)
        # Maintain FIFO queue for main buffer
        if len(self.buffer['transitions']) > self.max_buffer_size:
            self.buffer['transitions'].pop(0)  # Remove oldest data
    
    def clear_buffer(self):
        """Clear main experience buffer."""
        self.buffer['transitions'] = []
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes and limits for monitoring."""
        return {
            'main_buffer_size': len(self.buffer['transitions']),
            'main_buffer_limit': self.max_buffer_size,
        }
    
    def get_main_buffer_transitions(self) -> List[Dict]:
        """Get all transitions from main buffer."""
        return self.buffer['transitions']
    
    def is_main_buffer_empty(self) -> bool:
        """Check if main buffer is empty."""
        return len(self.buffer['transitions']) == 0
    
    def get_main_buffer_size(self) -> int:
        """Get current size of main buffer."""
        return len(self.buffer['transitions'])


class EpisodeBuffer:
    """
    Episode buffer for storing complete episodes.
    Designed for algorithms like COMA that train on episode-level data.
    """
    
    def __init__(
        self,
        max_episodes: int = 100
    ):
        """
        Initialize episode buffer.
        
        Args:
            max_episodes: Maximum number of complete episodes to store
        """
        self.max_episodes = max_episodes
        
        # Current episodes being built (keyed by env_id, default 0 for single env)
        self.current_episodes: Dict[int, List[Dict]] = defaultdict(list)
        
        # Completed episodes
        self.completed_episodes: List[List[Dict]] = []
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        hidden_states: Dict[int, torch.Tensor],
        info: Dict = None,
        masks: Dict[int, float] = None,
        available_actions: Optional[Dict[int, List[int]]] = None,
        log_probs: Optional[Dict[int, float]] = None,
        env_id: int = 0
    ):
        """
        Store transition in current episode for specific environment.
        
        Args:
            prev_observations: Dict mapping agent_id to previous observation tensor
            prev_rewards: Dict mapping agent_id to previous reward
            prev_actions: Dict mapping agent_id to previous action
            observations: Dict mapping agent_id to current observation tensor
            actions: Dict mapping agent_id to action
            rewards: Dict mapping agent_id to reward
            next_observations: Dict mapping agent_id to next observation tensor
            dones: Dict mapping agent_id to done flag
            hidden_states: Dict mapping agent_id to hidden state tensor
            info: Additional information from environment
            masks: Dict mapping agent_id to mask value (for RNN)
            available_actions: Dict mapping agent_id to available actions (optional)
            env_id: Environment ID (default 0)
        """
        transition = {
            'bold_o_t_minus_1': prev_observations,
            'bold_r_t_minus_1': prev_rewards,
            'bold_a_t_minus_1': prev_actions,
            'bold_o_t': observations,
            'bold_a_t': actions,
            'bold_r_t': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'hidden_states': hidden_states,
            'masks': masks if masks is not None else {},
            'available_actions': available_actions,
            'log_probs': log_probs,
            'info': info if info is not None else {},
        }
        
        self.current_episodes[env_id].append(transition)
    
    def finish_episode(self, env_id: int = 0) -> Optional[List[Dict]]:
        """
        Finish current episode for specific environment and move it to completed episodes.
        
        Args:
            env_id: Environment ID (default 0)
            
        Returns:
            Completed episode (list of transitions) if episode is not empty, None otherwise
        """
        if len(self.current_episodes[env_id]) == 0:
            return None
        
        completed_episode = self.current_episodes[env_id].copy()
        self.current_episodes[env_id] = []
        
        # Add to completed episodes
        self.completed_episodes.append(completed_episode)
        
        # Maintain max episodes limit (FIFO)
        if len(self.completed_episodes) > self.max_episodes:
            self.completed_episodes.pop(0)
        
        return completed_episode
    
    def get_completed_episodes(self) -> List[List[Dict]]:
        """
        Get all completed episodes.
        
        Returns:
            List of completed episodes, where each episode is a list of transitions
        """
        return self.completed_episodes.copy()
    
    def clear_completed_episodes(self):
        """Clear all completed episodes."""
        self.completed_episodes = []
    
    def clear_current_episode(self):
        """Clear all current episodes."""
        self.current_episodes = defaultdict(list)
    
    def clear_all(self):
        """Clear both current episode and completed episodes."""
        self.clear_current_episode()
        self.clear_completed_episodes()
    
    def has_completed_episodes(self) -> bool:
        """Check if there are any completed episodes."""
        return len(self.completed_episodes) > 0
    
    def has_incomplete_episode(self) -> bool:
        """Check if there is an incomplete episode in any current_episode."""
        return any(len(ep) > 0 for ep in self.current_episodes.values())
    
    def get_num_completed_episodes(self) -> int:
        """Get number of completed episodes."""
        return len(self.completed_episodes)
    
    def get_current_episode_length(self, env_id: int = 0) -> int:
        """Get length of current episode being built for env_id."""
        return len(self.current_episodes[env_id])
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get buffer information for monitoring."""
        return {
            'num_completed_episodes': len(self.completed_episodes),
            'current_episode_length_mean': sum(len(ep) for ep in self.current_episodes.values()) / max(1, len(self.current_episodes)),
            'has_incomplete_episode': 1 if self.has_incomplete_episode() else 0,
            'max_episodes': self.max_episodes,
        }

    def clear_buffer(self):
        """Clear buffer (alias for clear_all to match TransitionBuffer interface)."""
        self.clear_all()

