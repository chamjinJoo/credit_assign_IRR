import torch
from typing import Dict, List, Optional
from collections import defaultdict


class ExperienceBuffer:
    """
    Experience buffer with FIFO queue behavior for storing transitions.
    Supports both main policy buffer and causal inference buffer.
    """
    
    def __init__(
        self,
        max_buffer_size: int = 10000,
        max_causal_buffer_size: int = 2000
    ):
        self.max_buffer_size = max_buffer_size
        self.max_causal_buffer_size = max_causal_buffer_size
        
        # Main buffer for policy learning
        self.buffer = {'transitions': []}
        
        # Separate buffer for causal inference data
        self.causal_buffer = {'transitions': []}
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],  # o_{t-1}
        observations: Dict[int, torch.Tensor],       # o_t
        actions: Dict[int, int],                     # a_t
        rewards: Dict[int, float],                   # r_t
        next_observations: Dict[int, torch.Tensor],  # o_{t+1}
        dones: Dict[int, bool],
        hidden_states: Dict[int, torch.Tensor]
    ):
        """
        Store transition in buffer with FIFO queue behavior.
        Store (o_{t-1}, o_t, a_t, r_t, o_{t+1}) tuple to maintain proper causal relationships.
        
        Args:
            prev_observations: Dict mapping agent_id to previous observation tensor (o_{t-1})
            observations: Dict mapping agent_id to current observation tensor (o_t)
            actions: Dict mapping agent_id to action (a_t)
            rewards: Dict mapping agent_id to reward (r_t)
            next_observations: Dict mapping agent_id to next observation tensor (o_{t+1})
            dones: Dict mapping agent_id to done flag
            hidden_states: Dict mapping agent_id to hidden state tensor
        """
        # Store complete system state with proper causal structure
        transition = {
            'bold_o_t_minus_1': prev_observations,  # {o^1_{t-1}, ..., o^n_{t-1}} (z_t)
            'bold_o_t': observations,                # {o^1_t, ..., o^n_t} (w_t)
            'bold_a_t': actions,                     # {a^1_t, ..., a^n_t}  
            'bold_r_t': rewards,                     # {r^1_t, ..., r^n_t}
            'next_observations': next_observations,  # {o^1_{t+1}, ..., o^n_{t+1}}
            'dones': dones,
            'hidden_states': hidden_states,
        }
        
        # Store in main buffer with FIFO queue behavior
        self.buffer['transitions'].append(transition)
        # Maintain FIFO queue for main buffer
        if len(self.buffer['transitions']) > self.max_buffer_size:
            self.buffer['transitions'].pop(0)  # Remove oldest data
    
    def clear_buffer(self):
        """Clear main experience buffer."""
        self.buffer['transitions'] = []
    
    def clear_causal_buffer(self):
        """Clear causal experience buffer."""
        self.causal_buffer['transitions'] = []
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes and limits for monitoring."""
        return {
            'main_buffer_size': len(self.buffer['transitions']),
            'main_buffer_limit': self.max_buffer_size,
            'causal_buffer_size': len(self.causal_buffer['transitions']),
            'causal_buffer_limit': self.max_causal_buffer_size,
        }
    
    def get_main_buffer_transitions(self) -> List[Dict]:
        """Get all transitions from main buffer."""
        return self.buffer['transitions']
    
    def get_causal_buffer_transitions(self) -> List[Dict]:
        """Get all transitions from causal buffer."""
        return self.causal_buffer['transitions']
    
    def is_main_buffer_empty(self) -> bool:
        """Check if main buffer is empty."""
        return len(self.buffer['transitions']) == 0
    
    def is_causal_buffer_empty(self) -> bool:
        """Check if causal buffer is empty."""
        return len(self.causal_buffer['transitions']) == 0
    
    def get_main_buffer_size(self) -> int:
        """Get current size of main buffer."""
        return len(self.buffer['transitions'])
    
    def get_causal_buffer_size(self) -> int:
        """Get current size of causal buffer."""
        return len(self.causal_buffer['transitions'])
