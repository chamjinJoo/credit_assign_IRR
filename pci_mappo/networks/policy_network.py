import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict
from .mlp import MLPBase


class PolicyNetwork(nn.Module):
    """
    Policy Network following pseudocode: π_θ(a^i | o^i).
    Uses only observation information, no RNN.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64
    ):
        super(PolicyNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network using MLPBase
        self.policy_network = MLPBase(
            obs_dim=obs_dim,
            hidden_size=hidden_dim,
            layer_N=2,  # 2 hidden layers
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        
    def forward(
        self, 
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using only observation.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            
        Returns:
            action_logits: Action logits [batch_size, action_dim]
        """
        # Encode observations
        obs_encoded = self.policy_network(obs)
        
        # Apply policy head
        action_logits = self.policy_head(obs_encoded)

        return action_logits
    
    def get_action_and_log_prob(
        self, 
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.forward(obs)
        # Use logits directly for numerical stability
        dist = Categorical(logits=action_logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.forward(obs)
        
        # Use logits directly for numerical stability
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class CentralizedValueNetwork(nn.Module):
    """
    Centralized Value Network that takes individual agent observations as input.
    All agents share the same network parameters (parameter sharing).
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        hidden_dim: int = 64
    ):
        super(CentralizedValueNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Input dimension: individual agent observation
        input_dim = obs_dim
        
        # Value network using MLPBase (shared across all agents)
        self.value_network = MLPBase(
            obs_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,  # 2 hidden layers
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with individual agent observation.
        
        Args:
            obs: Individual agent observation [batch_size, obs_dim]
            
        Returns:
            value: State value for the agent [batch_size, 1]
        """
        # Encode observations
        obs_encoded = self.value_network(obs)
        
        # Apply value head
        value = self.value_head(obs_encoded)
        
        return value
