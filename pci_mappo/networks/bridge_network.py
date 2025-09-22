import torch
import torch.nn as nn
from typing import Tuple


class BridgeNetwork(nn.Module):
    """
    Bridge network h_φ for causal inference.
    Maps full action vector and confounding variable (w) to outcome.
    """
    
    def __init__(
        self, 
        action_dim: int, 
        num_agents: int,
        w_dim: int,
        outcome_dim: int,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8
    ):
        super(BridgeNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.w_dim = w_dim
        self.outcome_dim = outcome_dim
        self.hidden_dim = hidden_dim
        self.action_embedding_dim = action_embedding_dim
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        # Input is concatenation of all agents' action embeddings and confounder
        input_dim = num_agents * action_embedding_dim + w_dim
        
        # Bridge network with 3 hidden layers
        self.bridge_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outcome_dim)
        )
        
    def forward(
        self, 
        actions: torch.Tensor, 
        w: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of bridge network.
        Args:
            actions: Full action vector a = (a^1_t, a^2_t, ..., a^n_t) [batch_size, num_agents]
            w: Confounding variable w [batch_size, w_dim]
            
        Returns:
            outcome: Predicted outcomes μ for all agents [batch_size, num_agents]
                    Each column i contains the predicted reward for agent i
        """
        batch_size = actions.shape[0]
        # Get action embeddings for all agents
        # actions: [batch_size, num_agents] -> action_embeds: [batch_size, num_agents, action_embedding_dim]
        action_embeds = self.action_embedding(actions)  # [batch_size, num_agents, action_embedding_dim]
        
        # Flatten action embeddings: [batch_size, num_agents * action_embedding_dim]
        action_embeds_flat = action_embeds.view(batch_size, -1)
        
        # Concatenate all action embeddings and confounder
        bridge_input = torch.cat([action_embeds_flat, w], dim=-1)
        
        # Forward through network
        outcome = self.bridge_network(bridge_input)
        
        return outcome
    
    def compute_expected_outcome(
        self,
        actions: torch.Tensor,
        w: torch.Tensor,
        M: int = 10
    ) -> torch.Tensor:
        """
        Compute expected outcome by averaging over M sampled confounders.
        Args:
            actions: Full action vector a = (a^1_t, a^2_t, ..., a^n_t) [batch_size, num_agents]
            w: Sampled confounders [batch_size, M, w_dim]
            M: Number of samples
            
        Returns:
            expected_outcome: Expected rewards for all agents \hat r^(i,k)_t [batch_size, num_agents]
                             Each column i contains the expected reward for agent i
        """
        batch_size = actions.shape[0] # actions shape: [batch_size, num_agents]

        action_embeds = self.action_embedding(actions)  # [batch_size, num_agents, action_embedding_dim]
        action_embeds_flat = action_embeds.view(batch_size, -1) # [batch_size, num_agents * action_embedding_dim]
        action_embeds_expanded = action_embeds_flat.unsqueeze(1).expand(-1, M, -1) # [batch_size, M, num_agents * action_embedding_dim]
        
        # Prepare bridge input: batch_size * (a', w^(m))
        bridge_input = torch.cat([action_embeds_expanded, w], dim=-1) # [batch_size, M, num_agents * action_embedding_dim + w_dim]
        bridge_input_reshaped = bridge_input.view(batch_size * M, -1) # [batch_size * M, num_agents * action_embedding_dim + w_dim]
        
        # Forward through network: [batch_size * M, outcome_dim]
        outcomes = self.bridge_network(bridge_input_reshaped)
        outcomes = outcomes.view(batch_size, M, self.outcome_dim) # shape: [batch_size, M, outcome_dim]     
        expected_outcome = outcomes.mean(dim=1) # Copmute \hat r^(i,k)_t = (1/M) Σ^M_{m=1} h_φ(a', w^(m))
        
        return expected_outcome # shape: [batch_size, outcome_dim]