import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .mlp import MLPBase


class ActionBridgeNetwork(nn.Module):
    """
    Action Bridge Network q_θ(z, a) for Doubly Robust causal inference.
    Provides IPW-like weights for the doubly robust correction term.
    Maps context variables (z) and full action vector to scalar output.
    """
    
    def __init__(
        self, 
        z_dim: int,  # z dimension = concatenated observations dim
        action_dim: int,   # single agent action dimension
        num_agents: int,   # number of agents
        hidden_dim: int = 64,
        action_embedding_dim: int = 8  # Action embedding dimension
    ):
        super(ActionBridgeNetwork, self).__init__()
        
        self.z_dim = z_dim 
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.action_embedding_dim = action_embedding_dim
        
        # Action embedding for better action representation
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)

        # Input is concatenation of context and all agents' action embeddings
        input_dim = z_dim + num_agents * action_embedding_dim

        # Action bridge network using MLPBase
        self.action_bridge_network = MLPBase(
            obs_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,  # 2 hidden layers
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        
        # Output layer for scalar action bridge value
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        z: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of action bridge network.
        
        Args:
            z: Context variables z [batch_size, z_dim]
            actions: Full action vector a = (a^1_t, a^2_t, ..., a^n_t) [batch_size, num_agents]
            
        Returns:
            action_bridge_output: Scalar action bridge values [batch_size, 1]
        """
        batch_size = z.shape[0]
        
        # Get action embeddings for all agents
        # actions: [batch_size, num_agents] -> action_embeds: [batch_size, num_agents, action_embedding_dim]
        action_embeds = self.action_embedding(actions)  # [batch_size, num_agents, action_embedding_dim]
        # Flatten action embeddings: [batch_size, num_agents * action_embedding_dim]
        action_embeds_flat = action_embeds.view(batch_size, -1)
        
        # Concatenate z and all action embeddings
        action_bridge_input = torch.cat([z, action_embeds_flat], dim=-1)
        
        # Forward through action bridge network
        encoded = self.action_bridge_network(action_bridge_input)  # [batch_size, hidden_dim]
        
        # Output scalar action bridge value
        action_bridge_output = self.output_layer(encoded)  # [batch_size, 1]
        return action_bridge_output
