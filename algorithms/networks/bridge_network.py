import torch
import torch.nn as nn
from typing import Tuple, Optional
from .mlp import MLPBase

class OutcomeBridgeNetwork(nn.Module):
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
        action_embedding_dim: int = 8,
        x_dim: int = 0 # [ADDED] Covariates X: o^i_t
    ):
        super(OutcomeBridgeNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.w_dim = w_dim
        self.outcome_dim = outcome_dim
        self.hidden_dim = hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.x_dim = x_dim
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        # Input is an agent's action embedding and confounder
        input_dim = action_embedding_dim + w_dim + x_dim
        
        # Bridge network with 3 hidden layers
        self.mlp = MLPBase(
            input_dim=input_dim,
            hidden_size=2*hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        self.output_layer = nn.Linear(2*hidden_dim, outcome_dim)

        self.bridge_network = nn.Sequential(
            self.mlp,
            self.output_layer
        )

        
    def forward(
        self, 
        actions: torch.Tensor, 
        w: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of bridge network.
        """
        batch_size = actions.shape[0]
        action_embeds = self.action_embedding(actions)  # [batch_size, action_embedding_dim]
  
        # Concatenate all action embeddings and confounder
        if x is not None:
             bridge_input = torch.cat([action_embeds, w, x], dim=-1)
        else:
             bridge_input = torch.cat([action_embeds, w], dim=-1)
        
        # Forward through network
        outcome = self.bridge_network(bridge_input)
        
        return outcome
    
    def compute_expected_outcome(
        self,
        actions: torch.Tensor,
        w: torch.Tensor,
        x: Optional[torch.Tensor] = None, # [ADDED]
        M: int = 10
    ) -> torch.Tensor:
        """
        Compute expected outcome by averaging over M sampled confounders.
        Args:
            actions: Full action vector a = (a^1_t, a^2_t, ..., a^n_t) [batch_size, num_agents]
            w: Sampled confounders [batch_size, M, w_dim]
            x: Sampled observations [batch_size, M, x_dim]   
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
        if x is not None:
            bridge_input = torch.cat([action_embeds_expanded, w, x], dim=-1) 
        else:
            bridge_input = torch.cat([action_embeds_expanded, w], dim=-1) 
            
        bridge_input_reshaped = bridge_input.view(batch_size * M, -1) # [batch_size * M, num_agents * action_embedding_dim + w_dim]
        
        # Forward through network: [batch_size * M, outcome_dim]
        outcomes = self.bridge_network(bridge_input_reshaped)
        outcomes = outcomes.view(batch_size, M, self.outcome_dim) # shape: [batch_size, M, outcome_dim]     
        expected_outcome = outcomes.mean(dim=1) # Copmute \hat r^(i,k)_t = (1/M) Σ^M_{m=1} h_φ(a', w^(m))
        
        return expected_outcome # shape: [batch_size, outcome_dim]


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
        input_dim = z_dim + action_embedding_dim

        # Action bridge network using MLPBase
        self.action_bridge_network = MLPBase(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,  # 2 hidden layers
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)

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
        # actions: [batch_size, num_agents] -> action_embeds: [batch_size, action_embedding_dim]
        action_embeds = self.action_embedding(actions)  # [batch_size, action_embedding_dim]
        
        # Concatenate z and action embeddings
        action_bridge_input = torch.cat([z, action_embeds], dim=-1)
        
        # Forward through action bridge network
        encoded = self.action_bridge_network(action_bridge_input)  # [batch_size, hidden_dim]
        
        # Output scalar action bridge value
        action_bridge_output = self.output_layer(encoded)  # [batch_size, 1]
        return action_bridge_output

class RewardPredictor(nn.Module):
    """
    Simple Neural Network predicting the reward of complements 
    Using current the concat of complements observation \mathbf{o}^{-i]_t and 
    the action of the agent i
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        output_dim: int,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8
    ):
        super(RewardPredictor, self).__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.output_dim = output_dim
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        # Input is an agent's action embedding and confounder
        input_dim = action_embedding_dim + obs_dim
        
        # Bridge network with 3 hidden layers
        self.mlp = MLPBase(
            input_dim=input_dim,
            hidden_size=2*hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        self.output_layer = nn.Linear(2*hidden_dim, output_dim)

        self.bridge_network = nn.Sequential(
            self.mlp,
            self.output_layer
        )
        
    def forward(
        self, 
        actions: torch.Tensor, 
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of bridge network.
        """
        batch_size = actions.shape[0]
        action_embeds = self.action_embedding(actions)  # [batch_size, action_embedding_dim]
  
        # Concatenate all action embeddings and confounder
        x = torch.cat([action_embeds, obs], dim=-1)
        
        # Forward through network
        output = self.bridge_network(x)
        
        return output