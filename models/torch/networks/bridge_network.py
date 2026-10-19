import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .mlp import MLPBase

class OutcomeBridgeNetwork(nn.Module):
    """
    Bridge network h_φ for causal inference.
    Maps full action vector and confounding variable (w) to outcome.
    Uses one-hot encoding for actions (standard RL practice).
    """
    def __init__(
        self, 
        action_dim: int, 
        num_agents: int,
        w_dim: int,
        outcome_dim: int,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8, 
        x_dim: int = 0 
    ):
        super(OutcomeBridgeNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.w_dim = w_dim
        self.outcome_dim = outcome_dim
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        
        # Use one-hot encoding instead of nn.Embedding
        # Input is action one-hot and confounder
        input_dim = action_dim + w_dim + x_dim
        
        # Bridge network with 3 hidden layers
        self.mlp = MLPBase(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        self.output_layer = nn.Linear(hidden_dim, outcome_dim)

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
        # One-hot encode actions
        action_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [batch_size, action_dim]
  
        # Concatenate action one-hot and confounder
        if x is not None:
             bridge_input = torch.cat([action_onehot, w, x], dim=-1)
        else:
             bridge_input = torch.cat([action_onehot, w], dim=-1)
        
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

        # One-hot encode actions: [batch_size, num_agents, action_dim]
        action_onehot = F.one_hot(actions.long(), self.action_dim).float()
        action_onehot_flat = action_onehot.view(batch_size, -1)  # [batch_size, num_agents * action_dim]
        action_onehot_expanded = action_onehot_flat.unsqueeze(1).expand(-1, M, -1) # [batch_size, M, num_agents * action_dim]
        
        # Prepare bridge input: batch_size * (a', w^(m))
        if x is not None:
            bridge_input = torch.cat([action_onehot_expanded, w, x], dim=-1) 
        else:
            bridge_input = torch.cat([action_onehot_expanded, w], dim=-1) 
            
        bridge_input_reshaped = bridge_input.view(batch_size * M, -1) # [batch_size * M, num_agents * action_dim + w_dim]
        
        # Forward through network: [batch_size * M, outcome_dim]
        outcomes = self.bridge_network(bridge_input_reshaped)
        outcomes = outcomes.view(batch_size, M, self.outcome_dim) # shape: [batch_size, M, outcome_dim]     
        expected_outcome = outcomes.mean(dim=1) # Compute \hat r^(i,k)_t = (1/M) Σ^M_{m=1} h_φ(a', w^(m))
        
        return expected_outcome # shape: [batch_size, outcome_dim]

class NaivePredictor(nn.Module):
    """
    Simple Neural Network predicting the reward of complements 
    Using current the concat of complements observation \mathbf{o}^{-i]_t and 
    the action of the agent i.
    Uses one-hot encoding for actions (standard RL practice).
    """
    def __init__(
        self,
        action_dim: int,
        num_agents: int,
        x_dim: int,
        w_dim: int,
        output_dim: int = None,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8  # kept for API compat but unused
    ):
        super(NaivePredictor, self).__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.output_dim = output_dim if output_dim is not None else num_agents
        
        # Use one-hot encoding instead of nn.Embedding
        # Input is action one-hot and x
        input_dim = action_dim + w_dim + x_dim
        
        # Bridge network with 3 hidden layers
        self.mlp = MLPBase(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True
        )
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

        self.full_network = nn.Sequential(
            self.mlp,
            self.output_layer
        )
        
    def forward(
        self, 
        actions: torch.Tensor, 
        w: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of bridge network.
        """
        # One-hot encode actions
        action_onehot = F.one_hot(actions.long(), self.action_dim).float()  # [batch_size, action_dim]
  
        # Concatenate action one-hot and observation
        x = torch.cat([action_onehot, w, x], dim=-1)
        
        # Forward through network
        output = self.full_network(x)
        
        return output
