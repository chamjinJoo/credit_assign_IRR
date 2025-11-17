import torch
import torch.nn as nn
from .mlp import MLPBase

class DynamicsNetwork(nn.Module):
    """
    Dynamics network to predict next observations given current observations and actions.
    Usage: during computing causal reward shaping, precdicted next observations are used 
        for measuring effects of actions to other agents' values.
    Input: current observations and actions [B, N, obs_dim]
    Output: predicted next observations [B, N, obs_dim]
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
    ):
        super(DynamicsNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Dynamics network using MLPBase
        self.dynamics_network = MLPBase(
            input_dim=obs_dim * num_agents + action_dim * num_agents, 
            hidden_size=hidden_dim, 
            layer_N=2, 
            use_feature_normalization=True, 
            use_orthogonal=True, 
            use_ReLU=True,
            use_layer_norm=True
        )
        self.output_layer = nn.Linear(hidden_dim, obs_dim * num_agents)

    def forward(
        self,
        joint_obs: torch.Tensor,
        joint_actions_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of dynamics network.
        Args:
            joint_obs: Current joint observations. Accepts shape
                [batch_size, num_agents, obs_dim] or [batch_size, num_agents * obs_dim]
            joint_actions_onehot: Joint actions in one-hot form with shape
                [batch_size, num_agents, action_dim] or [batch_size, num_agents * action_dim]
        Returns:
            predicted_next_obs: Predicted next observations [batch_size, num_agents, obs_dim]
        """
        if joint_obs.dim() == 3:
            joint_obs_flat = joint_obs.view(joint_obs.size(0), -1)
        else:
            joint_obs_flat = joint_obs

        if joint_actions_onehot.dim() == 3:
            joint_actions_flat = joint_actions_onehot.view(joint_actions_onehot.size(0), -1)
        else:
            joint_actions_flat = joint_actions_onehot

        # Concatenate observations and actions
        input_tensor = torch.cat([joint_obs_flat, joint_actions_flat], dim=-1)
        
        # Forward through dynamics network
        x = self.dynamics_network(input_tensor)
        predicted_next_obs_flat = self.output_layer(x)
        predicted_next_obs = predicted_next_obs_flat.view(-1, self.num_agents, self.obs_dim)
        
        return predicted_next_obs