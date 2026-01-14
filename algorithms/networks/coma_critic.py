import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .mlp import MLPBase

class COMACritic(nn.Module):
    """
    Centralized Critic for COMA.
    Takes joint observations and joint actions (except the agent's own action) as input,
    and outputs Q-values for all possible actions of the specific agent.
    
    Q(s, u^-i) -> Outputs vector of size |U|
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device

        # Joint state
        self.state_dim = obs_dim * num_agents

        # Input: state + joint actions + agent id
        input_dim = (
            self.state_dim
            + (num_agents * action_dim)
            + num_agents
        )

        self.critic = MLPBase(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True,
        )

        # Output Q-values for all actions of agent i
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_id: int,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch_size, num_agents * obs_dim]
            actions: [batch_size, num_agents, action_dim] (one-hot)
            agent_id: int

        Returns:
            q_values: [batch_size, action_dim]
        """
        batch_size = obs.shape[0]

        # -------- 핵심: 자기 행동 마스킹 --------
        actions_masked = actions.clone()
        actions_masked[:, agent_id, :] = 0.0

        # Flatten joint actions
        actions_flat = actions_masked.reshape(batch_size, -1)

        # Agent ID one-hot
        agent_id_one_hot = torch.zeros(
            batch_size, self.num_agents, device=obs.device
        )
        agent_id_one_hot[:, agent_id] = 1.0

        # Concatenate inputs
        inputs = torch.cat(
            [obs, actions_flat, agent_id_one_hot],
            dim=-1,
        )

        # Critic forward
        features = self.critic(inputs)
        q_values = self.output_layer(features)

        return q_values