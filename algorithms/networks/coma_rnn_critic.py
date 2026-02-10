import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .mlp import MLPBase


class COMARNNCritic(nn.Module):
    """
    Centralized Critic for COMA with RNN-based policy.
    
    Based on pymarl implementation:
    - Input: joint_obs || agent_obs || masked_actions || last_actions || agent_id
    - Output: Q-values for all actions of the specific agent
    
    Note: Uses joint observation instead of state for partially observable environments.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Joint observation as state substitute
        self.joint_obs_dim = obs_dim * num_agents
        
        # Input dimensions (pymarl style):
        # - joint_obs: obs_dim * num_agents (state substitute)
        # - agent_obs: obs_dim (individual observation)
        # - actions: action_dim * num_agents (current joint actions, agent's own masked)
        # - last_actions: action_dim * num_agents (previous joint actions)
        # - agent_id: num_agents (one-hot)
        input_dim = (
            self.joint_obs_dim +       # joint observation
            obs_dim +                   # agent's individual observation
            action_dim * num_agents +   # current actions (one-hot, masked)
            action_dim * num_agents +   # last actions (one-hot)
            num_agents                  # agent id (one-hot)
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        joint_obs: torch.Tensor,      # [batch, num_agents * obs_dim]
        agent_obs: torch.Tensor,       # [batch, obs_dim]
        actions_onehot: torch.Tensor,  # [batch, num_agents, action_dim]
        last_actions_onehot: torch.Tensor,  # [batch, num_agents, action_dim]
        agent_id: int,
    ) -> torch.Tensor:
        """
        Forward pass for COMA critic.
        
        Args:
            joint_obs: Concatenated observations of all agents [batch, num_agents * obs_dim]
            agent_obs: Individual agent's observation [batch, obs_dim]
            actions_onehot: Current actions (one-hot) [batch, num_agents, action_dim]
            last_actions_onehot: Previous actions (one-hot) [batch, num_agents, action_dim]
            agent_id: Agent index for which to compute Q-values
            
        Returns:
            Q-values for all actions of agent_id: [batch, action_dim]
        """
        batch_size = joint_obs.shape[0]
        
        # Mask out agent's own action (for counterfactual evaluation)
        actions_masked = actions_onehot.clone()
        actions_masked[:, agent_id, :] = 0.0
        
        # Flatten actions
        actions_flat = actions_masked.reshape(batch_size, -1)
        last_actions_flat = last_actions_onehot.reshape(batch_size, -1)
        
        # Agent ID one-hot
        agent_id_onehot = torch.zeros(batch_size, self.num_agents, device=joint_obs.device)
        agent_id_onehot[:, agent_id] = 1.0
        
        # Concatenate inputs (pymarl style)
        inputs = torch.cat([
            joint_obs,
            agent_obs,
            actions_flat,
            last_actions_flat,
            agent_id_onehot
        ], dim=-1)
        
        # Forward through network
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values
    
    def forward_batch(
        self,
        joint_obs: torch.Tensor,           # [batch, seq, num_agents * obs_dim]
        agent_obs: torch.Tensor,            # [batch, seq, obs_dim]
        actions_onehot: torch.Tensor,       # [batch, seq, num_agents, action_dim]
        last_actions_onehot: torch.Tensor,  # [batch, seq, num_agents, action_dim]
        agent_id: int,
    ) -> torch.Tensor:
        """
        Batch forward pass for sequence data (episode-based training).
        
        Args:
            joint_obs: [batch, seq_len, num_agents * obs_dim]
            agent_obs: [batch, seq_len, obs_dim]
            actions_onehot: [batch, seq_len, num_agents, action_dim]
            last_actions_onehot: [batch, seq_len, num_agents, action_dim]
            agent_id: Agent index
            
        Returns:
            Q-values: [batch, seq_len, action_dim]
        """
        batch_size, seq_len = joint_obs.shape[:2]
        
        # Reshape to [batch * seq, ...]
        joint_obs_flat = joint_obs.reshape(batch_size * seq_len, -1)
        agent_obs_flat = agent_obs.reshape(batch_size * seq_len, -1)
        actions_flat = actions_onehot.reshape(batch_size * seq_len, self.num_agents, self.action_dim)
        last_actions_flat = last_actions_onehot.reshape(batch_size * seq_len, self.num_agents, self.action_dim)
        
        # Forward
        q_values = self.forward(
            joint_obs_flat,
            agent_obs_flat,
            actions_flat,
            last_actions_flat,
            agent_id
        )
        
        # Reshape back
        return q_values.reshape(batch_size, seq_len, self.action_dim)
