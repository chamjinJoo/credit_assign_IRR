import torch
import torch.nn as nn
from torch.distributions import Categorical

from .mlp import MLPBase
from .policy_network import PolicyNetwork


class FeedForwardPolicyNetworkCompat(nn.Module):
    """Feedforward policy with an RNN-compatible API."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.base = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.output_layer = self.base.policy_head

    def forward(self, obs, rnn_states, masks, available_actions=None, prev_actions=None):
        del masks, prev_actions
        if obs.dim() == 3:
            obs = obs.reshape(-1, obs.size(-1))
        if available_actions is not None and available_actions.dim() == 3:
            available_actions = available_actions.reshape(-1, available_actions.size(-1))
        logits = self.base(obs, available_actions=available_actions)
        return logits, rnn_states

    def get_action_and_log_prob(self, obs, rnn_states, masks, available_actions=None, prev_actions=None):
        logits, rnn_states = self.forward(
            obs,
            rnn_states,
            masks,
            available_actions=available_actions,
            prev_actions=prev_actions,
        )
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, prev_actions=None):
        logits, _ = self.forward(
            obs,
            rnn_states,
            masks,
            available_actions=available_actions,
            prev_actions=prev_actions,
        )
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class FeedForwardValueNetworkCompat(nn.Module):
    """Feedforward critic with an RNN-compatible API."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        self.value_backbone = MLPBase(
            input_dim=obs_dim,
            hidden_size=hidden_dim,
            layer_N=2,
            use_feature_normalization=True,
            use_orthogonal=True,
            use_ReLU=True,
            use_layer_norm=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs, rnn_states, masks, prev_actions=None):
        del masks, prev_actions
        if obs.dim() == 3:
            obs = obs.reshape(-1, obs.size(-1))
        encoded = self.value_backbone(obs)
        values = self.output_layer(encoded)
        return values, rnn_states
