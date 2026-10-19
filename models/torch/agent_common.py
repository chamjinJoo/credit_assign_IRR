from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.torch.networks import (
    RNNPolicyNetwork,
    RNNValueNetwork,
)


class RecurrentPolicyAgent:
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        device: Optional[torch.device] = None,
        use_recurrent_policy: bool = True,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.device = device
        self.use_recurrent_policy = use_recurrent_policy

        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        if device is not None:
            self.policy_net.to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)

        self.rnn_state_policy = None
        self.prev_action = None
        self.prev_actions_batch = None
        self.n_envs = 1
        self.rnn_states_policy_batch = None

        self.reset_rnn_state(device=device)

    def _default_device(self) -> torch.device:
        return next(self.policy_net.parameters()).device

    def reset_rnn_state(self, device=None):
        if device is None:
            device = self._default_device()
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim, device=device)
        self.prev_action = torch.zeros(self.action_dim, device=device)

    def reset_rnn_states_batch(self, n_envs: int, device=None):
        if device is None:
            device = self._default_device()
        self.n_envs = n_envs
        self.rnn_states_policy_batch = torch.zeros(
            n_envs,
            self.recurrent_N,
            self.hidden_dim,
            device=device,
        )
        self.prev_actions_batch = torch.zeros(n_envs, self.action_dim, device=device)

    def get_action(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor = None,
        available_actions: torch.Tensor = None,
        return_rnn_state: bool = True,
    ) -> Tuple[int, float, Optional[torch.Tensor]]:
        with torch.no_grad():
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs

            if masks is None:
                batch_size = obs_in.size(0)
                masks_in = torch.ones(batch_size, 1, device=self._default_device())
                return_rnn_state = False
            else:
                masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks

            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)

            if available_actions is not None:
                available_actions = (
                    available_actions.unsqueeze(0)
                    if available_actions.dim() == 1
                    else available_actions
                )

            prev_action_onehot = self.prev_action.unsqueeze(0)

            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in,
                curr_rnn_state,
                masks_in,
                available_actions,
                prev_actions=prev_action_onehot,
            )

            self.rnn_state_policy = next_rnn_state.squeeze(0)
            self.prev_action = F.one_hot(action, self.action_dim).float().squeeze(0)

        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone(), next_rnn_state.squeeze(0).clone()
        return action.item(), log_prob.item()

    def get_action_batch(
        self,
        obs_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        available_actions_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prev_states = self.rnn_states_policy_batch
            prev_actions_onehot = self.prev_actions_batch

            actions, log_probs, next_states = self.policy_net.get_action_and_log_prob(
                obs_batch,
                prev_states,
                masks_batch,
                available_actions_batch,
                prev_actions=prev_actions_onehot,
            )

            prev_states_copy = prev_states.clone()
            next_states_copy = next_states.clone()
            self.rnn_states_policy_batch = next_states
            self.prev_actions_batch = F.one_hot(actions, self.action_dim).float()

        return actions, log_probs, prev_states_copy, next_states_copy


class RecurrentActorCriticAgent(RecurrentPolicyAgent):
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: Optional[torch.device] = None,
        use_recurrent_policy: bool = True,
    ):
        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_N,
            lr_actor=lr_actor,
            device=device,
            use_recurrent_policy=use_recurrent_policy,
        )

        self.value_net = (
            RNNValueNetwork(
                obs_dim,
                hidden_dim,
                recurrent_N,
                action_dim=action_dim,
                num_agents=1,
            )
        )
        if device is not None:
            self.value_net.to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)

        self.rnn_state_value = None
        self.rnn_states_value_batch = None
        self.reset_rnn_state(device=device)

    def reset_rnn_state(self, device=None):
        super().reset_rnn_state(device=device)
        if device is None:
            device = self._default_device()
        self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim, device=device)

    def reset_rnn_states_batch(self, n_envs: int, device=None):
        super().reset_rnn_states_batch(n_envs, device=device)
        if device is None:
            device = self._default_device()
        self.rnn_states_value_batch = torch.zeros(
            n_envs,
            self.recurrent_N,
            self.hidden_dim,
            device=device,
        )

    def get_value(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        prev_action_onehot: torch.Tensor = None,
    ) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            if prev_action_onehot is None:
                prev_action_onehot = self.prev_action.unsqueeze(0)
            value, next_rnn_state = self.value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                masks.unsqueeze(0),
                prev_actions=prev_action_onehot,
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()

    def get_value_batch(
        self,
        obs_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        prev_actions_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prev_states = self.rnn_states_value_batch
            if prev_actions_onehot is None:
                prev_actions_onehot = self.prev_actions_batch

            values, next_states = self.value_net(
                obs_batch,
                prev_states,
                masks_batch,
                prev_actions=prev_actions_onehot,
            )

            prev_states_copy = prev_states.clone()
            self.rnn_states_value_batch = next_states

        return values.squeeze(-1), prev_states_copy
