import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from algorithms.networks import RNNPolicyNetwork, RNNValueNetwork
from utils.buffer import TransitionBuffer

class MAPPORNNAgent:
    """
    Single agent for MAPPO-RNN.
    """
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.device = device
        
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.policy_net.to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.rnn_state_policy = None
        self.reset_rnn_state()
        
    def reset_rnn_state(self):
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)

    def get_action(self, obs: torch.Tensor, masks: torch.Tensor = None, return_rnn_state=True) -> Tuple[int, float, Optional[torch.Tensor]]:
        with torch.no_grad():
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs 
            
            if masks is None:
                 # Called from trainer (evaluation)
                 batch_size = obs_in.size(0)
                 masks_in = torch.ones(batch_size, 1, device=self.device)
                 return_rnn_state = False
            else:
                 masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks
            
            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)
            
            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in
            )
            
            self.rnn_state_policy = next_rnn_state.squeeze(0)
            
        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone()
        else:
            return action.item(), log_prob.item()


class MAPPORNNAlgorithm:
    """
    MAPPO Algorithm with RNN support (Centralized RNN Critic).
    """
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        num_batches: int = 1,
        device: str = "cpu",
        rnn_chunk_length: int = 16,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.rnn_chunk_length = rnn_chunk_length
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        self.buffer_manager = TransitionBuffer(max_buffer_size=10000)
        self.training_steps = 0
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = MAPPORNNAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr=lr,
                device=self.device
            )
            
        # Central RNN Critic
        self.value_net = RNNValueNetwork(
            obs_dim=obs_dim * num_agents,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_N
        )
        self.value_net.to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        self.rnn_state_value = None # Central Value State
        self.reset_agents()
        
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.last_dones = {i: False for i in range(self.num_agents)} # Only used for getting masks
        
    def reset_agents(self):
        for agent in self.agents.values():
            agent.reset_rnn_state()
        self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        self.last_dones = {i: False for i in range(self.num_agents)}

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)

    def _update_central_value(self, global_obs: torch.Tensor, mask: torch.Tensor) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            
            value, next_rnn_state = self.value_net(
                global_obs.unsqueeze(0),
                curr_rnn_state,
                mask.unsqueeze(0)
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()

    def get_actions(self, observations: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        
        # We assume all agents have same Done status for masking Central Value?
        # Typically central value mask = min(all agent masks) or just one of them if synchronous.
        # We use agent 0's mask for central value.
        mask0 = 1.0 if not self.last_dones.get(0, False) else 0.0
        mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
        
        # Central Value Step
        global_obs = self._concatenate_observations(observations)
        _, v_state = self._update_central_value(global_obs, mask_tensor0)
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            action, log_prob, p_state = self.agents[agent_id].get_action(obs, mask_tensor)
            
            actions[agent_id] = (action, log_prob)
            # Store (policy_state, central_value_state) for each agent
            # We enforce v_state on all agents so buffer structure is consistent
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            
        return actions

    def store_transition(self, **kwargs):
        if 'dones' in kwargs:
            self.last_dones = kwargs['dones'].copy()
        
        kwargs['masks'] = self.temp_masks
        kwargs['hidden_states'] = self.temp_hidden_states
        self.buffer_manager.store_transition(**kwargs)

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return advantages, returns

    def _chunk_sequence(self, tensor: torch.Tensor, chunk_len: int) -> torch.Tensor:
        T = tensor.size(0)
        num_chunks = T // chunk_len
        if num_chunks == 0:
            return tensor.unsqueeze(1)
            
        tensor = tensor[:num_chunks * chunk_len]
        shape = (num_chunks, chunk_len) + tensor.shape[1:]
        tensor = tensor.view(shape)
        
        permute_dims = (1, 0) + tuple(range(2, tensor.dim()))
        return tensor.permute(permute_dims).contiguous()

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)
        
        # === Prepare Global Data ===
        global_obs_seq = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data]).to(self.device)
        global_masks_seq = torch.tensor([t['masks'][0] for t in batch_data], device=self.device).unsqueeze(1) # Use agent 0 mask
        
        # Central Value Hidden States
        # Stored in hidden_states[0][1]
        v_states_all = torch.stack([t['hidden_states'][0][1] for t in batch_data]).to(self.device)
        
        # Calculate Global Rewards
        global_rewards = []
        global_dones = []
        for t in batch_data:
            global_reward = sum(t['bold_r_t'].values()) / self.num_agents
            global_rewards.append(global_reward)
            global_dones.append(any(t['dones'].values()))
        
        all_global_rewards = torch.tensor(global_rewards, dtype=torch.float32, device=self.device)
        all_global_dones = torch.tensor(global_dones, dtype=torch.float32, device=self.device)

        # === Compute Global Values/GAE (Sequential) ===
        _, v_h0 = batch_data[0]['hidden_states'][0]
        v_h0 = v_h0.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            last = batch_data[-1]
            last_next_obs = self._concatenate_observations(last['next_observations']).to(self.device)
            last_mask = 1.0 - (1.0 if any(last['dones'].values()) else 0.0)
            
            rnn_obs = global_obs_seq.unsqueeze(1)
            rnn_masks = global_masks_seq
            
            extended_obs = torch.cat([rnn_obs, last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
            extended_masks = torch.cat([rnn_masks, torch.tensor([[last_mask]], device=self.device)], dim=0)
            
            all_values, _ = self.value_net(extended_obs, v_h0, extended_masks)
            global_values = all_values[:-1].reshape(-1)
            next_global_val = all_values[-1].item() * last_mask

        advantages_global, returns_global = self.compute_gae(
            all_global_rewards.tolist(),
            global_values.tolist(),
            next_global_val,
            all_global_dones.tolist()
        )
        global_returns = torch.tensor(returns_global, dtype=torch.float32, device=self.device)
        global_advantages = torch.tensor(advantages_global, dtype=torch.float32, device=self.device)
        
        # Normalize Advantages
        if global_advantages.numel() > 1:
            global_adv_norm = (global_advantages - global_advantages.mean()) / (global_advantages.std() + 1e-8)
        else:
            global_adv_norm = torch.zeros_like(global_advantages)

        # === Prepare Chunked Data for Value Update ===
        T_total = global_obs_seq.size(0)
        chunk_len = self.rnn_chunk_length
        num_chunks = T_total // chunk_len
        
        if num_chunks > 0:
             global_obs_batch = self._chunk_sequence(global_obs_seq, chunk_len)
             global_masks_batch = self._chunk_sequence(global_masks_seq, chunk_len)
             global_ret_batch = self._chunk_sequence(global_returns, chunk_len).reshape(-1)
             global_adv_batch = self._chunk_sequence(global_adv_norm, chunk_len).reshape(-1)
             
             chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
             v_h0_batch = v_states_all[chunk_indices] 
        else:
             global_obs_batch = global_obs_seq.unsqueeze(1)
             global_masks_batch = global_masks_seq
             global_ret_batch = global_returns
             global_adv_batch = global_adv_norm
             v_h0_batch = v_states_all[0].unsqueeze(0)

        # === Policy Updates (Per Agent) ===
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            masks_batch = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
            my_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
            
            if num_chunks > 0:
                 rnn_obs_batch = self._chunk_sequence(agent_obs, chunk_len)
                 rnn_masks_batch = self._chunk_sequence(masks_batch, chunk_len)
                 my_actions_batch = self._chunk_sequence(my_actions, chunk_len).reshape(-1)
                 p_h0_batch = p_states_all[chunk_indices]
            else:
                 rnn_obs_batch = agent_obs.unsqueeze(1)
                 rnn_masks_batch = masks_batch
                 my_actions_batch = my_actions
                 p_h0_batch = p_states_all[0].unsqueeze(0)

            with torch.no_grad():
                 old_log_probs, _ = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch)
                 old_log_probs = old_log_probs.detach()
            
            for _ in range(self.ppo_epochs):
                 new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch)
                 ratio = torch.exp(new_log_probs - old_log_probs)
                 
                 surr1 = ratio * global_adv_batch 
                 surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * global_adv_batch
                 policy_loss = -torch.min(surr1, surr2).mean()
                 entropy_loss = -entropy.mean()
                 
                 loss = policy_loss + self.entropy_coef * entropy_loss
                 
                 self.agents[agent_id].optimizer.zero_grad()
                 loss.backward()
                 torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                 self.agents[agent_id].optimizer.step()
                 
                 agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                 agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()

        # === Central Value Update ===
        for _ in range(self.ppo_epochs):
             new_vals, _ = self.value_net(global_obs_batch, v_h0_batch, global_masks_batch)
             val_loss = F.mse_loss(new_vals.squeeze(1), global_ret_batch)
             
             self.value_optimizer.zero_grad()
             (self.value_coef * val_loss).backward()
             torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
             self.value_optimizer.step()
             
             agent_losses['value_loss'] += val_loss.item()

        # Normalize value loss logging
        agent_losses['value_loss'] /= self.ppo_epochs
        
        return dict(agent_losses)

    def update(self, **kwargs):
        if self.buffer_manager.is_main_buffer_empty():
            return {}
        transitions = self.buffer_manager.get_main_buffer_transitions()
        batch_size = max(1, len(transitions) // self.num_batches)
        batches = [transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)]
        total_losses = defaultdict(float)
        for batch_data in batches:
            if not batch_data: continue
            losses = self.update_agents(batch_data)
            for k, v in losses.items(): total_losses[k] += v
        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        num = len([b for b in batches if b])
        if num == 0: return {}
        return {k: v / num for k, v in total_losses.items()}

    def clear_buffer(self):
         self.buffer_manager.clear_buffer()
    def get_buffer_info(self):
         return self.buffer_manager.get_buffer_info()
