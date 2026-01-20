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

class IPPORNNAgent:
    """
    Single agent for IPPO-RNN algorithm.
    Uses RNNPolicyNetwork.
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.device = device
        
        # RNN Policy network
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.policy_net.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        
        self.rnn_state_policy = None
        self.reset_rnn_state()
        
    def reset_rnn_state(self):
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)

    def get_action(self, obs: torch.Tensor, masks: torch.Tensor = None, available_actions: torch.Tensor = None, return_rnn_state=True) -> Tuple[int, float, Optional[torch.Tensor]]:
        """
        Get action from policy with RNN state handling.
        """
        with torch.no_grad():
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
            
            if masks is None:
                # Called from trainer (evaluation)
                batch_size = obs_in.size(0)
                masks_in = torch.ones(batch_size, 1, device=self.device)
                return_rnn_state = False
            else:
                masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks
            
            # RNNLayer expects (Batch, Layers, Dim) -> (1, Layers, Dim)
            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)
            
            # Handle available actions
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0) if available_actions.dim() == 1 else available_actions

            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in,
                available_actions
            )
            
            # Update internal state (Layers, Dim)
            self.rnn_state_policy = next_rnn_state.squeeze(0)
            
        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone()
        else:
            return action.item(), log_prob.item()


class IPPORNNAlgorithm:
    """
    IPPO Algorithm with RNN support.
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
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
        global_reward: bool = False,
    ):
        self.global_reward = global_reward
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
    
        self.training_steps = 0
        
        self.buffer_manager = TransitionBuffer(max_buffer_size=10000)
        
        # Agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = IPPORNNAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                device=self.device
            )

        # Shared RNN Value network
        self.value_net = RNNValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_N
        )
        self.value_net.to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Value RNN states for each agent (since they are independent instances sharing parameters)
        self.rnn_states_value = {}
        self.reset_agents()
        
        # Temp storage
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.last_dones = {i: False for i in range(self.num_agents)}

    def reset_agents(self):
        """Reset agent policy hidden states and value hidden states."""
        for agent in self.agents.values():
            agent.reset_rnn_state()
        
        self.rnn_states_value = {
            i: torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device) 
            for i in range(self.num_agents)
        }
        self.last_dones = {i: False for i in range(self.num_agents)}

    def _get_value(self, agent_id: int, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Helper to get value and update hidden state for an agent."""
        with torch.no_grad():
            curr_rnn_state = self.rnn_states_value[agent_id].unsqueeze(0) # (1, L, D)
            
            value, next_rnn_state = self.value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                mask.unsqueeze(0)
            )
            
            self.rnn_states_value[agent_id] = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()

    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            # Prepare available actions tensor
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            # Policy
            action, log_prob, p_state = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)
            
            # Value
            _, v_state = self._get_value(agent_id, obs, mask_tensor)
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            
        return actions
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None # [ADDED]
    ):
        self.last_dones = dones.copy()
        
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks=self.temp_masks,
            hidden_states=self.temp_hidden_states,
            info=info,
            available_actions=available_actions # [ADDED]
        )

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
        
        # NOTE: IPPO trains independent policies but shares value parameters here (or independent values?)
        # Original IPPO in ippo.py has: self.value_net = ValueNetwork(obs_dim, ...). Shared Net.
        # But fed with local obs. So it is parameter sharing.
        
        full_obs_batch = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data]).to(self.device) # Not used?
        actions_batch = torch.stack([self._concatenate_actions(t['bold_a_t']) for t in batch_data]).to(self.device)
        
        # Extract available actions
        available_actions_batch = {}
        if 'available_actions' in batch_data[0] and batch_data[0]['available_actions'] is not None:
            for agent_id in range(self.num_agents):
                if agent_id in batch_data[0]['available_actions']:
                     # t['available_actions'][agent_id] is List[int]
                     avail_list = [t['available_actions'][agent_id] for t in batch_data]
                     # Convert to tensor (Batch, ActionDim)
                     available_actions_batch[agent_id] = torch.tensor(avail_list, dtype=torch.float32, device=self.device)
        
        # Prepare data per agent
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            # Extract Batch Data
            # Extract Batch Data
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            
            if self.global_reward:
                 # Global reward (mean of all agents)
                 rewards_list = [sum(t['bold_r_t'].values()) / self.num_agents for t in batch_data]
                 agent_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            else:
                 # Individual reward
                 agent_rewards = torch.tensor([t['bold_r_t'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
                 
            agent_dones = torch.tensor([t['dones'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            masks_batch = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
            
            # Hidden states: (p_state, v_state)
            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
            v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
            
            # --- 1. Compute Values & Advantages ---
            # Re-run value network to get current values (gradient-free for rollout values)
            # Optimization: We can do it chunked here too if we want, but simple forward is fast.
            # But wait, we need next value for GAE.
            # For RNN, we need to run sequence.
            
            # Let's use the optimized chunk approach for GAE computation? 
            # No, GAE needs full sequential recurrence. We can't chunk GAE calculation itself easily unless we stitch.
            # So for GAE, we run full sequence forward (Batch=1, Seq=T).
            
            p_h0, _ = batch_data[0]['hidden_states'][agent_id]
            _, v_h0 = batch_data[0]['hidden_states'][agent_id]
            v_h0 = v_h0.to(self.device).unsqueeze(0) # (1, L, D)
            
            with torch.no_grad():
                # Prepare extended sequence for next value
                last_next_obs = batch_data[-1]['next_observations'][agent_id].to(self.device)
                last_mask = 1.0 - batch_data[-1]['dones'][agent_id]
                
                rnn_obs = agent_obs.unsqueeze(1) # (T, 1, D)
                # Masks for value forward
                rnn_masks = masks_batch # (T, 1)
                
                extended_obs = torch.cat([rnn_obs, last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                extended_masks = torch.cat([rnn_masks, torch.tensor([[last_mask]], device=self.device)], dim=0)
                
                all_values, _ = self.value_net(extended_obs, v_h0, extended_masks)
                agent_values = all_values[:-1].reshape(-1)
                next_val = all_values[-1].item() * last_mask
                
            advantages, returns = self.compute_gae(
                agent_rewards.tolist(),
                agent_values.tolist(),
                next_val,
                agent_dones.tolist()
            )
            
            agent_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            agent_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            
            if agent_advantages.numel() > 1:
                adv = (agent_advantages - agent_advantages.mean()) / (agent_advantages.std() + 1e-8)
            else:
                adv = torch.zeros_like(agent_advantages)

            # --- 2. Chunked PPO Update ---
            my_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
            
            T_total = agent_obs.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            
            if num_chunks > 0:
                # Chunk data
                rnn_obs_batch = self._chunk_sequence(agent_obs, chunk_len) # (L, Chunks, D)
                my_actions_batch = self._chunk_sequence(my_actions, chunk_len).reshape(-1) # (L*Chunks)
                rnn_masks_batch = self._chunk_sequence(masks_batch, chunk_len)
                ret_batch = self._chunk_sequence(agent_returns, chunk_len).reshape(-1)
                adv_batch = self._chunk_sequence(adv, chunk_len).reshape(-1)
                
                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                p_h0_batch = p_states_all[chunk_indices] # (Chunks, L, D) expecting (B, L, D) -> Correct
                v_h0_batch = v_states_all[chunk_indices]
                
                avail_actions_batch_merged = None
                if agent_id in available_actions_batch:
                     avail_actions_chunked = self._chunk_sequence(available_actions_batch[agent_id], chunk_len)
                     avail_actions_batch_merged = avail_actions_chunked.reshape(-1, avail_actions_chunked.size(-1))

            else:
                 rnn_obs_batch = agent_obs.unsqueeze(1)
                 rnn_masks_batch = masks_batch
                 my_actions_batch = my_actions
                 ret_batch = agent_returns
                 adv_batch = adv
                 p_h0_batch = p_states_all[0].unsqueeze(0)
                 v_h0_batch = v_states_all[0].unsqueeze(0)
                 
                 avail_actions_batch_merged = None
                 if agent_id in available_actions_batch:
                    avail_actions_batch_merged = available_actions_batch[agent_id]

            with torch.no_grad():
                old_log_probs, _ = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, available_actions=avail_actions_batch_merged)
                old_log_probs = old_log_probs.detach()
            
            for _ in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, available_actions=avail_actions_batch_merged)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
                
                # Update Value Network
                new_vals, _ = self.value_net(rnn_obs_batch, v_h0_batch, rnn_masks_batch)
                val_loss = F.mse_loss(new_vals.squeeze(1), ret_batch)
                
                self.value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                agent_losses[f'agent_{agent_id}_value_loss'] += val_loss.item()

        return dict(agent_losses)

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)

    def _concatenate_actions(self, actions: Dict[int, int]) -> torch.Tensor:
         # Just helper if needed, though we process per agent
         return torch.tensor([actions[i] for i in range(self.num_agents) if i in actions])

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
            for k, v in losses.items():
                total_losses[k] += v
                
        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        
        num = len([b for b in batches if b])
        if num == 0: return {}
        return {k: v / num for k, v in total_losses.items()}

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()

    def get_buffer_info(self):
        return self.buffer_manager.get_buffer_info()
