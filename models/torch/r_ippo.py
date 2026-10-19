import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from models.torch.networks import RNNValueNetwork

from models.torch.agent_common import RecurrentPolicyAgent
from models.torch.common import (
    build_agent_active_masks,
    build_available_actions_batch,
    build_team_done_sequence,
    build_transition_kwargs,
    chunk_sequence,
    compute_gae,
    normalize_advantages,
    prepare_env_info,
    resolve_step_episode_done,
    split_action_and_log_probs,
)
from utils.buffer import TransitionBuffer, EpisodeBuffer

class rIPPOAgent(RecurrentPolicyAgent):
    """
    Single agent for IPPO-RNN algorithm.
    Uses RNNPolicyNetwork.
    """
    pass


class rIPPOAlgorithm:
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
        use_episode_buffer: bool = False,
        use_recurrent_policy: bool = True,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.rnn_chunk_length = rnn_chunk_length
        self.use_recurrent_policy = use_recurrent_policy
        
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
        self.use_episode_buffer = use_episode_buffer
        
        if use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=100)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=10000)
        
        # Agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = rIPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                device=self.device,
                use_recurrent_policy=use_recurrent_policy,
            )

        # Shared RNN Value network
        self.value_net = RNNValueNetwork(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                out_dim=1,
                action_dim=action_dim,
                num_agents=1, # for prev action dim
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

    def reset_agents_batch(self, n_envs: int):
        """Reset agents for multiple environments (batch mode)"""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs)
        
        # Value RNN States Batch (per agent)
        self.rnn_states_value_batch = {
            i: torch.zeros(n_envs, self.recurrent_N, self.hidden_dim).to(self.device)
            for i in range(self.num_agents)
        }

        # Track done status for each agent in each environment
        self.last_dones_batch = [{i: False for i in range(self.num_agents)} for _ in range(n_envs)]

    def _get_value(self, agent_id: int, obs: torch.Tensor, mask: torch.Tensor, prev_action_onehot: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """Helper to get value and update hidden state for an agent."""
        with torch.no_grad():
            curr_rnn_state = self.rnn_states_value[agent_id].unsqueeze(0) # (1, L, D)
            
            if prev_action_onehot is None:
                prev_action_onehot = self.agents[agent_id].prev_action.unsqueeze(0)  # (1, action_dim)
            
            value, next_rnn_state = self.value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                mask.unsqueeze(0),
                prev_actions=prev_action_onehot
            )
            
            self.rnn_states_value[agent_id] = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()
        
    def _get_value_batch(self, agent_id: int, obs_batch: torch.Tensor, masks_batch: torch.Tensor, prev_actions_onehot: Optional[torch.Tensor] = None):
        """Helper to get batch values for an agent."""
        with torch.no_grad():
            # Get current value states for this agent across batch
            curr_rnn_state = self.rnn_states_value_batch[agent_id].clone()
            
            if prev_actions_onehot is None:
                prev_actions_onehot = self.agents[agent_id].prev_actions_batch  # (n_envs, action_dim)
            
            values, next_rnn_state = self.value_net(
                obs_batch,
                curr_rnn_state,
                masks_batch,
                prev_actions=prev_actions_onehot
            )
            
            # Update state
            self.rnn_states_value_batch[agent_id] = next_rnn_state
            
        return values.squeeze(1).squeeze(-1), curr_rnn_state # Return values as (Batch,), prev_state

    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
        available_actions_batch: Optional[List[Dict[int, List[int]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        """
        Get actions for a batch of environments (ParallelEnv execution).
        """
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]
        
        has_avail = (available_actions_batch is not None)
            
        for agent_id in range(self.num_agents):
            # Collect observations/masks/avail for this agent across envs
            agent_obs_list = []
            agent_masks_list = []
            agent_avail_list = []
            
            for i in range(n_envs):
                obs = observations_batch[i][agent_id]
                agent_obs_list.append(obs)
                
                # Mask
                mask = 1.0 if not self.last_dones_batch[i].get(agent_id, False) else 0.0
                agent_masks_list.append(mask)
                self.temp_masks_batch[i][agent_id] = mask
                
                # Available actions
                if has_avail and available_actions_batch[i] is not None and agent_id in available_actions_batch[i]:
                    agent_avail_list.append(available_actions_batch[i][agent_id])
                else:
                    agent_avail_list.append(None)
            
            agent_obs_batch = torch.stack(agent_obs_list).to(self.device)
            agent_masks_batch = torch.tensor(agent_masks_list, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                 agent_avail_tensor = torch.tensor(agent_avail_list, dtype=torch.float32, device=self.device)

            # Cache a_{t-1} before get_action_batch updates it to a_t
            agent_prev_actions_onehot = self.agents[agent_id].prev_actions_batch.clone()

            # 1. Policy Step
            actions, log_probs, prev_p_states, _ = self.agents[agent_id].get_action_batch(
                agent_obs_batch,
                agent_masks_batch,
                available_actions_batch=agent_avail_tensor
            )
            
            # 2. Value Step (Independent Value)
            # prev_v_states: (Batch, L, D)
            _, prev_v_states = self._get_value_batch(
                agent_id, 
                agent_obs_batch, 
                agent_masks_batch, 
                prev_actions_onehot=agent_prev_actions_onehot
            )
            
            # Distribute results
            for i in range(n_envs):
                if isinstance(actions[i], torch.Tensor):
                    act = actions[i].item()
                    lp = log_probs[i].item()
                else:
                    act = actions[i]
                    lp = log_probs[i]
                
                actions_batch[i][agent_id] = (act, lp)
                
                # Store Hidden States: (policy, value)
                self.temp_hidden_states_batch[i][agent_id] = (
                    prev_p_states[i],
                    prev_v_states[i]
                )
                
        return actions_batch

    def store_transition_from_env(
        self,
        env_idx: int,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, Tuple[int, float]],  # (action, log_prob)
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None
    ):
        """Store transition from a specific parallel environment."""
        actions_only, log_probs_dict = split_action_and_log_probs(actions)
        info = prepare_env_info(info, env_idx)
        episode_done = resolve_step_episode_done(dones, info)

        kwargs = build_transition_kwargs(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions_only,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks=self.temp_masks_batch[env_idx],
            hidden_states=self.temp_hidden_states_batch[env_idx],
            available_actions=available_actions,
            log_probs=log_probs_dict,
            info=info,
            env_id=env_idx if self.use_episode_buffer else None,
        )
        self.buffer_manager.store_transition(**kwargs)
        
        if self.use_episode_buffer:
            if episode_done:
                self.buffer_manager.finish_episode(env_id=env_idx)
        
        # Update done status and reset RNN if done
        for agent_id in range(self.num_agents):
            self.last_dones_batch[env_idx][agent_id] = dones[agent_id]
            if dones[agent_id]:
                # Reset Policy RNN
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                   self.recurrent_N, self.hidden_dim, device=self.device
                )
                # Reset prev_action for the done agent
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                   self.action_dim, device=self.device
                )
                # Reset Value RNN
                self.rnn_states_value_batch[agent_id][env_idx] = torch.zeros(
                   self.recurrent_N, self.hidden_dim, device=self.device
                )

    def finish_episode(self, env_id: int = 0):
        """Finish current episode if using EpisodeBuffer."""
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)

    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.temp_log_probs = {}
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            # --- Fix: Cache a_{t-1} before get_action updates it to a_t ---
            agent_prev_action_onehot = self.agents[agent_id].prev_action.unsqueeze(0).clone()

            # Policy
            action, log_prob, p_state, _ = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)
            
            # Value
            _, v_state = self._get_value(
                agent_id, 
                obs, 
                mask_tensor, 
                prev_action_onehot=agent_prev_action_onehot
            )
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            self.temp_log_probs[agent_id] = log_prob
            
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

        kwargs = build_transition_kwargs(
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
            available_actions=available_actions,
            log_probs=self.temp_log_probs,
            info=info,
        )
        self.buffer_manager.store_transition(**kwargs)

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)
        
        # NOTE: IPPO trains independent policies but shares value parameters here (or independent values?)
        # Original IPPO in ippo.py has: self.value_net = ValueNetwork(obs_dim, ...). Shared Net.
        # But fed with local obs. So it is parameter sharing.
        
        available_actions_batch = build_available_actions_batch(
            batch_data,
            self.num_agents,
            self.device,
        )
        
        # === PPO Epoch Updates ===
        for epoch in range(self.ppo_epochs):
            total_value_loss = 0.0
            total_active_masks_sum = 0.0
            
            # 1. Update each agent's policy independently and accumulate value loss
            for agent_id in range(self.num_agents):
                if agent_id not in batch_data[0]['bold_o_t']: continue
                
                # Fetch pre-prepared batch data for this agent
                # (For simplicity and exact reproduction of previous logic, we re-chunk here or 
                # use pre-chunked references if available. The previous code chunked inside the loop.)
                
                agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
                masks_batch = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
                p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
                v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
                my_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
                
                # Extract pre-computed Returns and Advantages
                agent_returns = torch.tensor([t['returns'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
                agent_advantages = torch.tensor([t['advantages'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
                
                adv = normalize_advantages(agent_advantages)

                T_total = agent_obs.size(0)
                chunk_len = self.rnn_chunk_length
                num_chunks = T_total // chunk_len
                agent_active_masks = build_agent_active_masks(batch_data, agent_id, self.device)
                
                if num_chunks > 0:
                    rnn_obs_batch = chunk_sequence(agent_obs, chunk_len)
                    my_actions_batch = chunk_sequence(my_actions, chunk_len).reshape(-1)
                    rnn_masks_batch = chunk_sequence(masks_batch, chunk_len)
                    ret_batch = chunk_sequence(agent_returns, chunk_len).reshape(-1)
                    adv_batch = chunk_sequence(adv, chunk_len).reshape(-1)
                    
                    active_masks_batch = chunk_sequence(agent_active_masks, chunk_len).reshape(-1)
                    
                    chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                    p_h0_batch = p_states_all[chunk_indices] 
                    v_h0_batch = v_states_all[chunk_indices]
                    
                    prev_actions_raw = torch.tensor(
                        [t['bold_a_t_minus_1'][agent_id] for t in batch_data[:num_chunks * chunk_len]],
                        dtype=torch.long, device=self.device
                    )
                    prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
                    prev_actions_chunked = chunk_sequence(prev_actions_onehot_seq, chunk_len)
                    
                    avail_actions_batch_merged = None
                    if agent_id in available_actions_batch:
                         avail_actions_chunked = chunk_sequence(available_actions_batch[agent_id], chunk_len)
                         avail_actions_batch_merged = avail_actions_chunked.reshape(-1, avail_actions_chunked.size(-1))
                else:
                     rnn_obs_batch = agent_obs.unsqueeze(1)
                     rnn_masks_batch = masks_batch
                     my_actions_batch = my_actions
                     ret_batch = agent_returns
                     adv_batch = adv
                     p_h0_batch = p_states_all[0].unsqueeze(0)
                     v_h0_batch = v_states_all[0].unsqueeze(0)
                     active_masks_batch = agent_active_masks
                     
                     prev_actions_raw = torch.tensor(
                         [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
                         dtype=torch.long, device=self.device
                     )
                     prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
                     prev_actions_chunked = prev_actions_onehot_seq.unsqueeze(1)
                     
                     avail_actions_batch_merged = None
                     if agent_id in available_actions_batch:
                        avail_actions_batch_merged = available_actions_batch[agent_id]

                old_log_probs_raw = torch.tensor([t['log_probs'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
                if num_chunks > 0:
                    old_log_probs = chunk_sequence(old_log_probs_raw, chunk_len).reshape(-1)
                else:
                    old_log_probs = old_log_probs_raw
                
                # --- Policy Evaluation ---
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(
                    rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch,
                    available_actions=avail_actions_batch_merged, prev_actions=prev_actions_chunked
                )
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
                entropy_loss = -entropy 
                entropy_loss = (entropy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
                
                # --- Shared Value Net Evaluation ---
                new_vals, _ = self.value_net(rnn_obs_batch, v_h0_batch, rnn_masks_batch, prev_actions=prev_actions_chunked)
                val_error = (new_vals.squeeze(1) - ret_batch) ** 2
                val_loss = (val_error * active_masks_batch).sum()
                
                total_value_loss += val_loss
                total_active_masks_sum += active_masks_batch.sum()
                
            # 2. Single Global Value Update per Epoch
            if total_active_masks_sum > 0:
                mean_val_loss = total_value_loss / total_active_masks_sum
                self.value_optimizer.zero_grad()
                (self.value_coef * mean_val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # We can store the shared value loss 
                agent_losses['value_loss'] += mean_val_loss.item()
                
        # 3. Normalize logs by ppo_epochs at the end
        agent_losses['value_loss'] /= self.ppo_epochs
        for agent_id in range(self.num_agents):
            if f'agent_{agent_id}_policy_loss' in agent_losses:
                agent_losses[f'agent_{agent_id}_policy_loss'] /= self.ppo_epochs
                agent_losses[f'agent_{agent_id}_entropy_loss'] /= self.ppo_epochs

        return dict(agent_losses)

    def update(self, **kwargs):
        # Get trajectories based on buffer type
        if self.use_episode_buffer:
            if not self.buffer_manager.has_completed_episodes():
                return {}
            episode_trajectories = self.buffer_manager.get_completed_episodes()
            # episode_trajectories is already grouped by episodes
            env_trajectories = {i: traj for i, traj in enumerate(episode_trajectories)}
        else:
            if self.buffer_manager.is_main_buffer_empty():
                return {}
            transitions = self.buffer_manager.get_main_buffer_transitions()
            
            # Group by env_id for TransitionBuffer
            env_trajectories = defaultdict(list)
            for t in transitions:
                env_id = t['info'].get('env_id', 0)
                env_trajectories[env_id].append(t)
        
        # 2. Compute GAE for each trajectory
        for env_id, traj in env_trajectories.items():
            if len(traj) == 0:
                continue
            team_dones_seq = build_team_done_sequence(traj)
            
            # For each agent, compute GAE
            for agent_id in range(self.num_agents):
                if agent_id not in traj[0]['bold_o_t']:
                    continue

                # Optimized: Stack on CPU, then single GPU transfer
                agent_obs_list = [t['bold_o_t'][agent_id] for t in traj]
                agent_obs_seq = torch.stack(agent_obs_list).to(self.device, non_blocking=True)
                agent_masks_seq = torch.tensor([t['masks'][agent_id] for t in traj], device=self.device).unsqueeze(1)

                agent_rewards_seq = torch.tensor(
                    [sum(t['bold_r_t'].values()) / self.num_agents for t in traj],
                    dtype=torch.float32,
                    device=self.device,
                )

                # Initial hidden state for value network
                _, v_h0 = traj[0]['hidden_states'][agent_id]
                v_h0 = v_h0.to(self.device, non_blocking=True).unsqueeze(0)

                # Compute values for entire trajectory
                with torch.no_grad():
                    last_next_obs = traj[-1]['next_observations'][agent_id].to(self.device, non_blocking=True)
                    last_mask = 1.0 - (1.0 if team_dones_seq[-1] else 0.0)
                    done_signal = team_dones_seq
                    
                    extended_obs = torch.cat([agent_obs_seq.unsqueeze(1),
                                            last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                    extended_masks = torch.cat([agent_masks_seq,
                                                torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)], dim=0)
                    
                    # Extract prev_actions for sequence
                    agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in traj], dtype=torch.long, device=self.device)
                    agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                    # For the extra last step padding, simply repeat the last action
                    last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                    extended_prev_actions = torch.cat([agent_prev_actions_onehot, last_prev_action_onehot], dim=0).unsqueeze(1)
                    
                    all_values, _ = self.value_net(extended_obs, v_h0, extended_masks, prev_actions=extended_prev_actions)
                    values = all_values[:-1].view(-1)
                    next_val = all_values[-1].item() * last_mask
                    
                    # Compute GAE
                    advantages, returns = compute_gae(
                        agent_rewards_seq.tolist(),
                        values.tolist(),
                        next_val,
                        done_signal, self.gamma, self.gae_lambda,
                    )
                    
                    # Store in transitions
                    for t_idx, t in enumerate(traj):
                        if 'advantages' not in t:
                            t['advantages'] = {}
                            t['returns'] = {}
                        t['advantages'][agent_id] = advantages[t_idx]
                        t['returns'][agent_id] = returns[t_idx]
        
        # 3. Create chunks from trajectories with Padding
        all_chunks = []
        chunk_len = self.rnn_chunk_length
        
        for env_id, traj in env_trajectories.items():
            if len(traj) == 0: continue
            
            # Add active_mask to existing transitions
            for t in traj:
                t['active_mask'] = 1.0
            
            T_traj = len(traj)
            num_full_chunks = T_traj // chunk_len
            remainder = T_traj % chunk_len
            
            # Full chunks
            for i in range(num_full_chunks):
                start = i * chunk_len
                end = start + chunk_len
                chunk = traj[start:end]
                all_chunks.append(chunk)
            
            # Handling remainder with padding
            if remainder > 0:
                last_chunk = traj[num_full_chunks * chunk_len:]
                
                # Create dummy padding transitions
                padding_len = chunk_len - remainder
                last_valid_trans = last_chunk[-1]
                
                padding_chunk = []
                for _ in range(padding_len):
                    pad_t = last_valid_trans.copy() 
                    pad_t['active_mask'] = 0.0
                    padding_chunk.append(pad_t)
                
                full_last_chunk = last_chunk + padding_chunk
                all_chunks.append(full_last_chunk)

        # # 4. Shuffle chunks for IID
        # random.shuffle(all_chunks)
        
        # 5. Create batches
        n_chunks_per_batch = max(1, len(all_chunks) // self.num_batches)
        chunk_batches = [all_chunks[i:i + n_chunks_per_batch] 
                        for i in range(0, len(all_chunks), n_chunks_per_batch)]
        
        # 6. Flatten chunks back to transitions
        batches = []
        for b in chunk_batches:
            flat_batch = []
            for chunk in b:
                flat_batch.extend(chunk)
            if flat_batch:
                batches.append(flat_batch)
                    
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
