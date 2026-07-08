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
    concatenate_agent_observations,
    prepare_env_info,
    normalize_advantages,
    resolve_step_episode_done,
    split_action_and_log_probs,
)
from utils.buffer import TransitionBuffer, EpisodeBuffer

class rMAPPOAgent(RecurrentPolicyAgent):
    """
    Single agent for MAPPO-RNN.
    """
    pass


class rMAPPOAlgorithm:
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
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
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
        self.num_batches = num_batches
        self.use_episode_buffer = use_episode_buffer
        
        if use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=100)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=10000)
        self.training_steps = 0
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = rMAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                device=self.device,
                use_recurrent_policy=use_recurrent_policy,
            )
            
        # Central RNN Critic
        self.value_net = RNNValueNetwork(
                obs_dim=obs_dim * num_agents,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                out_dim=1,
                action_dim=action_dim,
                num_agents=num_agents
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

    def reset_agents_batch(self, n_envs: int):
        """Reset agents for multiple environments (batch mode)"""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs)
        
        # Central Value RNN State Batch
        self.rnn_states_value_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim).to(self.device)

        # Track done status for each agent in each environment
        self.last_dones_batch = [{i: False for i in range(self.num_agents)} for _ in range(n_envs)]

    def _update_central_value(self, global_obs: torch.Tensor, mask: torch.Tensor, global_prev_action_onehot: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            
            if global_prev_action_onehot is None:
                # Construct global prev action from all agents
                agent_prev_actions = [self.agents[i].prev_action for i in range(self.num_agents)]
                global_prev_action_onehot = torch.cat(agent_prev_actions, dim=-1).unsqueeze(0) # (1, num_agents * action_dim)
            
            value, next_rnn_state = self.value_net(
                global_obs.unsqueeze(0),
                curr_rnn_state,
                mask.unsqueeze(0),
                prev_actions=global_prev_action_onehot
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()
        
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
        
        # 1. Global Observations for Central Critic
        global_obs_list = []
        for i in range(n_envs):
            global_obs_list.append(
                concatenate_agent_observations(
                    observations_batch[i],
                    self.num_agents,
                    self.obs_dim,
                    self.device,
                )
            )
        global_obs_batch = torch.stack(global_obs_list).to(self.device)
        
        # Construct global prev_actions for batch
        global_prev_actions_list = []
        for i in range(self.num_agents): # (n_envs, action_dim)
            global_prev_actions_list.append(self.agents[i].prev_actions_batch)
        global_prev_action_onehot_batch = torch.cat(global_prev_actions_list, dim=-1) # (n_envs, num_agents * action_dim)
        
        # 2. Masks based on episode done status
        masks_list = []
        for i in range(n_envs):
            episode_done = all(self.last_dones_batch[i].get(agent_id, False) for agent_id in range(self.num_agents))
            mask = 1.0 if not episode_done else 0.0
            masks_list.append(mask)
        masks_batch = torch.tensor(masks_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 3. Central Value Step (Batch)
        with torch.no_grad():
            # rnn_states_value_batch: (n_envs, recurrent_N, hidden_dim)
            # value_net expects (batch, recurrent_N, hidden_dim) as hidden state input (checked RNNValueNetwork)
            
            curr_states_value = self.rnn_states_value_batch.clone()
            
            # Forward pass
            # Note: RNNValueNetwork forward expects (x, h, mask)
            # x: (batch, dim) -> unsqueeze? No, MLPBase/RNNLayer usually handles (Batch, Dim) or (Batch, Seq, Dim)
            # Checking algorithms/networks/rnn.py... it uses RNNLayer.
            # RNNLayer forward(x, h, mask): if x is (Batch, Dim), internally adds Seq dim.
            
            values_batch, next_states_value = self.value_net(
                global_obs_batch, 
                self.rnn_states_value_batch, 
                masks_batch, 
                prev_actions=global_prev_action_onehot_batch
            )
            
            # Save next states
            self.rnn_states_value_batch = next_states_value
            
            # Store hidden states for buffer (prev_state)
            # We need to distribute this back to each env
            # values_batch shape follows centralized scalar critic output
        
        # 4. Policy Step (Per Agent)
        for agent_id in range(self.num_agents):
            # Collect observations for this agent across envs
            agent_obs_list = []
            agent_masks_list = []
            agent_avail_list = []
            has_avail = (available_actions_batch is not None)
            
            for i in range(n_envs):
                obs = observations_batch[i][agent_id]
                agent_obs_list.append(obs)
                
                mask = 1.0 if not self.last_dones_batch[i].get(agent_id, False) else 0.0
                agent_masks_list.append(mask)
                self.temp_masks_batch[i][agent_id] = mask
                
                if has_avail and available_actions_batch[i] is not None and agent_id in available_actions_batch[i]:
                    agent_avail_list.append(available_actions_batch[i][agent_id])
                else:
                    agent_avail_list.append(None) # Handle variable avail actions?
                    
            agent_obs_batch = torch.stack(agent_obs_list).to(self.device)
            agent_masks_batch = torch.tensor(agent_masks_list, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                 # Assume if one has availability, all should have (or we need padding/masking)
                 # SMAC always returns available actions.
                 agent_avail_tensor = torch.tensor(agent_avail_list, dtype=torch.float32, device=self.device)
            
            # Get actions batch
            actions, log_probs, prev_p_states, _ = self.agents[agent_id].get_action_batch(
                agent_obs_batch,
                agent_masks_batch,
                available_actions_batch=agent_avail_tensor
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
                # Note: 'curr_states_value[i]' is (recurrent_N, hidden_dim)
                self.temp_hidden_states_batch[i][agent_id] = (
                    prev_p_states[i], 
                    curr_states_value[i]
                )
                
        return actions_batch

    def store_transition_from_env( # for vectorized env
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
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                   self.action_dim, device=self.device
                )
                
        # Central Value RNN Reset (if global done)
        if episode_done:
             self.rnn_states_value_batch[env_idx] = torch.zeros(
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
        
        episode_done = bool(all(self.last_dones.values())) if len(self.last_dones) > 0 else False
        mask0 = 1.0 if not episode_done else 0.0
        mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
        
        # Central Value Step
        global_obs = concatenate_agent_observations(
            observations,
            self.num_agents,
            self.obs_dim,
            self.device,
        )
        _, v_state = self._update_central_value(global_obs, mask_tensor0)
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            action, log_prob, p_state, _ = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            self.temp_log_probs[agent_id] = log_prob
            
        return actions

    def store_transition(self, **kwargs):
        if 'dones' in kwargs:
            self.last_dones = kwargs['dones'].copy()

        kwargs['masks'] = self.temp_masks
        kwargs['hidden_states'] = self.temp_hidden_states
        if 'log_probs' not in kwargs:
            kwargs['log_probs'] = self.temp_log_probs
        self.buffer_manager.store_transition(**kwargs)

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)
        
        # === Prepare Global Data ===
        global_obs_seq = torch.stack([
            concatenate_agent_observations(
                t['bold_o_t'],
                self.num_agents,
                self.obs_dim,
                self.device,
            )
            for t in batch_data
        ]).to(self.device)
        global_masks_seq = torch.tensor(
            [max(t['masks'][i] for i in range(self.num_agents)) for t in batch_data],
            device=self.device
        ).unsqueeze(1)
        
        # Extract available actions
        available_actions_batch = build_available_actions_batch(
            batch_data,
            self.num_agents,
            self.device,
        )
        
        # Central Value Hidden States
        v_states_all = torch.stack([t['hidden_states'][0][1] for t in batch_data]).to(self.device)
        
        # === Extract Pre-computed Advantages and Returns ===
        all_returns = torch.tensor([t['returns'] for t in batch_data], dtype=torch.float32, device=self.device)
        all_advantages = torch.tensor([t['advantages'] for t in batch_data], dtype=torch.float32, device=self.device)

        # Normalize Advantages
        if all_advantages.numel() > 1:
            global_adv_norm = normalize_advantages(all_advantages)
        else:
            global_adv_norm = torch.zeros_like(all_advantages)

        # === Prepare Chunked Data for Value Update ===
        T_total = global_obs_seq.size(0)
        chunk_len = self.rnn_chunk_length
        num_chunks = T_total // chunk_len
        
        if num_chunks > 0:
            global_obs_batch = chunk_sequence(global_obs_seq, chunk_len)
            global_masks_batch = chunk_sequence(global_masks_seq, chunk_len)

            # Active Masks (Padding Mask)
            if 'active_mask' in batch_data[0]:
                active_masks = torch.tensor([t['active_mask'] for t in batch_data], device=self.device).unsqueeze(1) # (T, 1)
                active_masks_chunked = chunk_sequence(active_masks, chunk_len) # (L, B, 1)
                active_masks_batch = active_masks_chunked.reshape(-1)
            else:
                active_masks = torch.ones(T_total, 1, device=self.device)
                active_masks_chunked = chunk_sequence(active_masks, chunk_len)
                active_masks_batch = active_masks_chunked.reshape(-1)
            global_ret_batch = chunk_sequence(all_returns, chunk_len).reshape(-1)
            global_adv_batch = chunk_sequence(global_adv_norm, chunk_len).reshape(-1)
            
            chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
            v_h0_batch = v_states_all[chunk_indices] 
        else:
            global_obs_batch = global_obs_seq.unsqueeze(1)
            global_masks_batch = global_masks_seq
            
            # Fallback Active Masks
            if 'active_mask' in batch_data[0]:
                active_masks = torch.tensor([t['active_mask'] for t in batch_data], device=self.device).unsqueeze(1)
                active_masks_batch = active_masks.squeeze(1)
            else:
                active_masks_batch = torch.ones(T_total, device=self.device)
            global_ret_batch = all_returns
            global_adv_batch = global_adv_norm
            v_h0_batch = v_states_all[0].unsqueeze(0)

        # === Policy Updates (Per Agent) ===
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            agent_active_masks = build_agent_active_masks(batch_data, agent_id, self.device)
            
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            masks_batch = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
            my_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
            
            if num_chunks > 0:
                rnn_obs_batch = chunk_sequence(agent_obs, chunk_len)
                rnn_masks_batch = chunk_sequence(masks_batch, chunk_len)
                my_actions_batch = chunk_sequence(my_actions, chunk_len).reshape(-1)
                p_h0_batch = p_states_all[chunk_indices]
                agent_active_masks_batch = chunk_sequence(agent_active_masks, chunk_len).reshape(-1)
                
                # Construct prev_actions one-hot sequence from buffer
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
                p_h0_batch = p_states_all[0].unsqueeze(0)
                agent_active_masks_batch = agent_active_masks
                
                # Construct prev_actions for single batch
                prev_actions_raw = torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data], 
                    dtype=torch.long, device=self.device
                )
                prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
                prev_actions_chunked = prev_actions_onehot_seq.unsqueeze(1)
                
                avail_actions_batch_merged = None
                if agent_id in available_actions_batch:
                    avail_actions_batch_merged = available_actions_batch[agent_id]

            # Use stored log_probs from rollout (not recalculated)
            old_log_probs_raw = torch.tensor([t['log_probs'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            if num_chunks > 0:
                old_log_probs = chunk_sequence(old_log_probs_raw, chunk_len).reshape(-1)
            else:
                old_log_probs = old_log_probs_raw
        
            for _ in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(
                    rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, 
                    available_actions=avail_actions_batch_merged, prev_actions=prev_actions_chunked
                )
                adv_target = global_adv_batch

                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * adv_target 
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_target
                
                # Masked Policy Loss
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * agent_active_masks_batch).sum() / (agent_active_masks_batch.sum() + 1e-8)
                
                # Masked Entropy Loss
                entropy_loss = -entropy # entropy is (T)
                entropy_loss = (entropy_loss * agent_active_masks_batch).sum() / (agent_active_masks_batch.sum() + 1e-8)
                
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()

        # === Prepare Joint Prev Actions for Central Value ===
        # Reconstruct the joint prev_actions for the batch across all agents
        agent_prev_actions_list = []
        for agent_id in range(self.num_agents):
            agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
            agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
            agent_prev_actions_list.append(agent_prev_actions_onehot)
        
        joint_prev_actions_batch_raw = torch.cat(agent_prev_actions_list, dim=-1) # (T_total, num_agents * action_dim)
        
        if num_chunks > 0:
            global_prev_actions_batch = chunk_sequence(joint_prev_actions_batch_raw, chunk_len) # (L, B, num_agents * action_dim)
        else:
            global_prev_actions_batch = joint_prev_actions_batch_raw.unsqueeze(1) # (T, 1, num_agents * action_dim)

        # === Central Value Update ===
        for _ in range(self.ppo_epochs):
             new_vals, _ = self.value_net(global_obs_batch, v_h0_batch, global_masks_batch, prev_actions=global_prev_actions_batch)
             
             val_error = (new_vals.squeeze(1).squeeze(-1) - global_ret_batch) ** 2
             
             # Masked Value Loss
             active_masks_batch_val = active_masks_batch

             val_loss = (val_error * active_masks_batch_val).sum() / (active_masks_batch_val.sum() + 1e-8)
             
             self.value_optimizer.zero_grad()
             (self.value_coef * val_loss).backward()
             torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
             self.value_optimizer.step()
             
             agent_losses['value_loss'] += val_loss.item()

        # Normalize value loss logging
        agent_losses['value_loss'] /= self.ppo_epochs
        
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

            # Extract data for this trajectory
            T = len(traj)
            # Optimized: Stack on CPU first, then single GPU transfer
            global_obs_list = [
                concatenate_agent_observations(
                    t['bold_o_t'],
                    self.num_agents,
                    self.obs_dim,
                    self.device,
                )
                for t in traj
            ]
            global_obs_seq = torch.stack(global_obs_list).to(self.device, non_blocking=True)
            global_masks_seq = torch.tensor(
                [max(t['masks'][i] for i in range(self.num_agents)) for t in traj],
                device=self.device
            ).unsqueeze(1)

            # Rewards - Direct GPU tensor creation
            rewards_seq = torch.tensor(
                [sum(t['bold_r_t'].values()) / self.num_agents for t in traj],
                dtype=torch.float32,
                device=self.device,
            )

            # Dones
            team_dones = build_team_done_sequence(traj)
            dones_seq = torch.tensor(team_dones, dtype=torch.float32, device=self.device)

            # Initial hidden state for value network
            v_h0 = traj[0]['hidden_states'][0][1].to(self.device, non_blocking=True).unsqueeze(0)

            # Compute values for entire trajectory
            with torch.no_grad():
                last_next_obs = concatenate_agent_observations(
                    traj[-1]['next_observations'],
                    self.num_agents,
                    self.obs_dim,
                    self.device,
                ).to(self.device, non_blocking=True)
                last_mask = 1.0 - (1.0 if team_dones[-1] else 0.0)
                
                extended_obs = torch.cat([global_obs_seq.unsqueeze(1), 
                                        last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                extended_masks = torch.cat([global_masks_seq, 
                                            torch.tensor([[last_mask]], device=self.device)], dim=0)
                
                # Extract joint prev_actions for the sequence
                agent_prev_actions_list = []
                for agent_id in range(self.num_agents):
                    agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in traj], dtype=torch.long, device=self.device)
                    agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                    # Pad the last step with the last action
                    last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                    extended_agent_prev_actions = torch.cat([agent_prev_actions_onehot, last_prev_action_onehot], dim=0)
                    agent_prev_actions_list.append(extended_agent_prev_actions)
                    
                joint_prev_actions = torch.cat(agent_prev_actions_list, dim=-1).unsqueeze(1) # (T+1, 1, num_agents * action_dim)
                
                all_values, _ = self.value_net(extended_obs, v_h0, extended_masks, prev_actions=joint_prev_actions)
                
                values = all_values[:-1].view(-1)
                next_val = all_values[-1].item() * last_mask

                advantages, returns = compute_gae(
                    rewards_seq.tolist(),
                    values.tolist(),
                    next_val,
                    dones_seq.tolist(),
                    self.gamma,
                    self.gae_lambda,
                )

                # Store in transitions
                for t_idx, t in enumerate(traj):
                    t['advantages'] = advantages[t_idx]
                    t['returns'] = returns[t_idx]
        
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
        
        # Loop through each mini-batch to perform updates
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
