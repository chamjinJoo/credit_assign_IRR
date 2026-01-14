import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from algorithms.networks import PolicyNetwork
from algorithms.networks.coma_critic import COMACritic
from utils.buffer import EpisodeBuffer

class COMAAgent:
    """
    Single agent for COMA algorithm.
    """
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int, 
        hidden_dim: int = 64,
        lr: float = 3e-4
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        
        # Optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def get_action(self, obs: torch.Tensor) -> Tuple[int, float]:
        """
        Get action from policy.
        """
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_and_log_prob(obs.unsqueeze(0))
        
        return action.item(), log_prob.item()


class COMAAlgorithm:
    """
    COMA (Counterfactual Multi-Agent Policy Gradients) Algorithm implementation.
    """
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95, # Used for TD-lambda in Critic
        clip_epsilon: float = 0.2, # Not used in standard COMA, typically for PPO
        entropy_coef: float = 0.01,
        value_coef: float = 1.0, # Not strictly used as loss is separate, but can be used for weighting
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 1, # COMA is usually off-policy or single update, but we can do multiple epochs
        batch_size: int = 64,
        num_batches: int = 4,
        target_update_interval: int = 200, # Hard update interval for target critic
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self.ppo_epochs = ppo_epochs # We'll treat this as 'updates per batch' if desired
        self.num_batches = num_batches
        
        # Create episode buffer for COMA (episode-based training)
        self.buffer_manager = EpisodeBuffer(max_episodes=100)

        # Create agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = COMAAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr
            )
            
        # Centralized Critic
        self.critic = COMACritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # Target Critic
        self.target_critic = COMACritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            device=device
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self._move_networks_to_device()
        
        self.training_steps = 0
        
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)

    def reset_agents(self):
        pass

    def get_actions(self, observations: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        actions = {}
        for agent_id, obs in observations.items():
            action, log_prob = self.agents[agent_id].get_action(obs)
            actions[agent_id] = (action, log_prob)
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
        info: Dict = None
    ):
        """
        Store transition in episode buffer.
        """
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            hidden_states={i: None for i in range(self.num_agents)},
            info=info
        )
    
    def finish_episode(self):
        """
        Finish current episode and move it to completed episodes.
        Should be called when an episode ends.
        """
        return self.buffer_manager.finish_episode()

    def _to_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: [batch_size, num_agents]
        return F.one_hot(actions, num_classes=self.action_dim).float()

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        if not batch_data:
            return {}

        agent_losses = defaultdict(float)
        
        # --- 데이터 준비 ---
        obs_list, next_obs_list, actions_list, rewards_list, dones_list = [], [], [], [], []
        indiv_obs = [[] for _ in range(self.num_agents)]
        indiv_next_obs = [[] for _ in range(self.num_agents)]

        for t in batch_data:
            obs_list.append(self._concatenate_observations(t['bold_o_t']))
            next_obs_list.append(self._concatenate_observations(t.get('next_observations', t['bold_o_t'])))
            actions_list.append([t['bold_a_t'][i] for i in range(self.num_agents)])
            rewards_list.append(sum(t['bold_r_t'].values()) / self.num_agents) 
            dones_list.append(any(t['dones'].values()))
            for i in range(self.num_agents):
                indiv_obs[i].append(t['bold_o_t'][i])
                indiv_next_obs[i].append(t.get('next_observations', t['bold_o_t'])[i])

        obs_tensor = torch.stack(obs_list).to(self.device)
        next_obs_tensor = torch.stack(next_obs_list).to(self.device)
        actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor = torch.tensor(dones_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        actions_one_hot = self._to_one_hot(actions_tensor)

        # --- Critic Update (COMA: 1-step TD) ---
        with torch.no_grad():
            # 각 agent의 다음 상태 정책 분포 π_i(o_i')
            next_pi_list = []
            for i in range(self.num_agents):
                a_next_obs = torch.stack(indiv_next_obs[i]).to(self.device)
                logits = self.agents[i].policy_net(a_next_obs)
                next_pi_list.append(F.softmax(logits, dim=-1))

            # V(s') = 1/N * Σ_i E_{a^i~π_i}[ Q^i(s', u^-i, a^i) ]
            # 다음 상태에서 다른 에이전트들의 행동 분포를 계산
            v_next_list = []
            for i in range(self.num_agents):
                # 다음 상태에서 다른 에이전트들의 행동 분포를 계산
                # 각 다른 에이전트 j의 행동 분포를 계산 (기대값 사용)
                next_actions_dist = []
                for j in range(self.num_agents):
                    a_next_obs_j = torch.stack(indiv_next_obs[j]).to(self.device)
                    logits_j = self.agents[j].policy_net(a_next_obs_j)
                    next_actions_dist.append(F.softmax(logits_j, dim=-1))
                
                # 다른 에이전트들의 행동 분포를 one-hot 형태로 변환 (기대값 사용)
                # [batch_size, num_agents, action_dim]
                next_actions_one_hot = torch.stack(next_actions_dist, dim=1)
                
                # Q(s', u^-i, a^i) 계산
                # critic 내부에서 agent_id에 해당하는 행동이 마스킹됨
                q_next_all = self.target_critic(
                    next_obs_tensor,
                    next_actions_one_hot,
                    i
                )
                v_i = (next_pi_list[i] * q_next_all).sum(dim=1, keepdim=True)
                v_next_list.append(v_i)

            v_next = torch.stack(v_next_list).mean(dim=0)
            
            # TD(lambda) Target Calculation, 
            # in the paper "An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms 
            # in Complex Fully Cooperative Tasks", they used 10-step TD(lambda) target. 
            targets = torch.zeros_like(rewards_tensor)
            # v_next[t]는 상태 t+1의 값이므로, 마지막 타임스텝의 경우 다음 상태가 없음
            # 역순으로 계산하므로 마지막 타임스텝부터 시작
            running_target = v_next[-1] if len(v_next) > 0 else torch.zeros(1, 1, device=self.device)
            
            for t_idx in reversed(range(len(rewards_tensor))):
                mask = 1.0 - dones_tensor[t_idx]
                # G_t_lambda = r_t + gamma * mask * ( (1-lambda) * v_{t+1} + lambda * G_{t+1}_lambda )
                # v_next[t_idx]는 상태 t+1의 값이므로 올바름
                term1 = (1 - self.gae_lambda) * v_next[t_idx]
                term2 = self.gae_lambda * running_target
                
                targets[t_idx] = rewards_tensor[t_idx] + self.gamma * mask * (term1 + term2)
                
                running_target = targets[t_idx]

        # Critic Loss 계산 및 업데이트
        critic_loss = 0
        for i in range(self.num_agents):
            q_vals = self.critic(obs_tensor, actions_one_hot, i)
            q_taken = q_vals.gather(1, actions_tensor[:, i].unsqueeze(1))
            critic_loss += F.mse_loss(q_taken, targets.detach())

        self.critic_optimizer.zero_grad()
        (critic_loss / self.num_agents).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        agent_losses['critic_loss'] = critic_loss.item() / self.num_agents

        # Actor Update (Counterfactual Advantage) ---
        for i in range(self.num_agents):
            a_obs = torch.stack(indiv_obs[i]).to(self.device)
            logits = self.agents[i].policy_net(a_obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                q_vals_all = self.critic(obs_tensor, actions_one_hot, i)

            # Counterfactual Baseline = \sum_a \pi^i(a|o^i) * Q(s, (u^-i, a))
            baseline = (probs * q_vals_all).sum(dim=1, keepdim=True)
            q_taken = q_vals_all.gather(1, actions_tensor[:, i].unsqueeze(1))
            advantage = (q_taken - baseline).detach()

            # PG Loss + Entropy
            log_prob_taken = log_probs.gather(1, actions_tensor[:, i].unsqueeze(1))
            pg_loss = -(log_prob_taken * advantage).mean()
            entropy = -(probs * log_probs).sum(dim=1).mean()
            
            total_loss = pg_loss - self.entropy_coef * entropy

            self.agents[i].optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].policy_net.parameters(), self.max_grad_norm)
            self.agents[i].optimizer.step()

            agent_losses[f'agent_{i}_pg_loss'] = pg_loss.item()

        # Target Update
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        return dict(agent_losses)

    def update(self, **kwargs):
        """
        Update agents using completed episodes.
        COMA trains on episode-level data, so we process all completed episodes.
        
        Note: Only completed episodes are used for training. Incomplete episodes
        (cut short due to max_steps_per_iteration) remain in current_episode
        and will be continued in the next iteration.
        """
        if not self.buffer_manager.has_completed_episodes():
            return {}

        completed_episodes = self.buffer_manager.get_completed_episodes()
        
        if not completed_episodes:
            return {}
        
        total_losses = defaultdict(float)
        num_episodes_processed = 0
        
        # Process each completed episode
        # Incomplete episodes are not included here (they remain in current_episode)
        for episode in completed_episodes:
            if not episode:
                continue
            
            # Train on the entire episode
            losses = self.update_agents(episode)
            
            for k, v in losses.items():
                total_losses[k] += v

            num_episodes_processed += 1
        
        # Clear completed episodes after processing
        # Note: current_episode (incomplete episodes) is NOT cleared
        self.buffer_manager.clear_completed_episodes()
        
        # Average losses across episodes
        if num_episodes_processed > 0:
            for k in total_losses:
                total_losses[k] /= num_episodes_processed
                
        return total_losses

    def clear_buffer(self):
        """Clear all buffers (both current episode and completed episodes)."""
        self.buffer_manager.clear_all()
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get buffer information for monitoring."""
        return self.buffer_manager.get_buffer_info()
