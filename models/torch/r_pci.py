import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from models.torch.agent_common import RecurrentActorCriticAgent
from models.torch.common import (
    build_agent_active_masks,
    build_agent_masks_batch,
    build_available_actions_batch,
    build_team_done_sequence,
    build_transition_kwargs,
    chunk_sequence,
    classify_staghunt_delta_scenario,
    compute_gae,
    concatenate_agent_observations,
    move_modules_to_device,
    normalize_advantages,
    prepare_env_info,
    resolve_step_episode_done,
    split_action_and_log_probs,
)
from models.torch.networks.rnn import RNNValueNetwork
from models.torch.networks import (
    OutcomeBridgeNetwork,
    NaivePredictor,
)
from models.torch.proxy_shuffle import (
    ProxyShuffleDiagnostics,
    apply_episode_axis_proxy_shuffle,
    assign_proxy_shuffle_metadata,
)
from utils.buffer import TransitionBuffer, EpisodeBuffer

class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=False)

    def set_gamma(self, x: torch.Tensor):
        # median based gamma
        with torch.no_grad():
            if x.size(0) < 2: return
            dists_sq = torch.pdist(x, p=2).pow(2)
            median_dist_sq = dists_sq.median()
            gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            self.log_gamma.data.fill_(torch.log(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_asymmetric(x, x)

    def forward_asymmetric(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gamma = torch.exp(self.log_gamma)
        x1_norm_sq = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm_sq = x2.pow(2).sum(dim=-1, keepdim=True)
        dists_sq = x1_norm_sq - 2 * torch.matmul(x1, x2.T) + x2_norm_sq.T
        dists_sq = torch.clamp(dists_sq, min=0)
        return torch.exp(-gamma * dists_sq)

    def forward_elementwise(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gamma = torch.exp(self.log_gamma)
        dists_sq = torch.sum((x1 - x2)**2, dim=-1) 
        return torch.exp(-gamma * dists_sq)

class ConditionalMeanNet(nn.Module):
    def __init__(self, z_dim, x_dim, action_dim, emb_dim=8, hidden=64):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z, a, x):
        ea = F.one_hot(a.long(), self.action_dim).float()  # (B, action_dim)
        inp = torch.cat([z, x, ea], dim=-1)
        return self.net(inp).squeeze(-1)  # (B,)

class rPCIAgent(RecurrentActorCriticAgent):
    pass

class rPCIAlgorithm:
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1, # RNN layers
        lr_actor: float = 3e-4,
        lr_bridge: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_delta: float = 0.0, # for delta clipping
        causal_lambda: float = 1.0,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.2,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
        bridge_epoch: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        num_batches: int = 4,
        max_buffer_size: int = 10000,
        device: str = "cpu",
        rkhs_gamma: float = 1.0,
        env_required_agents: Optional[Dict[int, List[int]]] = None,
        # Optimization parameter
        causal_mini_batch_size: Optional[int] = 1000,
        use_causal_rff: bool = False,
        causal_rff_features: int = 512,
        causal_rff_seed: int = 0,
        proxy_shuffle_mode: str = "none",
        proxy_shuffle_prob: float = 0.0,
        proxy_shuffle_seed: int = 0,
        proxy_shuffle_strategy: str = "episode_axis",
        tikhonov_gamma: float = 1e-4,
        stabilizer_lambda: float = 1.0,
        train_naivepredictor: bool = False,
        # Value network parameter sharing
        use_value_parameter_sharing: bool = True,
        rnn_chunk_length: int = 16, # Chunk length for RNN training
        use_episode_buffer: bool = False, # Use EpisodeBuffer instead of TransitionBuffer
        use_centralized_critic: bool = False, # MAPPO/IPPO-style centralized critic
        use_recurrent_policy: bool = True,
        global_reward: bool = True,
        add_agent_id: bool = False,
    ):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.base_obs_dim = obs_dim - num_agents if add_agent_id else obs_dim
        if self.base_obs_dim <= 0:
            raise ValueError(
                f"Invalid observation dimensions for add_agent_id={add_agent_id}: "
                f"obs_dim={obs_dim}, num_agents={num_agents}"
            )
        self.policy_obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_reward = global_reward
        self.add_agent_id = add_agent_id
        self.w_dim = self.policy_obs_dim * (num_agents - 1)
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.rnn_chunk_length = rnn_chunk_length
        self.env_required_agents = env_required_agents
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.causal_mini_batch_size = causal_mini_batch_size
        self.use_causal_rff = use_causal_rff
        self.causal_rff_features = max(2, int(causal_rff_features))
        if self.causal_rff_features % 2 != 0:
            self.causal_rff_features += 1
        self.causal_rff_seed = int(causal_rff_seed)
        self.proxy_shuffle_mode = str(proxy_shuffle_mode).lower()
        self.proxy_shuffle_prob = float(proxy_shuffle_prob)
        self.proxy_shuffle_seed = int(proxy_shuffle_seed)
        self.proxy_shuffle_strategy = str(proxy_shuffle_strategy)
        if self.proxy_shuffle_mode not in {"none", "w", "z", "wz"}:
            raise ValueError(f"Unsupported proxy_shuffle_mode: {self.proxy_shuffle_mode}")
        if self.proxy_shuffle_prob < 0.0 or self.proxy_shuffle_prob > 1.0:
            raise ValueError(f"proxy_shuffle_prob must be in [0, 1], got {self.proxy_shuffle_prob}")
        if self.proxy_shuffle_strategy != "episode_axis":
            raise ValueError(f"Unsupported proxy_shuffle_strategy: {self.proxy_shuffle_strategy}")
        self._rff_omega = None
        self._rff_signature = None
        self.train_naivepredictor = train_naivepredictor
        self.use_value_parameter_sharing = use_value_parameter_sharing
        self.use_episode_buffer = use_episode_buffer
        self.use_centralized_critic = use_centralized_critic
        self.use_recurrent_policy = use_recurrent_policy
        self.bridge_hidden_dim = hidden_dim * recurrent_N * (num_agents - 1) if use_recurrent_policy else 0
        self.prev_action_embedding_dim = min(8, max(4, math.ceil(math.log2(max(action_dim, 2)))))
        # Previous per-agent embedding concat path:
        # self.bridge_prev_action_dim = self.prev_action_embedding_dim * (num_agents - 1)
        self.bridge_prev_action_dim = action_dim * (num_agents - 1)
        self.kernel_prev_action_dim = action_dim
        self.bridge_context_dim = self.bridge_hidden_dim + self.bridge_prev_action_dim

        # for causal intrinsic reward
        self.clip_delta = clip_delta
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch

        # training steps
        self.training_steps = 0
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_batches = num_batches
        
        if self.use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=max_buffer_size) # Treat buffer_size as max_episodes roughly
        else:
            self.buffer_manager = TransitionBuffer(
                max_buffer_size=max_buffer_size
            )
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = rPCIAgent(
                agent_id=i,
                obs_dim=self.policy_obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                use_recurrent_policy=use_recurrent_policy,
            )
        
        # Value network 설정
        if self.use_centralized_critic:
            # MAPPO-style centralized critic
            self.value_net = RNNValueNetwork(
                    obs_dim=self.policy_obs_dim * num_agents,
                    hidden_dim=hidden_dim,
                    recurrent_N=recurrent_N,
                    out_dim=1 if global_reward else num_agents,
                    action_dim=action_dim,
                    num_agents=num_agents
            )
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
            self.rnn_state_value = None  # 초기화는 reset_agents에서
            self.value_nets = {agent_id: self.value_net for agent_id in range(num_agents)}
            self.value_optimizers = {agent_id: self.value_optimizer for agent_id in range(num_agents)}
            critic_out_dim = 1 if global_reward else num_agents
            print(f'[R-PCI] Using MAPPO-style centralized critic (input_dim={self.policy_obs_dim * num_agents}, out_dim={critic_out_dim})')

        elif self.use_value_parameter_sharing:
            # one shared value network (IPPO-style)
            self.shared_value_net = RNNValueNetwork(obs_dim=self.policy_obs_dim, hidden_dim=hidden_dim, recurrent_N=recurrent_N, 
                                out_dim=1, action_dim=action_dim, num_agents=1)
             
            self.shared_value_optimizer = optim.Adam(self.shared_value_net.parameters(), lr=lr_critic)
            
            # parameter sharing
            for agent_id, agent in self.agents.items(): 
                self.agents[agent_id].value_net = self.shared_value_net
                self.agents[agent_id].value_optimizer = self.shared_value_optimizer 

            # for checkpointing and loading
            self.value_nets = {agent_id: self.shared_value_net for agent_id in range(num_agents)}
            self.value_optimizers = {agent_id: self.shared_value_optimizer for agent_id in range(num_agents)}

        else: # individual value network for each agent
            self.value_nets = {agent_id: agent.value_net for agent_id, agent in self.agents.items()}
            self.value_optimizers = {agent_id: agent.value_optimizer for agent_id, agent in self.agents.items()}

        self.rnn_state_delta_value = None
        self.rnn_states_delta_value = None
        self.rnn_states_delta_value_batch = None
        delta_obs_dim = self.policy_obs_dim * num_agents if self.use_centralized_critic else self.policy_obs_dim
        delta_num_agents = num_agents if self.use_centralized_critic else 1
        delta_out_dim = num_agents if self.use_centralized_critic else 1
        self.delta_value_net = RNNValueNetwork(
            obs_dim=delta_obs_dim,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_N,
            out_dim=delta_out_dim,
            action_dim=action_dim,
            num_agents=delta_num_agents,
        )
        self.delta_value_optimizer = optim.Adam(self.delta_value_net.parameters(), lr=lr_critic)
        if self.use_centralized_critic:
            print(f'[R-PCI] Using centralized delta critic (input_dim={delta_obs_dim}, out_dim={delta_out_dim})')
        else:
            print(f'[R-PCI] Using shared per-agent delta critic (input_dim={delta_obs_dim}, out_dim={delta_out_dim})')
        
        # Current proxy setting:
        #   W^i_t = other agents' current observations
        #   Z^i_t = agent i's previous hidden state h^i_{t-1}
        #   X^i_t = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1})
        x_dim = self.policy_obs_dim + self.bridge_context_dim
        

        self.outcome_bridge_net = OutcomeBridgeNetwork( 
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=self.policy_obs_dim * (num_agents-1), # W^i_t = others' observations
            outcome_dim=1 if global_reward else num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            x_dim=x_dim
        )
        
        self.prev_action_embedding = nn.Embedding(action_dim, self.prev_action_embedding_dim)
        nn.init.normal_(
            self.prev_action_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.prev_action_embedding_dim),
        )
        for param in self.prev_action_embedding.parameters():
            param.requires_grad = False
        # ---------------------------------------------------------------------------------------- #

        # --- Legacy single-kernel path (kept for compatibility / experimentation) ---
        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device) # default gamma=1.0

        # --- Block-wise product kernel over active bridge covariates ---
        # Current kernel factorization is over:
        #   Z^i_t = agent i's previous hidden state h^i_{t-1},
        #   A^i_t = my action,
        #   X^i_t = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1}).
        self.kernel_z_block = RBFKernel(gamma=1.0).to(self.device)     # kernel for Z^i_t
        self.kernel_a_block = RBFKernel(gamma=1.0).to(self.device)     # kernel for A^i_t one-hot
        
        # kernel X is splited again. 
        self.kernel_xobs_block = RBFKernel(gamma=1.0).to(self.device)  # reserved for an observation sub-block inside X
        self.kernel_xhid_block = RBFKernel(gamma=1.0).to(self.device)  # kernel for hidden-state block in X^i_t
        self.kernel_xact_block = RBFKernel(gamma=1.0).to(self.device)  # reserved for action sub-block inside X
        
        self.stabilizer_lambda = stabilizer_lambda
        self.tikhonov_gamma = tikhonov_gamma

        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        
        if self.train_naivepredictor:
            self.naive_predictor = NaivePredictor(
                action_dim=action_dim,
                num_agents=num_agents,
                w_dim=self.w_dim,
                x_dim=x_dim,
                output_dim=1 if global_reward else num_agents,
                hidden_dim=hidden_dim
            )
            self.naive_optimizer = optim.Adam(self.naive_predictor.parameters(), lr=lr_bridge)
        
        # Keep track of done status for masks
        self.last_dones = {i: False for i in range(num_agents)}
        
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            move_modules_to_device(self.device, agent.policy_net)
            if not self.use_value_parameter_sharing and not self.use_centralized_critic:
                move_modules_to_device(self.device, agent.value_net)
        if self.use_centralized_critic:
            move_modules_to_device(self.device, self.value_net)
        elif self.use_value_parameter_sharing:
            move_modules_to_device(self.device, self.shared_value_net)
        move_modules_to_device(self.device, self.delta_value_net)
        move_modules_to_device(
            self.device,
            self.outcome_bridge_net,
            self.critic_kernel_w,
            self.prev_action_embedding,
        )
        if getattr(self, "train_naivepredictor", False):
            move_modules_to_device(self.device, self.naive_predictor)
        move_modules_to_device(
            self.device,
            self.kernel_z_block,
            self.kernel_a_block,
            self.kernel_xobs_block,
            self.kernel_xhid_block,
            self.kernel_xact_block,
        )
    
    def get_current_causal_lambda(self) -> float:
        if not self.causal_lambda_schedule:
            return self.causal_lambda
        if self.training_steps < self.causal_lambda_startingpoint:
            return self.causal_lambda_start
        effective_steps = self.training_steps - self.causal_lambda_startingpoint
        if effective_steps >= self.causal_lambda_warmup_steps:
            return self.causal_lambda_final
        else:
            progress = effective_steps / self.causal_lambda_warmup_steps
            return self.causal_lambda_start + progress * (self.causal_lambda_final - self.causal_lambda_start)

    def _global_reward_from_dict(self, rewards: Dict[int, float]) -> float:
        return sum(rewards.values())

    def _build_reward_batch_tensor(self, transitions: List[Dict]) -> torch.Tensor:
        if self.global_reward:
            return torch.tensor(
                [[self._global_reward_from_dict(t['bold_r_t'])] for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )
        return torch.tensor(
            [[t['bold_r_t'][i] for i in range(self.num_agents)] for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def _standardize_kernel_block(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True, unbiased=False).clamp_min(eps)

    def _normalize_z_kernel_block(self, z: torch.Tensor, joint_obs_dim: int, eps: float = 1e-6) -> torch.Tensor:
        if z.size(1) == joint_obs_dim + self.num_agents:
            z_obs_n = self._standardize_kernel_block(z[:, :joint_obs_dim], eps=eps)
            z_rew_n = self._standardize_kernel_block(z[:, joint_obs_dim:], eps=eps)
            return torch.cat([z_obs_n, z_rew_n], dim=1)
        return self._standardize_kernel_block(z, eps=eps)


    def _rff_kernel_weighted_residual(self, phi: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        feature_dim = phi.size(1)
        eye = torch.eye(feature_dim, device=phi.device, dtype=phi.dtype)
        system = self.tikhonov_gamma * eye + self.stabilizer_lambda * (phi.T @ phi)
        rhs = phi.T @ residuals
        try:
            coeff = torch.linalg.solve(system, rhs)
        except RuntimeError:
            jitter = max(float(self.tikhonov_gamma), 1e-6)
            coeff = torch.linalg.solve(system + jitter * eye, rhs)
        return phi @ coeff

    def _sample_rff_offdiag_values(self, phi: torch.Tensor, max_pairs: int = 4096) -> Optional[torch.Tensor]:
        n = phi.size(0)
        if n < 2:
            return None
        pair_count = min(max_pairs, n * (n - 1))
        row = torch.randint(n, (pair_count,), device=phi.device)
        col = torch.randint(n - 1, (pair_count,), device=phi.device)
        col = col + (col >= row).long()
        return (phi[row] * phi[col]).sum(dim=-1)
    
    def _iterate_causal_minibatches(self, te_total: int, device):
        N = self.num_agents
        chunk_size = (
            min(self.causal_mini_batch_size, te_total)
            if (self.causal_mini_batch_size and self.causal_mini_batch_size > 0)
            else te_total
        )
        te_idx = torch.arange(te_total, device=device)
        for te_start in range(0, te_total, chunk_size):
            te_end = te_start + chunk_size
            if te_end <= te_total:
                te_chunk = te_idx[te_start:te_end]
            else:
                pad = chunk_size - (te_total - te_start)
                te_chunk = torch.cat([te_idx[te_start:], te_idx[:pad]])
            yield (te_chunk.unsqueeze(1) * N + torch.arange(N, device=device).unsqueeze(0)).reshape(-1)

    def update_causal_networks(self, batch_data: List[Dict]) -> Dict[str, float]:
        if self.get_current_causal_lambda() == 0.0:
            return {'outcome_bridge_loss': 0.0}
        
        batch_size = len(batch_data)
        num_agents = self.num_agents
        device = self.device

        # === 1. Data Preparation (Vectorized) ===
        prev_action_batch = torch.tensor(
            [[t['bold_a_t_minus_1'][i] for i in range(num_agents)] for t in batch_data],
            dtype=torch.long,
            device=device,
        )
        obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(device)
        prev_obs_batch = torch.stack([torch.stack(list(t['bold_o_t_minus_1'].values())) for t in batch_data]).to(device)
        action_batch = torch.tensor([[t['bold_a_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        reward_batch = self._build_reward_batch_tensor(batch_data)
        dones_batch = torch.tensor([[t['dones'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device)
        prev_actions_onehot_batch = F.one_hot(prev_action_batch, self.action_dim).float()
        # Previous per-agent embedding concat path:
        # prev_actions_emb_batch = self.prev_action_embedding(prev_action_batch)

        # Extract hidden states
        hidden_states_batch = {}    # h_{t-1}: for policy evaluation / bridge X construction
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
            if any(batch_data[0]['hidden_states'].values()):
                for agent_id in range(self.num_agents):
                    if agent_id in batch_data[0]['hidden_states']:
                        p_states = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device) # (Batch, recurrent_N, hidden_dim)
                        hidden_states_batch[agent_id] = p_states

        A_super = action_batch.view(-1)
        expanded_obs = obs_batch.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.policy_obs_dim
        )
        obs_others_super = expanded_obs[mask].view(batch_size * num_agents, -1)

        expanded_prev_actions = prev_actions_onehot_batch.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask_prev_actions = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.action_dim
        )
        prev_action_others_super = expanded_prev_actions[mask_prev_actions].view(
            batch_size * num_agents, -1
        )
        prev_action_others_kernel_super = expanded_prev_actions[mask_prev_actions].view(
            batch_size * num_agents, -1
        )
        W_super = obs_others_super # Default W
        Y_super = reward_batch.repeat_interleave(num_agents, dim=0)

        # 1. Gather h_{t-1} for all agents: (B, N, RecN*H)  — used by evaluate_actions
        h_list = []
        for i in range(num_agents):
            if i in hidden_states_batch:
                h_list.append(hidden_states_batch[i].view(batch_size, -1))
            else:
                h_list.append(torch.zeros(batch_size, self.recurrent_N * self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1)  # (B, N, RecN*H)

        obs_super = obs_batch.view(-1, self.policy_obs_dim)
        
        # 2. Construct X_super = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1})
        if self.use_recurrent_policy:
            H_all_expanded = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (B, N, N, RecN*H)
            hid_mask = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(
                batch_size, -1, -1, self.recurrent_N * self.hidden_dim
            )
            H_others_super = H_all_expanded[hid_mask].view(batch_size * num_agents, -1)  # (B*N, (N-1)*RecN*H)
            X_super = torch.cat([obs_super, H_others_super, prev_action_others_super], dim=1)
            X_kernel_super = torch.cat([obs_super, H_others_super, prev_action_others_kernel_super], dim=1)
        else:
            X_super = torch.cat([obs_super, prev_action_others_super], dim=1)
            X_kernel_super = torch.cat([obs_super, prev_action_others_kernel_super], dim=1)

        if self.use_recurrent_policy:
            Z_super = H_all.reshape(batch_size * num_agents, -1) # Z^i_t = h^i_{t-1}
        else:
            Z_super = torch.zeros(batch_size * num_agents, 0, device=device)

        with torch.no_grad():
            log_probs_list = []
            for i in range(num_agents):
                obs_i = obs_batch[:, i, :]
                act_i = action_batch[:, i]
                prev_actions_i = prev_actions_onehot_batch[:, i, :]
                
                # Handling RNN states
                if i in hidden_states_batch:                    
                    p_state = hidden_states_batch[i]
                    # Construct masks
                    masks_i = torch.ones_like(dones_batch[:, i]).unsqueeze(1) # (Batch, 1)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(
                        obs_i,
                        p_state,
                        act_i,
                        masks_i,
                        prev_actions=prev_actions_i,
                    )
                else:
                    # Fallback to zeros if no state
                    bs = obs_i.size(0)
                    dummy_state = torch.zeros(bs, self.recurrent_N, self.hidden_dim, device=device)
                    dummy_mask = torch.ones(bs, 1, device=device)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(
                        obs_i,
                        dummy_state,
                        act_i,
                        dummy_mask,
                        prev_actions=prev_actions_i,
                    )

                log_probs_list.append(lp)
            
            policy_log_probs = torch.stack(log_probs_list, dim=1).view(-1)
            policy_probs_super = torch.exp(policy_log_probs) 
            
        obs_super = obs_batch.view(-1, self.policy_obs_dim)

        # Episode end mask: Global per-timestep (same for all N agents), 0 at episode-terminal transitions.
        team_dones = build_team_done_sequence(batch_data)  # List[bool], length B
        episode_done_t = torch.tensor(team_dones, dtype=torch.float32, device=device)  # (B,)
        loss_mask_super = (1.0 - episode_done_t).unsqueeze(1).expand(-1, num_agents).reshape(-1)  # (B*N,)

        active_mask_t = torch.tensor(
            [float(t.get('active_mask', 1.0)) for t in batch_data],
            dtype=torch.float32,
            device=device,
        )
        proxy_valid_super = ((1.0 - episode_done_t) * active_mask_t).unsqueeze(1).expand(
            -1, num_agents
        ).reshape(-1)
        proxy_episode_ids = torch.tensor(
            [int(t.get('_proxy_episode_id', idx)) for idx, t in enumerate(batch_data)],
            dtype=torch.long,
            device=device,
        )
        proxy_timesteps = torch.tensor(
            [int(t.get('_proxy_timestep', idx)) for idx, t in enumerate(batch_data)],
            dtype=torch.long,
            device=device,
        )

        # === 2. Mini-batch Training Loop ===
        # the per-agent Gram size is causal_mini_batch_size. 
        # Each chunk yields causal_mini_batch_size * N flat indices
        total_samples = A_super.size(0)
        N = self.num_agents
        te_total = total_samples // N
        chunk_size = (
            min(self.causal_mini_batch_size, te_total)
            if (self.causal_mini_batch_size and self.causal_mini_batch_size > 0)
            else te_total
        )
        B_n = chunk_size  # per-agent Gram size, uniform across all agents and chunks

        avg_outcome_loss = 0.0
        avg_action_loss = 0.0
        avg_naive_loss = 0.0

        sum_off_mean = 0.0
        sum_off_std = 0.0
        proxy_shuffle_diagnostics = ProxyShuffleDiagnostics()
        num_updates = 0

        for idx in self._iterate_causal_minibatches(te_total, device):
            a_mb = A_super[idx]; w_mb = W_super[idx]; y_mb = Y_super[idx]
            z_mb = Z_super[idx]; x_mb = X_super[idx]; x_kernel_mb = X_kernel_super[idx]
            loss_mask_mb = loss_mask_super[idx]  # (B_n * N,)
            proxy_valid_mb = proxy_valid_super[idx]
            transition_idx_mb = idx // N

            w_for_bridge = w_mb
            z_for_kernel = z_mb
            w_donors = {}
            if self.proxy_shuffle_mode in {"w", "wz"} and self.proxy_shuffle_prob > 0.0:
                w_for_bridge, w_diag, w_donors = apply_episode_axis_proxy_shuffle(
                    w_mb,
                    transition_idx_mb,
                    proxy_episode_ids,
                    proxy_timesteps,
                    proxy_valid_mb,
                    num_agents=N,
                    mode=self.proxy_shuffle_mode,
                    prob=self.proxy_shuffle_prob,
                    base_seed=self.proxy_shuffle_seed,
                    training_step=self.training_steps,
                    proxy_type="w",
                    strategy=self.proxy_shuffle_strategy,
                )
                proxy_shuffle_diagnostics.update(w_diag)
            if self.proxy_shuffle_mode in {"z", "wz"} and self.proxy_shuffle_prob > 0.0:
                z_for_kernel, z_diag, _ = apply_episode_axis_proxy_shuffle(
                    z_mb,
                    transition_idx_mb,
                    proxy_episode_ids,
                    proxy_timesteps,
                    proxy_valid_mb,
                    num_agents=N,
                    mode=self.proxy_shuffle_mode,
                    prob=self.proxy_shuffle_prob,
                    base_seed=self.proxy_shuffle_seed,
                    training_step=self.training_steps,
                    proxy_type="z",
                    strategy=self.proxy_shuffle_strategy,
                    forbidden_donors=w_donors if self.proxy_shuffle_mode == "wz" else None,
                )
                proxy_shuffle_diagnostics.update(z_diag)

            assert w_for_bridge.shape == w_mb.shape
            assert z_for_kernel.shape == z_mb.shape
            assert w_for_bridge.dtype == w_mb.dtype
            assert z_for_kernel.dtype == z_mb.dtype
            assert w_for_bridge.device == w_mb.device
            assert z_for_kernel.device == z_mb.device
            if self.proxy_shuffle_mode == "none" or self.proxy_shuffle_prob == 0.0:
                assert torch.equal(w_for_bridge, w_mb)
                assert torch.equal(z_for_kernel, z_mb)

            eps = 1e-6
            joint_obs_dim = self.policy_obs_dim * self.num_agents

            # Reshape interleaved (B_n*N, ...) → per-agent slices without sorting.
            # idx layout: idx[b*N + n] = te_pair b, agent n, so reshape(B_n, N, *) is exact.
            a_by_agent  = a_mb.reshape(B_n, N)                   # (B_n, N)
            z_by_agent  = z_for_kernel.reshape(B_n, N, -1)       # (B_n, N, z_dim)
            w_by_agent  = w_for_bridge.reshape(B_n, N, -1)
            y_by_agent  = y_mb.reshape(B_n, N, -1)
            x_by_agent  = x_mb.reshape(B_n, N, -1)
            xk_by_agent = x_kernel_mb.reshape(B_n, N, -1)        # (B_n, N, kernel_dim)
            mask_by_agent = loss_mask_mb.reshape(B_n, N)          # (B_n, N)

            # Derive z_dim from the first agent slice for gamma computation.
            z_dim_sample = z_by_agent[:, 0, :].shape[1]
            x_act_dim = xk_by_agent.shape[-1] - self.policy_obs_dim - self.bridge_hidden_dim
            gamma_z = 1.0 / max(z_dim_sample, 1)
            gamma_a = 1.0
            gamma_xobs = 1.0 / max(self.policy_obs_dim, 1)
            gamma_xact = 1.0

            with torch.no_grad():
                self.kernel_z_block.log_gamma.data.fill_(math.log(gamma_z))
                self.kernel_a_block.log_gamma.data.fill_(math.log(gamma_a))
                if self.policy_obs_dim > 0:
                    self.kernel_xobs_block.log_gamma.data.fill_(math.log(gamma_xobs))
                if x_act_dim > 0:
                    self.kernel_xact_block.log_gamma.data.fill_(math.log(gamma_xact))

            # --- Per-agent kernel and regularized kernel ---
            K_blocks     = []
            reg_K_blocks = []
            rff_blocks   = []
            off_vals_list = []

            for ag_i in range(num_agents):
                Bi = B_n  # uniform per-agent Gram size

                z_i    = z_by_agent[:, ag_i, :]                   # (B_n, z_dim)
                xk_i   = xk_by_agent[:, ag_i, :]                  # (B_n, kernel_dim)
                xobs_cont_i = xk_i[:, :self.base_obs_dim]
                xobs_id_i   = xk_i[:, self.base_obs_dim:self.policy_obs_dim]
                xhid_i      = xk_i[:, self.policy_obs_dim:self.policy_obs_dim + self.bridge_hidden_dim]
                xact_i      = xk_i[:, self.policy_obs_dim + self.bridge_hidden_dim:]
                a_i         = a_by_agent[:, ag_i]                 # (B_n,)

                z_norm_i = self._normalize_z_kernel_block(z_i, joint_obs_dim, eps=eps)

                if xobs_cont_i.size(1) > 0:
                    xobs_cont_norm_i = self._standardize_kernel_block(xobs_cont_i, eps=eps)
                else:
                    xobs_cont_norm_i = xobs_cont_i
                xobs_norm_i = torch.cat([xobs_cont_norm_i, xobs_id_i], dim=1)

                if xhid_i.size(1) > 0:
                    xhid_norm_i = self._standardize_kernel_block(xhid_i, eps=eps)
                else:
                    xhid_norm_i = xhid_i

                a_oh_i = F.one_hot(a_i.long(), self.action_dim).float()
                gamma_xhid_i = None
                if xhid_norm_i.size(1) > 0:
                    effective_dim = torch.mean(
                        torch.sum(torch.abs(xhid_norm_i) > 1e-6, dim=-1).float()
                    ).clamp_min(1.0)
                    gamma_xhid_i = 1.0 / max(effective_dim.item(), 1.0)

                K_i = self.kernel_z_block(z_norm_i) * self.kernel_a_block(a_oh_i)
                if xobs_norm_i.size(1) > 0:
                    K_i = K_i * self.kernel_xobs_block(xobs_norm_i)
                if xhid_norm_i.size(1) > 0 and gamma_xhid_i is not None:
                    with torch.no_grad():
                        self.kernel_xhid_block.log_gamma.data.fill_(math.log(gamma_xhid_i))
                    K_i = K_i * self.kernel_xhid_block(xhid_norm_i)
                if xact_i.size(1) > 0:
                    K_i = K_i * self.kernel_xact_block(xact_i)

                # Monitor off-diagonal within this agent's block.
                with torch.no_grad():
                    if Bi >= 2:
                        off_mask_i = ~torch.eye(Bi, dtype=torch.bool, device=device)
                        off_vals_list.append(K_i[off_mask_i].detach())

                reg_K_i = self.tikhonov_gamma * torch.eye(Bi, device=device) + self.stabilizer_lambda * K_i
                try:
                    torch.linalg.solve(reg_K_i, y_by_agent[:, ag_i, :])  # check regularity
                except RuntimeError:
                    reg_K_i = reg_K_i + 1e-3 * torch.eye(Bi, device=device)

                K_blocks.append(K_i)
                reg_K_blocks.append(reg_K_i)
                rff_blocks.append(None)

            # Aggregate off-diagonal stats across all agent blocks.
            if off_vals_list:
                all_off = torch.cat(off_vals_list)
                sum_off_mean += all_off.mean().item()
                sum_off_std  += all_off.std(unbiased=False).item()
            # --- Update Outcome Bridge h_φ for bridge_epoch times ---
            # Forward pass on full interleaved chunk; per-agent solve on B_n×B_n blocks.
            a_flat = a_by_agent.reshape(-1)
            w_flat = w_by_agent.reshape(B_n * N, -1)
            x_flat = x_by_agent.reshape(B_n * N, -1)
            epoch_outcome_loss = 0.0
            epoch_naive_loss = 0.0
            for _ in range(self.bridge_epoch):
                y_hat = self.outcome_bridge_net(a_flat, w_flat, x=x_flat)  # (B_n*N, d_y)
                y_hat_by_agent = y_hat.reshape(B_n, N, -1)                 # (B_n, N, d_y)

                per_agent_losses = []
                for ag_i in range(num_agents):
                    y_hat_i = y_hat_by_agent[:, ag_i, :]  # (B_n, d_y)
                    y_i     = y_by_agent[:, ag_i, :]       # (B_n, d_y)
                    mask_i  = mask_by_agent[:, ag_i]       # (B_n,)
                    res_i   = y_i - y_hat_i                 # (B_n, d_y)

                    phi_i = rff_blocks[ag_i]
                    if phi_i is not None:
                        weighted_res_i = self._rff_kernel_weighted_residual(phi_i, res_i)
                    else:
                        K_i     = K_blocks[ag_i]
                        reg_K_i = reg_K_blocks[ag_i]
                        solved_i = torch.linalg.solve(reg_K_i, res_i)
                        weighted_res_i = K_i @ solved_i

                    per_sample_i = (res_i * weighted_res_i).sum(dim=1)     # (B_n,)
                    # Exclude episode-terminal transitions from the loss average.
                    # Gram is computed over all samples; mask only weights the mean.
                    denom_i = mask_i.sum().clamp(min=1.0)
                    per_agent_losses.append((per_sample_i * mask_i).sum() / denom_i)

                # Per-agent averaging: (1/N) Σ_i (1/|B_i|) r_i^T K_i (γI+λK_i)^{-1} r_i
                loss_h = torch.stack(per_agent_losses).mean() if per_agent_losses else torch.tensor(0.0, device=device)

                # --- Update Action Bridge q_θ ---
                loss_q = torch.tensor(0.0, device=device)

                total_loss = loss_h + loss_q

                self.outcome_bridge_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.outcome_bridge_net.parameters(), self.max_grad_norm)
                self.outcome_bridge_optimizer.step()

                epoch_outcome_loss += loss_h.item()

                # --- Update Naive Predictor ---
                if getattr(self, "train_naivepredictor", False):
                    naive_output = self.naive_predictor(a_mb, w_mb, x_mb) # (B, N)
                    naive_loss = F.mse_loss(naive_output, y_mb)
                    self.naive_optimizer.zero_grad()
                    naive_loss.backward()
                    self.naive_optimizer.step()
                    epoch_naive_loss += naive_loss.item()

            avg_outcome_loss += epoch_outcome_loss / self.bridge_epoch
            avg_action_loss += loss_q.item()
            if getattr(self, "train_naivepredictor", False):
                avg_naive_loss += epoch_naive_loss / self.bridge_epoch
                
            num_updates += 1

        return_dict = {
            'outcome_bridge_loss': avg_outcome_loss / num_updates,
            'Kz_offdiag_mean': sum_off_mean / num_updates, # close to 0: Kernel is close to Identity I, close to 1: Kenrel is close to 11^\top
            'Kz_offdiag_std': sum_off_std / num_updates,
        }
        return_dict.update(proxy_shuffle_diagnostics.as_metrics())
        
        if getattr(self, "train_naivepredictor", False):
            return_dict['naive_loss'] = avg_naive_loss / num_updates

        return return_dict

    def reset_agents(self):
        """Reset agent hidden states."""
        for agent in self.agents.values():
            agent.reset_rnn_state(device=self.device)
        if self.use_centralized_critic:
            self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        if self.use_centralized_critic:
            self.rnn_state_delta_value = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        else:
            self.rnn_states_delta_value = {
                agent_id: torch.zeros(self.recurrent_N, self.hidden_dim, device=self.device)
                for agent_id in range(self.num_agents)
            }
        self.last_dones = {i: False for i in range(self.num_agents)}
    
    def reset_agents_batch(self, n_envs: int):
        """Reset agent hidden states for batch mode (parallel environments)."""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs, device=self.device)
        if self.use_centralized_critic:
            self.rnn_states_value_batch = torch.zeros(
                n_envs, self.recurrent_N, self.hidden_dim, device=self.device
            )
        if self.use_centralized_critic:
            self.rnn_states_delta_value_batch = torch.zeros(
                n_envs, self.recurrent_N, self.hidden_dim, device=self.device
            )
        else:
            self.rnn_states_delta_value_batch = {
                agent_id: torch.zeros(n_envs, self.recurrent_N, self.hidden_dim, device=self.device)
                for agent_id in range(self.num_agents)
            }
        # Track done status for each agent in each environment
        self.last_dones_batch = {i: [False] * n_envs for i in range(self.num_agents)}

    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {} # Store current states for store_transition
        self.temp_masks = {} # Store masks used
        self.temp_log_probs = {} # Store log_probs from rollout
        
        # Centralized critic: compute central value once
        if self.use_centralized_critic:
            central_v_state = self._compute_central_value_state(observations)
        if self.use_centralized_critic:
            central_delta_v_state = self._compute_central_delta_value_state(observations)
        else:
            delta_v_states = {}
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            # --- Fix: Cache a_{t-1} before get_action updates it to a_t ---
            agent_prev_action_onehot = self.agents[agent_id].prev_action.unsqueeze(0).clone()

            # Policy
            action, log_prob, p_state, p_next_state = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)

            if self.use_centralized_critic:
                v_state = central_v_state
            else:
                # Value (Update state) - IPPO style
                _, v_state = self.agents[agent_id].get_value(
                    obs,
                    mask_tensor,
                    prev_action_onehot=agent_prev_action_onehot
                )

            if self.use_centralized_critic:
                delta_v_state = central_delta_v_state
            else:
                delta_v_state = self._compute_agent_delta_value_state(
                    agent_id,
                    obs,
                    mask_tensor,
                    agent_prev_action_onehot,
                )
                delta_v_states[agent_id] = delta_v_state

            actions[agent_id] = (action, log_prob)
            hidden_state_entry = (p_state, v_state, p_next_state, delta_v_state)
            self.temp_hidden_states[agent_id] = hidden_state_entry
            self.temp_masks[agent_id] = mask
            self.temp_log_probs[agent_id] = log_prob
            
        return actions
    
    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],  # List of length n_envs
        available_actions_batch: Optional[List[Optional[Dict[int, List[int]]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        """
        Get actions for a batch of observations (parallel environments).
        
        Args:
            observations_batch: List of observation dicts, length = n_envs
                               Each dict maps agent_id to observation tensor
            available_actions_batch: List of available actions dicts or None
        
        Returns:
            List of action dicts, length = n_envs
            Each dict maps agent_id to (action, log_prob) tuple
        """
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        # Store hidden states and masks for later use in store_transition
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]
        
        # Centralized critic: compute central values in batch
        if self.use_centralized_critic:
            curr_states_value = self._compute_central_value_states_batch(observations_batch)
        if self.use_centralized_critic:
            curr_states_delta_value = self._compute_central_delta_value_states_batch(observations_batch)
        else:
            curr_states_delta_value_by_agent = {}
        
        # Process each agent
        for agent_id in range(self.num_agents):
            # Stack observations from all environments
            obs_stacked = torch.stack([
                observations_batch[env_idx][agent_id]
                for env_idx in range(n_envs)
            ]).to(self.device)  # (n_envs, obs_dim)
            
            # Create masks
            masks_stacked = torch.tensor([
                [1.0 if not self.last_dones_batch[agent_id][env_idx] else 0.0]
                for env_idx in range(n_envs)
            ], dtype=torch.float32, device=self.device)  # (n_envs, 1)
            
            # Stack available actions if provided
            avail_actions_stacked = None
            if available_actions_batch is not None and available_actions_batch[0] is not None:
                avail_actions_stacked = torch.stack([
                    torch.tensor(available_actions_batch[env_idx][agent_id], dtype=torch.float32, device=self.device)
                    for env_idx in range(n_envs)
                ])  # (n_envs, action_dim)
            
            # Cache a_{t-1} before get_action_batch updates it to a_t
            agent_prev_actions_onehot = self.agents[agent_id].prev_actions_batch.clone()
            
            # Get actions in batch
            actions, log_probs, p_states, p_next_states = self.agents[agent_id].get_action_batch(
                obs_stacked,
                masks_stacked,
                avail_actions_stacked
            )

            if not self.use_centralized_critic:
                # Get values in batch (IPPO style)
                _, v_states = self.agents[agent_id].get_value_batch(
                    obs_stacked,
                    masks_stacked,
                    prev_actions_onehot=agent_prev_actions_onehot
                )
                curr_states_delta_value_by_agent[agent_id] = self._compute_agent_delta_value_states_batch(
                    agent_id,
                    obs_stacked,
                    masks_stacked,
                    agent_prev_actions_onehot,
                )

            # Distribute results to each environment
            for env_idx in range(n_envs):
                actions_batch[env_idx][agent_id] = (
                    actions[env_idx].item(),
                    log_probs[env_idx].item()
                )
                if self.use_centralized_critic:
                    v_state_for_env = curr_states_value[env_idx]
                else:
                    v_state_for_env = v_states[env_idx].clone()
                hidden_state_entry = (
                    p_states[env_idx].clone(),
                    v_state_for_env,
                    p_next_states[env_idx].clone(),
                )
                if self.use_centralized_critic:
                    delta_v_state_for_env = curr_states_delta_value[env_idx].clone()
                else:
                    delta_v_state_for_env = curr_states_delta_value_by_agent[agent_id][env_idx].clone()
                hidden_state_entry = hidden_state_entry + (delta_v_state_for_env,)
                self.temp_hidden_states_batch[env_idx][agent_id] = hidden_state_entry
                self.temp_masks_batch[env_idx][agent_id] = masks_stacked[env_idx, 0].item()
        
        return actions_batch

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
        available_actions: Optional[Dict[int, List[int]]] = None, # [ADDED]
        env_id: int = 0
    ):
        # Update last_dones for next step's mask
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
            env_id=env_id if self.use_episode_buffer else None,
        )
        self.buffer_manager.store_transition(**kwargs)
    
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
        """
        Store transition from a specific parallel environment.
        
        Args:
            env_idx: Index of the environment (0 to n_envs-1)
            ... (same as store_transition)
        """
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
        
        # Update done status for this environment
        for agent_id in range(self.num_agents):
            self.last_dones_batch[agent_id][env_idx] = dones[agent_id]
            
            # Reset RNN states if episode is done
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                    self.recurrent_N, self.hidden_dim, device=self.device
                )
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                    self.action_dim, device=self.device
                )
                if not self.use_centralized_critic:
                    self.agents[agent_id].rnn_states_value_batch[env_idx] = torch.zeros(
                        self.recurrent_N, self.hidden_dim, device=self.device
                    )
        
        # Central Value RNN Reset (if global done)
        if self.use_centralized_critic and episode_done:
            self.rnn_states_value_batch[env_idx] = torch.zeros(
                self.recurrent_N, self.hidden_dim, device=self.device
            )
            self.rnn_states_delta_value_batch[env_idx] = torch.zeros(
                self.recurrent_N, self.hidden_dim, device=self.device
            )
        elif not self.use_centralized_critic:
            for agent_id in range(self.num_agents):
                if dones[agent_id]:
                    self.rnn_states_delta_value_batch[agent_id][env_idx] = torch.zeros(
                        self.recurrent_N, self.hidden_dim, device=self.device
                    )

    def finish_episode(self, env_id: int = 0):
        """Finish current episode if using EpisodeBuffer."""
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)
    
    def compute_causal_rewards(
        self,
        w_t_batch: torch.Tensor,
        z_t_batch: torch.Tensor,
        full_obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards_batch: torch.Tensor,
        hidden_states_batch: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        dones_batch: Optional[torch.Tensor] = None,
        baseline_type: str = 'weighted',
        available_actions_batch: Optional[Dict[int, torch.Tensor]] = None, # [ADDED]
        prev_actions_batch: Optional[torch.Tensor] = None,  # (B, N) previous actions for RNN input
        debug: bool = False
    ) -> torch.Tensor:
        """
        Computes the pairwise Causal Effect Matrix C.
        C[b, i, j] = Effect of Agent i on Agent j in batch b.
        
        Returns:
            C_matrix: [B, N, N] tensor
        """
        if self.get_current_causal_lambda() == 0.0:
            return torch.zeros(actions_batch.size(0), self.num_agents, self.num_agents, device=self.device)

        batch_size = actions_batch.size(0)
        num_agents = self.num_agents
        action_dim = self.action_dim
        
        # Prepare Output Matrix
        C_matrix = torch.zeros(batch_size, num_agents, num_agents, device=self.device)
        
        infer_chunk_size = 1024 
        
        with torch.no_grad():
            for i in range(0, batch_size, infer_chunk_size):
                end = min(i + infer_chunk_size, batch_size)
                
                act_b = actions_batch[i:end]
                w_b = w_t_batch[i:end]
                # z_b = z_t_batch[i:end] # Unused now
                r_b = rewards_batch[i:end]
                obs_b = full_obs_batch[i:end]
                curr_bs = act_b.size(0)
                
                # --- Vectorized Logic ---
                actions_flat = act_b.view(-1)
                w_t_flat = w_b.view(-1, w_b.size(-1))
                
                # r_pred_flat = self.outcome_bridge_net(actions_flat, w_t_flat) 
                reward_dim = r_b.size(-1)
                rewards_flat = r_b.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, reward_dim)
                
                actions_range = torch.arange(action_dim, device=self.device).repeat(curr_bs * num_agents)
                w_t_expanded = w_t_flat.repeat_interleave(action_dim, dim=0)

                if prev_actions_batch is not None:
                    prev_actions_onehot_b = F.one_hot(
                        prev_actions_batch[i:end].long(),
                        self.action_dim,
                    ).float()
                    expanded_prev_actions = prev_actions_onehot_b.unsqueeze(1).expand(-1, num_agents, -1, -1)
                    action_mask = ~torch.eye(num_agents, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(
                        curr_bs, -1, -1, self.action_dim
                    )
                    prev_actions_others_inf = expanded_prev_actions[action_mask].view(
                        curr_bs * num_agents, -1
                    )
                else:
                    prev_actions_others_inf = torch.zeros(
                        curr_bs * num_agents,
                        self.bridge_prev_action_dim,
                        device=self.device,
                    )

                # X^i_t = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1})
                if self.use_recurrent_policy:
                    h_prev_list_inf = []
                    for k in range(num_agents):
                        if hidden_states_batch is not None and k in hidden_states_batch:
                            h_prev_list_inf.append(hidden_states_batch[k][0][i:end].view(curr_bs, -1))
                        else:
                            h_prev_list_inf.append(torch.zeros(curr_bs, self.recurrent_N * self.hidden_dim, device=self.device))
                    H_prev_all_inf = torch.stack(h_prev_list_inf, dim=1)  # (curr_bs, N, RecN*H)
                    H_prev_all_inf_expanded = H_prev_all_inf.unsqueeze(1).expand(-1, num_agents, -1, -1)
                    hid_mask = ~torch.eye(num_agents, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(
                        curr_bs, -1, -1, self.recurrent_N * self.hidden_dim
                    )
                    H_prev_others_inf = H_prev_all_inf_expanded[hid_mask].view(curr_bs * num_agents, -1)
                    obs_flat = obs_b.reshape(curr_bs * num_agents, self.policy_obs_dim)
                    x_flat = torch.cat([obs_flat, H_prev_others_inf, prev_actions_others_inf], dim=1)
                else:
                    x_flat = torch.cat(
                        [obs_b.reshape(curr_bs * num_agents, self.policy_obs_dim), prev_actions_others_inf],
                        dim=1,
                    )

                x_expanded = x_flat.repeat_interleave(action_dim, dim=0)
                
                reg_estimates_cf = self.outcome_bridge_net(actions_range, w_t_expanded, x=x_expanded) 
                # reg_estimates_cf = self.naive_predictor(actions_range, w_t_expanded, x_expanded) # Observational-regressor ablation for the interventional-mean estimate.
                r_hat_all = reg_estimates_cf.view(curr_bs * num_agents, action_dim, reward_dim) # no q

                # collect r_hat by factual actions
                r_hat_factual = r_hat_all[torch.arange(r_hat_all.size(0)), actions_flat]

                probs_list = []
                precomputed_states = {}
                precomputed_masks = {}
                for k in range(num_agents):
                    # Actor RNN state (fallback to zeros if not provided)
                    if hidden_states_batch is not None and k in hidden_states_batch:
                        p_state_b = hidden_states_batch[k][0][i:end]  # (curr_bs, recurrent_N, hidden_dim)
                        if p_state_b.device != self.device:
                            p_state_b = p_state_b.to(self.device)
                    else:
                        p_state_b = torch.zeros(curr_bs, self.recurrent_N, self.hidden_dim, device=self.device)

                    # RNN mask is not needed here because we feed a precomputed hidden state per timestep.
                    masks_b = torch.ones(curr_bs, 1, device=self.device)

                    precomputed_states[k] = p_state_b
                    precomputed_masks[k] = masks_b

                for k in range(num_agents):
                    avail_actions_b = None
                    # available_actions_batch: masking for distinguish dead/alive agents
                    # dead agents only possible no-op aciton [1, 0, ..., 0] and alive ones cannot do no-op action [0, 1, ..., 1]

                    if available_actions_batch is not None and k in available_actions_batch:
                        avail_actions_b = available_actions_batch[k][i:end]

                    p_state_b = precomputed_states[k]
                    masks_b = precomputed_masks[k]
                    # Construct prev_actions one-hot for agent k
                    prev_actions_k = None
                    if prev_actions_batch is not None:
                        prev_act_k = prev_actions_batch[i:end, k]  # (curr_bs,)
                        prev_actions_k = F.one_hot(prev_act_k.long(), self.action_dim).float()
                    logits, _ = self.agents[k].policy_net(obs_b[:, k, :], p_state_b, masks_b, available_actions=avail_actions_b, prev_actions=prev_actions_k)
                    probs = torch.softmax(logits, dim=-1)
                    probs_list.append(probs)
                    
                action_probs_flat = torch.stack(probs_list, dim=1).view(-1, action_dim)

                # Available action mask (B*N, action_dim) for baseline weighting / dead handling
                if available_actions_batch is not None:
                    avail_mask_list = []
                    for kk in range(num_agents):
                        if kk in available_actions_batch:
                            avail_mask_list.append(available_actions_batch[kk][i:end])
                        else:
                            avail_mask_list.append(torch.ones(curr_bs, action_dim, device=self.device))
                    avail_mask_flat = torch.stack(avail_mask_list, dim=1).view(-1, action_dim)
                else:
                    avail_mask_flat = None

                # baseline: exclude factual action
                factual_mask = torch.zeros_like(action_probs_flat).scatter_(1, actions_flat.unsqueeze(1), 1.0)
                non_factual_mask = 1.0 - factual_mask
                
                if baseline_type == 'weighted':
                    probs_counterfactual = action_probs_flat * non_factual_mask
                    if avail_mask_flat is not None:
                        probs_counterfactual = probs_counterfactual * avail_mask_flat

                    # NOTE: renormalization
                    renormalized_probs = probs_counterfactual / (probs_counterfactual.sum(dim=1, keepdim=True) + 1e-10) 
                    baseline = (renormalized_probs.unsqueeze(2) * r_hat_all).sum(dim=1)

                    # NOTE: non-renorm ver. if want to include factual action, active the below line
                    # baseline = (action_probs_flat.unsqueeze(2) * r_hat_all).sum(dim=1)

                elif baseline_type == 'max':
                    valid_non_factual = non_factual_mask.bool()
                    if avail_mask_flat is not None:
                        valid_non_factual = valid_non_factual & avail_mask_flat.bool()

                    masked_r_hat = r_hat_all.masked_fill(~valid_non_factual.unsqueeze(2), float('-inf'))
                    baseline = masked_r_hat.max(dim=1).values

                    no_valid_cf = ~valid_non_factual.any(dim=1, keepdim=True)
                    baseline = torch.where(no_valid_cf, r_hat_factual, baseline)

                elif baseline_type == 'averaged':
                    # Factual action을 제외한 나머지 action들의 단순 평균 계산 (가능한 action들만 평균)
                    if avail_mask_flat is not None:
                        valid_non_factual = non_factual_mask * avail_mask_flat
                        denom = valid_non_factual.sum(dim=1, keepdim=True).clamp_min(1.0)
                        baseline = (r_hat_all * valid_non_factual.unsqueeze(2)).sum(dim=1) / denom
                    else:
                        baseline = (r_hat_all * non_factual_mask.unsqueeze(2)).sum(dim=1) / (action_dim - 1)

                else:
                    raise ValueError(f"Invalid baseline type: {baseline_type}")

                if avail_mask_flat is not None:
                    r_hat_all = r_hat_all * avail_mask_flat.unsqueeze(2)  # 불가능한 action의 예측값 제거

                r_factual = rewards_flat
                raw_effects = r_factual - baseline
                
                if self.global_reward:
                    actor_team_effects = raw_effects.view(curr_bs, num_agents, 1)
                else:
                    C_chunk = raw_effects.view(curr_bs, num_agents, num_agents)
                
                # Dead agent masking: zero out causal effects from/to dead agents
                alive = None
                if available_actions_batch is not None:
                    alive_list = []
                    for kk in range(num_agents):
                        if kk in available_actions_batch:
                            # dead: only no-op available => no non-noop actions available
                            alive_list.append((available_actions_batch[kk][i:end][:, 1:].sum(dim=-1) > 0))
                        else:
                            alive_list.append(torch.ones(curr_bs, dtype=torch.bool, device=self.device))
                    alive = torch.stack(alive_list, dim=1).float()  # (curr_bs, num_agents)
                if self.global_reward:
                    if alive is None:
                        C_chunk = actor_team_effects.expand(-1, -1, num_agents) / num_agents
                    else:
                        alive_counts = alive.sum(dim=1, keepdim=True).clamp_min(1.0)
                        recipient_weights = alive.unsqueeze(1) / alive_counts.unsqueeze(1)
                        C_chunk = actor_team_effects.expand(-1, -1, num_agents) * recipient_weights
                        C_chunk = C_chunk * alive.unsqueeze(2)
                elif alive is not None:
                    C_chunk = C_chunk * alive.unsqueeze(2) * alive.unsqueeze(1)
                
                C_matrix[i:end] = C_chunk
            
        return C_matrix

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)

        full_obs_batch, actions_batch, available_actions_batch, init_hidden_states, masks_batch = \
            self._prepare_update_batch_tensors(batch_data)
        T, N, D = full_obs_batch.shape
        
        # Extract pre-computed advantages and returns
        agent_returns = {}
        agent_advantages = {}
        agent_env_advantages = {}  # [Direction 2] env-only advantages
        agent_pci_advantages = {}  # [Direction 2] pci-only advantages
        agent_delta_returns = {}

        causal_lambda = self.get_current_causal_lambda()  # [Direction 2]

        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue

            # Extract pre-computed values
            returns_list = [t['returns'][agent_id] for t in batch_data]
            advantages_list = [t['advantages'][agent_id] for t in batch_data]

            agent_returns[agent_id] = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)

            # [Direction 2] Extract env and pci advantages separately
            agent_env_advantages[agent_id] = torch.tensor(
                [t['env_advantages'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            agent_pci_advantages[agent_id] = torch.tensor(
                [t['pci_advantages'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            agent_delta_returns[agent_id] = torch.tensor(
                [t['delta_returns'][agent_id] for t in batch_data],
                dtype=torch.float32,
                device=self.device,
            )

        # Log delta (causal intrinsic rewards) by scenario for StagHunt
        for i in range(self.num_agents):
            for transition in batch_data:
                self._log_transition_level_pci_metrics(
                    agent_losses,
                    transition,
                    i,
                    self.get_current_causal_lambda(),
                )

        # PPO Update
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = full_obs_batch[:, agent_id, :]
            my_actions = actions_batch[:, agent_id]
            
            avail_actions_all = None
            if agent_id in available_actions_batch:
                avail_actions_all = available_actions_batch[agent_id] # (T, ActionDim)
            prepared = self._prepare_agent_training_batch(
                batch_data=batch_data,
                agent_id=agent_id,
                agent_obs=agent_obs,
                my_actions=my_actions,
                agent_advantage=agent_advantages[agent_id], # [Direction 1] combined (env_norm + λ*pci_norm); uncomment to revert
                agent_return=agent_returns[agent_id],
                agent_delta_return=agent_delta_returns[agent_id],
                masks_batch=masks_batch,
                init_hidden_states=init_hidden_states,
                available_actions_all=avail_actions_all,
            )
            num_chunks = prepared['num_chunks']
            rnn_obs_batch = prepared['rnn_obs_batch']
            rnn_masks_batch = prepared['rnn_masks_batch']
            p_h0_batch = prepared['p_h0_batch']
            v_h0_batch = prepared['v_h0_batch']
            my_actions_batch = prepared['my_actions_batch']
            adv_batch = prepared['adv_batch']
            ret_batch = prepared['ret_batch']
            delta_ret_batch = prepared.get('delta_ret_batch')
            active_masks_batch = prepared['active_masks_batch']
            prev_actions_batch = prepared['prev_actions_batch']
            prev_actions_batch_merged = prepared['prev_actions_batch_merged']
            avail_actions_batch_merged = prepared['avail_actions_batch_merged']
            old_log_probs = prepared['old_log_probs']
            delta_v_h0_batch = prepared.get('delta_v_h0_batch')

            for epoch in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(
                    rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch,
                    available_actions=avail_actions_batch_merged, prev_actions=prev_actions_batch
                )
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                # Masked Entropy Loss
                entropy_loss = -entropy # entropy is (T)
                entropy_loss = (entropy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
                
                if not self.use_centralized_critic:
                    # IPPO-style per-agent value update
                    new_vals, _ = self.agents[agent_id].value_net(
                        rnn_obs_batch, v_h0_batch, rnn_masks_batch, 
                        prev_actions=prev_actions_batch
                    )
                    
                    # Masked Value Loss
                    val_error = (new_vals.squeeze(1) - ret_batch) ** 2
                    val_loss = (val_error * active_masks_batch).sum()
                    
                    if not self.use_value_parameter_sharing:
                        # Individual Critics: Update immediately per agent
                        mean_val_loss = val_loss / (active_masks_batch.sum() + 1e-8)
                        self.agents[agent_id].value_optimizer.zero_grad()
                        (self.value_coef * mean_val_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].value_net.parameters(), self.max_grad_norm)
                        self.agents[agent_id].value_optimizer.step()
                        agent_losses[f'agent_{agent_id}_value_loss'] += mean_val_loss.item()
                    else:
                        # Shared Critic: Accumulate loss to update once per epoch outside the agent loop
                        if not hasattr(self, '_temp_epoch_value_loss'):
                            self._temp_epoch_value_loss = 0.0
                            self._temp_epoch_mask_sum = 0.0
                        self._temp_epoch_value_loss += val_loss
                        self._temp_epoch_mask_sum += active_masks_batch.sum()

                if not self.use_centralized_critic:
                    delta_vals, _ = self.delta_value_net(
                        rnn_obs_batch,
                        delta_v_h0_batch,
                        rnn_masks_batch,
                        prev_actions=prev_actions_batch,
                    )
                    delta_error = (delta_vals.squeeze(1).squeeze(-1) - delta_ret_batch) ** 2
                    delta_loss = (delta_error * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                    self.delta_value_optimizer.zero_grad()
                    (self.value_coef * delta_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.delta_value_net.parameters(), self.max_grad_norm)
                    self.delta_value_optimizer.step()
                    agent_losses[f'agent_{agent_id}_delta_critic_loss'] += delta_loss.item()
            
            # === Shared Value Update (only if parameter sharing AND not centralized) ===
            if not self.use_centralized_critic and self.use_value_parameter_sharing:
                if hasattr(self, '_temp_epoch_mask_sum') and self._temp_epoch_mask_sum > 0:
                    mean_val_loss = self._temp_epoch_value_loss / self._temp_epoch_mask_sum
                    self.shared_value_optimizer.zero_grad()
                    (self.value_coef * mean_val_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.shared_value_net.parameters(), self.max_grad_norm)
                    self.shared_value_optimizer.step()
                    
                    agent_losses['value_loss'] += mean_val_loss.item()
                    
                    # Reset temporary accumulators for the next epoch/batch
                    del self._temp_epoch_value_loss
                    del self._temp_epoch_mask_sum

        # === Central Value Update (only if centralized critic) ===
        if self.use_centralized_critic:
            central_value_batch = self._prepare_centralized_value_update_batch(batch_data)
            global_obs_batch = central_value_batch['global_obs_batch']
            global_masks_batch = central_value_batch['global_masks_batch']
            active_masks_batch = central_value_batch['active_masks_batch']
            v_h0_batch = central_value_batch['v_h0_batch']
            global_ret_batch = central_value_batch['global_ret_batch']
            global_prev_actions_batch = central_value_batch['global_prev_actions_batch']

            for _ in range(self.ppo_epochs):
                # new_vals: (..., num_agents)
                new_vals, _ = self.value_net(global_obs_batch, v_h0_batch, global_masks_batch, prev_actions=global_prev_actions_batch)
                if self.global_reward:
                    val_error = (new_vals.squeeze(1).squeeze(-1) - global_ret_batch) ** 2
                    val_loss = (val_error * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                else:
                    val_error = (new_vals.squeeze(1) - global_ret_batch) ** 2
                    val_loss = (val_error * active_masks_batch.unsqueeze(1)).sum() / (active_masks_batch.sum() * self.num_agents + 1e-8)
                
                self.value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                agent_losses['value_loss'] += val_loss.item()

        if self.use_centralized_critic:
            central_delta_batch = self._prepare_centralized_delta_value_update_batch(batch_data)
            global_obs_batch = central_delta_batch['global_obs_batch']
            global_masks_batch = central_delta_batch['global_masks_batch']
            active_masks_batch = central_delta_batch['active_masks_batch']
            delta_v_h0_batch = central_delta_batch['delta_v_h0_batch']
            global_delta_ret_batch = central_delta_batch['global_delta_ret_batch']
            global_prev_actions_batch = central_delta_batch['global_prev_actions_batch']

            for _ in range(self.ppo_epochs):
                new_delta_vals, _ = self.delta_value_net(
                    global_obs_batch,
                    delta_v_h0_batch,
                    global_masks_batch,
                    prev_actions=global_prev_actions_batch,
                )
                delta_error = (new_delta_vals.squeeze(1) - global_delta_ret_batch) ** 2
                delta_loss = (delta_error * active_masks_batch.unsqueeze(1)).sum() / (active_masks_batch.sum() * self.num_agents + 1e-8)

                self.delta_value_optimizer.zero_grad()
                (self.value_coef * delta_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.delta_value_net.parameters(), self.max_grad_norm)
                self.delta_value_optimizer.step()

                agent_losses['delta_critic_loss'] += delta_loss.item()

        for k in list(agent_losses.keys()):
            if k == 'value_loss':
                if self.use_centralized_critic or self.use_value_parameter_sharing:
                    # Centralized critic and Shared parameter critic: value_loss is already summed over ppo_epochs
                    agent_losses['value_loss'] /= self.ppo_epochs
                else:
                    total_value_loss = sum([agent_losses.get(f'agent_{i}_value_loss', 0.0) for i in range(self.num_agents)])
                    agent_losses['value_loss'] = total_value_loss / (self.num_agents * self.ppo_epochs)
            elif k.endswith('_sum') or k.endswith('_count'):
                # Keep raw sum/count pairs for cross-batch aggregation in update().
                continue
            else:
                agent_losses[k] /= self.ppo_epochs
        
        return dict(agent_losses)

    def update(self, **kwargs):
        trajectories = []
        
        if self.use_episode_buffer:
            if not self.buffer_manager.has_completed_episodes(): return {}
            trajectories = self.buffer_manager.get_completed_episodes()

        else: # TransitionBuffer need preproccessing to align the data to episodic trajectories.
            if self.buffer_manager.is_main_buffer_empty(): return {}
            transitions = self.buffer_manager.get_main_buffer_transitions()
            
            # Group by env_id
            env_trajectories = defaultdict(list)
            for t in transitions:
                env_id = t['info'].get('env_id', 0)
                env_trajectories[env_id].append(t)
            trajectories = list(env_trajectories.values())
  
        assign_proxy_shuffle_metadata(trajectories)

        if not hasattr(self, '_update_call_count'):
            self._update_call_count = 0
        self._update_call_count += 1
        # _do_causal_debug = (self._update_call_count % 50 == 1)
        current_causal_lambda = self.get_current_causal_lambda()

        # Compute GAE on full trajectories before chunking
        # 2. Compute causal rewards for each trajectory
        for traj in trajectories:
            if len(traj) == 0:
                continue
            (full_obs_batch, actions_batch, 
                rewards_batch, dones_batch, 
                z_t_batch, w_t_all, hidden_states_batch, 
                available_actions_batch, prev_actions_batch_cr) = self._build_causal_reward_inputs(traj)
            N = self.num_agents

            # Compute causal rewards
            with torch.no_grad():
                C_matrix = self.compute_causal_rewards(
                    w_t_all, z_t_batch, full_obs_batch, actions_batch, rewards_batch,
                    hidden_states_batch=hidden_states_batch,
                    dones_batch=dones_batch,
                    baseline_type='weighted', # weighted / averaged / max
                    available_actions_batch=available_actions_batch,
                    prev_actions_batch=prev_actions_batch_cr,
                )
            
                delta = C_matrix.sum(dim=2)
                # delta[torch.abs(delta) < self.clip_delta] = 0.0 # NOTE: clip delta
                
                # Store delta for later advantage-level PCI construction
                for t_idx, t in enumerate(traj):
                    if 'delta' not in t:
                        t['delta'] = {}
                    for agent_id in range(N):
                        t['delta'][agent_id] = delta[t_idx, agent_id].item()

        # 3. Compute raw-return GAE for critic and PCI-augmented advantages for actor
        for traj in trajectories:
            if len(traj) == 0:
                continue

            T = len(traj)
            N = self.num_agents
            team_dones_seq_list = build_team_done_sequence(traj)
            raw_team_rewards = torch.tensor( 
                [self._global_reward_from_dict(t['bold_r_t']) for t in traj],
                dtype=torch.float32,
                device=self.device,
            )
            
            if self.use_centralized_critic:
                # === Centralized critic GAE ===
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

                # Initial hidden state for central value network (agent 0's v_state)
                v_h0 = traj[0]['hidden_states'][0][1].to(self.device, non_blocking=True).unsqueeze(0)

                with torch.no_grad():
                    last_next_obs = concatenate_agent_observations(
                        traj[-1]['next_observations'],
                        self.num_agents,
                        self.obs_dim,
                        self.device,
                    ).to(self.device, non_blocking=True)
                    last_mask = 1.0 - (1.0 if team_dones_seq_list[-1] else 0.0)
                    
                    extended_obs = torch.cat([global_obs_seq.unsqueeze(1),
                                             last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                    extended_masks = torch.cat([global_masks_seq,
                                               torch.tensor([[last_mask]], device=self.device)], dim=0)
                    
                    joint_prev_actions = self._build_joint_prev_actions_sequence(traj) # (T+1, 1, num_agents * action_dim)
                    
                    all_values, _ = self.value_net(extended_obs, v_h0, extended_masks, prev_actions=joint_prev_actions)
                    if self.global_reward:
                        values = all_values[:-1].view(-1)
                        next_val = all_values[-1].item() * last_mask
                        env_advantages, returns = compute_gae(
                            raw_team_rewards.tolist(),
                            values.tolist(),
                            next_val,
                            team_dones_seq_list,
                            self.gamma,
                            self.gae_lambda,
                        )
                        env_advantages_tensor = torch.tensor(env_advantages, 
                                                             dtype=torch.float32, device=self.device)
                    else:
                        values = all_values[:-1].view(-1, self.num_agents)
                        next_val = all_values[-1].view(self.num_agents) * last_mask

                    if len(traj[0]['hidden_states'][0]) > 3:
                        delta_v_h0 = traj[0]['hidden_states'][0][3].to(self.device, non_blocking=True).unsqueeze(0)
                    else:
                        delta_v_h0 = torch.zeros(1, self.recurrent_N, self.hidden_dim, device=self.device)
                    all_delta_values, _ = self.delta_value_net(
                        extended_obs,
                        delta_v_h0,
                        extended_masks,
                        prev_actions=joint_prev_actions,
                    )
                    delta_values = all_delta_values[:-1].view(-1, self.num_agents)
                    next_delta_val = all_delta_values[-1].view(self.num_agents) * last_mask
                
                # Compute per-agent GAE using centralized values and per-agent PCI deltas
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        continue
                    
                    pci_delta_seq = torch.tensor(
                        [t['delta'][agent_id] for t in traj],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    agent_dones_seq = torch.tensor(
                        [t['dones'][agent_id] for t in traj],
                        dtype=torch.float32, device=self.device
                    )

                    if self.global_reward:
                        env_adv_input = env_advantages_tensor
                        returns_to_store = returns
                        done_signal = team_dones_seq_list
                    else:
                        env_advantages, returns_to_store = compute_gae(
                            [t['bold_r_t'][agent_id] for t in traj],
                            values[:, agent_id].tolist(),
                            next_val[agent_id].item(),
                            agent_dones_seq.tolist(),
                            self.gamma,
                            self.gae_lambda,
                        )
                        env_adv_input = torch.tensor(
                            env_advantages,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        done_signal = agent_dones_seq.tolist()

                    pci_advantages, delta_returns = compute_gae(
                        pci_delta_seq.tolist(),
                        delta_values[:, agent_id].tolist(),
                        next_delta_val[agent_id].item(),
                        done_signal,
                        self.gamma,
                        self.gae_lambda,
                    )

                    combined_advantages, env_adv_norm, pci_adv_residual = self._combine_env_and_pci_advantages(
                        env_adv_input,
                        torch.tensor(pci_advantages, dtype=torch.float32, device=self.device),
                        current_causal_lambda,
                    )
                    
                    for t_idx, t in enumerate(traj):
                        self._store_advantage_entry(
                            t,
                            agent_id,
                            combined_advantages[t_idx].item(),
                            returns_to_store[t_idx],
                            env_adv_norm[t_idx].item(),
                            pci_advantages[t_idx],
                            pci_adv_residual[t_idx].item(),
                            delta_returns[t_idx],
                        )
            else:
                # === IPPO-style per-agent GAE ===
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        continue

                    # Optimized: Batch tensor creation and transfer
                    agent_obs_list = [t['bold_o_t'][agent_id] for t in traj]
                    agent_obs_seq = torch.stack(agent_obs_list).to(self.device, non_blocking=True)
                    agent_masks_seq = torch.tensor([t['masks'][agent_id] for t in traj], device=self.device).unsqueeze(1)
                    agent_dones_seq = torch.tensor([t['dones'][agent_id] for t in traj], dtype=torch.float32, device=self.device)
                    pci_delta_seq = torch.tensor(
                        [t['delta'][agent_id] for t in traj],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    # Initial hidden state for value network
                    v_h0 = traj[0]['hidden_states'][agent_id][1]
                    v_h0 = v_h0.to(self.device, non_blocking=True).unsqueeze(0)

                    # Compute values for entire trajectory
                    with torch.no_grad():
                        last_next_obs = traj[-1]['next_observations'][agent_id].to(self.device, non_blocking=True)
                        if self.global_reward:
                            last_mask = 1.0 - (1.0 if team_dones_seq_list[-1] else 0.0)
                        else:
                            last_mask = 1.0 - traj[-1]['dones'][agent_id]

                        # Extended sequence: (T+1, 1, D)
                        extended_obs = torch.cat([agent_obs_seq.unsqueeze(1),
                                                last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                        extended_masks = torch.cat([agent_masks_seq,
                                                    torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)], dim=0)
                        
                        extended_agent_prev_actions = self._build_agent_prev_actions_sequence(traj, agent_id).unsqueeze(1)
                        
                        all_values, _ = self.agents[agent_id].value_net(extended_obs, v_h0, extended_masks, prev_actions=extended_agent_prev_actions)
                        values = all_values[:-1].view(-1) # value of agent_id
                        next_val = all_values[-1].item() * last_mask # next value

                        if len(traj[0]['hidden_states'][agent_id]) > 3:
                            delta_v_h0 = traj[0]['hidden_states'][agent_id][3]
                            delta_v_h0 = delta_v_h0.to(self.device, non_blocking=True).unsqueeze(0)
                        else:
                            delta_v_h0 = torch.zeros(1, self.recurrent_N, self.hidden_dim, device=self.device)
                        all_delta_values, _ = self.delta_value_net(
                            extended_obs,
                            delta_v_h0,
                            extended_masks,
                            prev_actions=extended_agent_prev_actions,
                        )
                        delta_values = all_delta_values[:-1].view(-1)
                        next_delta_val = all_delta_values[-1].item() * last_mask
                        
                        reward_signal = (
                            raw_team_rewards
                            if self.global_reward
                            else torch.tensor(
                                [t['bold_r_t'][agent_id] for t in traj],
                                dtype=torch.float32,
                                device=self.device,
                            )
                        )
                        done_signal = (
                            team_dones_seq_list
                            if self.global_reward
                            else agent_dones_seq.tolist()
                        )
                        env_advantages, returns = compute_gae(
                            reward_signal.tolist(),
                            values.tolist(),
                            next_val,
                            done_signal,
                            self.gamma,
                            self.gae_lambda,
                        )

                        pci_advantages, delta_returns = compute_gae(
                            pci_delta_seq.tolist(),
                            delta_values.tolist(),
                            next_delta_val,
                            done_signal,
                            self.gamma,
                            self.gae_lambda,
                        )

                        combined_advantages, env_adv_norm, pci_adv_residual = self._combine_env_and_pci_advantages(
                            torch.tensor(env_advantages, dtype=torch.float32, device=self.device),
                            torch.tensor(pci_advantages, dtype=torch.float32, device=self.device),
                            current_causal_lambda,
                        )
                        
                        # Store in transitions
                        for t_idx, t in enumerate(traj):
                            self._store_advantage_entry(
                                t,
                                agent_id,
                                combined_advantages[t_idx].item(),
                                returns[t_idx],
                                env_adv_norm[t_idx].item(),
                                pci_advantages[t_idx],
                                pci_adv_residual[t_idx].item(),
                                delta_returns[t_idx],
                            )
        
        # 4. Create chunks from trajectories with Padding
        all_chunks = []
        chunk_len = self.rnn_chunk_length
        
        for traj in trajectories:
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
                # We copy the last transition but set active_mask to 0
                padding_len = chunk_len - remainder
                last_valid_trans = last_chunk[-1]
                
                padding_chunk = []
                for _ in range(padding_len):
                    pad_t = last_valid_trans.copy() 
                    pad_t['active_mask'] = 0.0
                    padding_chunk.append(pad_t)
                
                full_last_chunk = last_chunk + padding_chunk
                all_chunks.append(full_last_chunk)

        
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
            
        # Clear buffers
        if self.use_episode_buffer:
            self.buffer_manager.clear_completed_episodes()
        else:
            self.buffer_manager.clear_buffer()
            
        total_losses = defaultdict(float)
        
        for batch_data in batches:
            if not batch_data: continue
            
            if current_causal_lambda != 0.0:
                causal_losses = self.update_causal_networks(batch_data)
            else:
                causal_losses = {'outcome_bridge_loss': 0.0}
            
            agent_and_value_losses = self.update_agents(batch_data)

            for loss_name, loss_value in causal_losses.items():
                total_losses[loss_name] += loss_value
            for loss_name, loss_value in agent_and_value_losses.items():
                total_losses[loss_name] += loss_value

        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        
        num = len([b for b in batches if b])
        if num == 0: return {}
        final_losses = {}
        handled_keys = set()
        for k, v in total_losses.items():
            if k in handled_keys:
                continue
            if k.endswith('_sum'):
                base_key = k[:-4]
                count_key = base_key + '_count'
                if count_key in total_losses:
                    total_count = total_losses.get(count_key, 0.0)
                    mean_key = base_key + '_mean'
                    if total_count > 0:
                        final_losses[mean_key] = v / total_count
                        final_losses[count_key] = total_count
                    else:
                        final_losses[mean_key] = 0.0
                        final_losses[count_key] = 0.0
                    handled_keys.add(k)
                    handled_keys.add(count_key)
                    continue
            if k.endswith('_count') and k in handled_keys:
                continue
            # Other losses: average across batches
            if num > 0:
                final_losses[k] = v / num
            else:
                final_losses[k] = v
        final_losses['causal_lambda'] = current_causal_lambda
        
        return final_losses

    def causal_intrinsic_reward(
        self, 
        agent_id: int, 
        w_t: torch.Tensor, 
        z_t: torch.Tensor, 
        o_t: torch.Tensor, 
        current_actions: Dict[int, int], 
        current_rewards: Dict[int, float],
        hidden_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[float, float]:
        """
        Wrapper for compute_causal_rewards to be used by evaluation scripts.
        Returns:
            delta: The calculated intrinsic reward (causal effect sum).
            dummy: A placeholder return value.
        """
        with torch.no_grad():
            # Actions to tensor
            actions_list = [current_actions[i] for i in range(self.num_agents)]
            actions_batch = torch.tensor([actions_list], dtype=torch.long, device=self.device) # (1, N)
            
            # Rewards to tensor
            if self.global_reward:
                rewards_batch = torch.tensor(
                    [[self._global_reward_from_dict(current_rewards)]],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                rewards_list = [current_rewards[i] for i in range(self.num_agents)]
                rewards_batch = torch.tensor([rewards_list], dtype=torch.float32, device=self.device) # (1, N)
            
            # Reconstruct full observations batch (1, N, D)
            # Assuming w_t is concatenation of N observations of obs_dim
            full_obs_batch = w_t.view(1, self.num_agents, -1)
            B, N, D = full_obs_batch.shape

            # Prepare w_t_batch (1, N, (N-1)*D) - Confounders (other agents' observations)
            expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
            w_t_batch = expanded_obs[mask].view(B, N, (N-1)*D)

            # Z input is currently unused inside compute_causal_rewards, but we preserve
            # the wrapper path so callers can keep passing the active proxy definition.
            z_t_batch = z_t.unsqueeze(0)
            # Wrapper for hidden_states
            hidden_states_batch = None
            if hidden_states is not None:
                hidden_states_batch = {}
                for k, v in hidden_states.items():
                    # v is (h, c) or (h, v_h). h is (RecN, H). Need (1, RecN, H)
                    h_unsqueezed = v[0].unsqueeze(0)
                    c_unsqueezed = v[1].unsqueeze(0)
                    hidden_states_batch[k] = (h_unsqueezed, c_unsqueezed)

            C_matrix = self.compute_causal_rewards(
                w_t_batch, z_t_batch, full_obs_batch, actions_batch, rewards_batch, 
                hidden_states_batch=hidden_states_batch,
                prev_actions_batch=torch.tensor(
                    [[int(torch.argmax(self.agents[i].prev_action).item()) for i in range(N)]],
                    dtype=torch.long,
                    device=self.device,
                ),
                baseline_type='weighted'
            )
            
            # Calculate delta for the specific agent (sum of effects this agent has on others)
            # C_matrix[0, agent_id, :] contains effects of agent_id on others
            delta = C_matrix[0, agent_id].sum().item()
            
            return delta, 0.0

    def _compute_central_value_state(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        episode_done = bool(all(self.last_dones.values())) if len(self.last_dones) > 0 else False
        mask0 = 1.0 if not episode_done else 0.0
        mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
        global_obs = concatenate_agent_observations(
            observations,
            self.num_agents,
            self.obs_dim,
            self.device,
        )
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            agent_prev_actions = [self.agents[i].prev_action for i in range(self.num_agents)]
            global_prev_action_onehot = torch.cat(agent_prev_actions, dim=-1).unsqueeze(0)
            _, next_rnn_state = self.value_net(
                global_obs.unsqueeze(0),
                curr_rnn_state,
                mask_tensor0.unsqueeze(0),
                prev_actions=global_prev_action_onehot
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
            return curr_rnn_state.squeeze(0).clone()

    def _compute_central_value_states_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        n_envs = len(observations_batch)
        global_obs_list = [
            concatenate_agent_observations(
                observations_batch[env_idx],
                self.num_agents,
                self.obs_dim,
                self.device,
            )
            for env_idx in range(n_envs)
        ]
        global_obs_batch_tensor = torch.stack(global_obs_list).to(self.device)
        global_prev_actions_list = [self.agents[i].prev_actions_batch for i in range(self.num_agents)]
        global_prev_action_onehot_batch = torch.cat(global_prev_actions_list, dim=-1)
        central_masks = torch.tensor([
            [
                1.0
                if not all(self.last_dones_batch[agent_id][env_idx] for agent_id in range(self.num_agents))
                else 0.0
            ]
            for env_idx in range(n_envs)
        ], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            curr_states_value = self.rnn_states_value_batch.clone()
            _, next_states_value = self.value_net(
                global_obs_batch_tensor,
                self.rnn_states_value_batch,
                central_masks,
                prev_actions=global_prev_action_onehot_batch
            )
            self.rnn_states_value_batch = next_states_value
            return curr_states_value

    def _compute_central_delta_value_state(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        episode_done = bool(all(self.last_dones.values())) if len(self.last_dones) > 0 else False
        mask0 = 1.0 if not episode_done else 0.0
        mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
        global_obs = concatenate_agent_observations(
            observations,
            self.num_agents,
            self.obs_dim,
            self.device,
        )
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_delta_value.unsqueeze(0)
            agent_prev_actions = [self.agents[i].prev_action for i in range(self.num_agents)]
            global_prev_action_onehot = torch.cat(agent_prev_actions, dim=-1).unsqueeze(0)
            _, next_rnn_state = self.delta_value_net(
                global_obs.unsqueeze(0),
                curr_rnn_state,
                mask_tensor0.unsqueeze(0),
                prev_actions=global_prev_action_onehot,
            )
            self.rnn_state_delta_value = next_rnn_state.squeeze(0)
            return curr_rnn_state.squeeze(0).clone()

    def _compute_agent_delta_value_state(
        self,
        agent_id: int,
        obs: torch.Tensor,
        mask_tensor: torch.Tensor,
        prev_action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            curr_rnn_state = self.rnn_states_delta_value[agent_id].unsqueeze(0)
            _, next_rnn_state = self.delta_value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                mask_tensor.unsqueeze(0),
                prev_actions=prev_action_onehot,
            )
            self.rnn_states_delta_value[agent_id] = next_rnn_state.squeeze(0)
            return curr_rnn_state.squeeze(0).clone()

    def _compute_central_delta_value_states_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        n_envs = len(observations_batch)
        global_obs_list = [
            concatenate_agent_observations(
                observations_batch[env_idx],
                self.num_agents,
                self.obs_dim,
                self.device,
            )
            for env_idx in range(n_envs)
        ]
        global_obs_batch_tensor = torch.stack(global_obs_list).to(self.device)
        global_prev_actions_list = [self.agents[i].prev_actions_batch for i in range(self.num_agents)]
        global_prev_action_onehot_batch = torch.cat(global_prev_actions_list, dim=-1)
        central_masks = torch.tensor([
            [
                1.0
                if not all(self.last_dones_batch[agent_id][env_idx] for agent_id in range(self.num_agents))
                else 0.0
            ]
            for env_idx in range(n_envs)
        ], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            curr_states_delta_value = self.rnn_states_delta_value_batch.clone()
            _, next_states_delta_value = self.delta_value_net(
                global_obs_batch_tensor,
                self.rnn_states_delta_value_batch,
                central_masks,
                prev_actions=global_prev_action_onehot_batch,
            )
            self.rnn_states_delta_value_batch = next_states_delta_value
            return curr_states_delta_value

    def _compute_agent_delta_value_states_batch(
        self,
        agent_id: int,
        obs_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        prev_actions_onehot: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            curr_states_delta_value = self.rnn_states_delta_value_batch[agent_id].clone()
            _, next_states_delta_value = self.delta_value_net(
                obs_batch,
                self.rnn_states_delta_value_batch[agent_id],
                masks_batch,
                prev_actions=prev_actions_onehot,
            )
            self.rnn_states_delta_value_batch[agent_id] = next_states_delta_value
            return curr_states_delta_value

    def _prepare_centralized_value_update_batch(self, batch_data: List[Dict]):
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
        v_states_all_central = torch.stack([
            t['hidden_states'][0][1] for t in batch_data
        ]).to(self.device)
        if self.global_reward:
            all_returns = torch.tensor(
                [t['returns'][0] for t in batch_data],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            all_returns = torch.tensor(
                [[t['returns'][i] for i in range(self.num_agents)] for t in batch_data],
                dtype=torch.float32,
                device=self.device,
            )

        T_total = global_obs_seq.size(0)
        chunk_len = self.rnn_chunk_length
        num_chunks = T_total // chunk_len

        if num_chunks > 0:
            global_obs_batch = chunk_sequence(global_obs_seq, chunk_len)
            global_masks_batch = chunk_sequence(global_masks_seq, chunk_len)

            if 'active_mask' in batch_data[0]:
                active_masks = torch.tensor(
                    [t['active_mask'] for t in batch_data],
                    device=self.device
                ).unsqueeze(1)
                active_masks_batch = chunk_sequence(active_masks, chunk_len).reshape(-1)
            else:
                active_masks_batch = torch.ones(num_chunks * chunk_len, device=self.device)

            chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
            v_h0_batch = v_states_all_central[chunk_indices]
            if self.global_reward:
                global_ret_batch = chunk_sequence(all_returns, chunk_len).reshape(-1)
            else:
                global_ret_batch = chunk_sequence(all_returns, chunk_len).reshape(-1, self.num_agents)
        else:
            global_obs_batch = global_obs_seq.unsqueeze(1)
            global_masks_batch = global_masks_seq
            active_masks_batch = torch.ones(T_total, device=self.device)
            v_h0_batch = v_states_all_central[0].unsqueeze(0)
            global_ret_batch = all_returns

        joint_prev_actions_batch_raw = torch.cat([
            F.one_hot(
                torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
                    dtype=torch.long,
                    device=self.device
                ),
                self.action_dim
            ).float()
            for agent_id in range(self.num_agents)
        ], dim=-1)

        if num_chunks > 0:
            global_prev_actions_batch = chunk_sequence(joint_prev_actions_batch_raw, chunk_len)
        else:
            global_prev_actions_batch = joint_prev_actions_batch_raw.unsqueeze(1)

        return {
            "global_obs_batch": global_obs_batch,
            "global_masks_batch": global_masks_batch,
            "active_masks_batch": active_masks_batch,
            "v_h0_batch": v_h0_batch,
            "global_ret_batch": global_ret_batch,
            "global_prev_actions_batch": global_prev_actions_batch,
        }

    def _prepare_centralized_delta_value_update_batch(self, batch_data: List[Dict]):
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
            device=self.device,
        ).unsqueeze(1)
        delta_states_all_central = torch.stack([
            (
                t['hidden_states'][0][3]
                if len(t['hidden_states'][0]) > 3
                else torch.zeros(self.recurrent_N, self.hidden_dim)
            )
            for t in batch_data
        ]).to(self.device)
        all_delta_returns = torch.tensor(
            [[t['delta_returns'][i] for i in range(self.num_agents)] for t in batch_data],
            dtype=torch.float32,
            device=self.device,
        )

        T_total = global_obs_seq.size(0)
        chunk_len = self.rnn_chunk_length
        num_chunks = T_total // chunk_len

        if num_chunks > 0:
            global_obs_batch = chunk_sequence(global_obs_seq, chunk_len)
            global_masks_batch = chunk_sequence(global_masks_seq, chunk_len)

            if 'active_mask' in batch_data[0]:
                active_masks = torch.tensor(
                    [t['active_mask'] for t in batch_data],
                    device=self.device,
                ).unsqueeze(1)
                active_masks_batch = chunk_sequence(active_masks, chunk_len).reshape(-1)
            else:
                active_masks_batch = torch.ones(num_chunks * chunk_len, device=self.device)

            chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
            delta_v_h0_batch = delta_states_all_central[chunk_indices]
            global_delta_ret_batch = chunk_sequence(all_delta_returns, chunk_len).reshape(-1, self.num_agents)
        else:
            global_obs_batch = global_obs_seq.unsqueeze(1)
            global_masks_batch = global_masks_seq
            active_masks_batch = torch.ones(T_total, device=self.device)
            delta_v_h0_batch = delta_states_all_central[0].unsqueeze(0)
            global_delta_ret_batch = all_delta_returns

        joint_prev_actions_batch_raw = torch.cat([
            F.one_hot(
                torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
                    dtype=torch.long,
                    device=self.device,
                ),
                self.action_dim,
            ).float()
            for agent_id in range(self.num_agents)
        ], dim=-1)

        if num_chunks > 0:
            global_prev_actions_batch = chunk_sequence(joint_prev_actions_batch_raw, chunk_len)
        else:
            global_prev_actions_batch = joint_prev_actions_batch_raw.unsqueeze(1)

        return {
            "global_obs_batch": global_obs_batch,
            "global_masks_batch": global_masks_batch,
            "active_masks_batch": active_masks_batch,
            "delta_v_h0_batch": delta_v_h0_batch,
            "global_delta_ret_batch": global_delta_ret_batch,
            "global_prev_actions_batch": global_prev_actions_batch,
        }

    def _normalize_advantage_signal(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.numel() <= 1:
            return torch.zeros_like(signal)
        std = signal.std(unbiased=False).clamp_min(1e-8)
        return (signal - signal.mean()) / std

    def _combine_env_and_pci_advantages(
        self,
        env_advantages: torch.Tensor,
        pci_signal: torch.Tensor,
        causal_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_norm = self._normalize_advantage_signal(env_advantages)
        pci_norm = self._normalize_advantage_signal(pci_signal)
        combined = env_norm + causal_lambda * pci_norm # NOTE: 1st normalization
        # combined = env_advantages + causal_lambda * pci_signal # test no normalization
        return combined, env_norm, pci_norm
    
    def _ensure_advantage_storage(self, transition: Dict):
        if 'advantages' not in transition:
            transition['advantages'] = {}
            transition['returns'] = {}
            transition['env_advantages'] = {}
            transition['pci_raw_advantages'] = {}
            transition['pci_advantages'] = {}
            transition['delta_returns'] = {}
        else:
            transition.setdefault('delta_returns', {})

    def _store_advantage_entry(
        self,
        transition: Dict,
        agent_id: int,
        combined_advantage: float,
        ret: float,
        env_adv_norm: float,
        pci_raw_advantage: float,
        pci_adv_residual: float,
        delta_return: Optional[float] = None,
    ):
        self._ensure_advantage_storage(transition)
        transition['advantages'][agent_id] = combined_advantage
        transition['returns'][agent_id] = ret
        transition['env_advantages'][agent_id] = env_adv_norm
        transition['pci_raw_advantages'][agent_id] = pci_raw_advantage
        transition['pci_advantages'][agent_id] = pci_adv_residual
        if delta_return is not None:
            transition['delta_returns'][agent_id] = delta_return

    def _build_agent_prev_actions_sequence(self, traj: List[Dict], agent_id: int) -> torch.Tensor:
        agent_prev_actions = torch.tensor(
            [t['bold_a_t_minus_1'][agent_id] for t in traj],
            dtype=torch.long,
            device=self.device,
        )
        agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
        last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
        return torch.cat([agent_prev_actions_onehot, last_prev_action_onehot], dim=0)

    def _build_joint_prev_actions_sequence(self, traj: List[Dict]) -> torch.Tensor:
        agent_prev_actions_list = []
        for agent_id in range(self.num_agents):
            extended_agent_prev_actions = self._build_agent_prev_actions_sequence(traj, agent_id)
            agent_prev_actions_list.append(extended_agent_prev_actions)
        return torch.cat(agent_prev_actions_list, dim=-1).unsqueeze(1)

    def _build_causal_reward_inputs(
        self,
        traj: List[Dict],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        Dict[int, torch.Tensor],
        torch.Tensor,
    ]:
        B = len(traj)
        N = self.num_agents

        full_obs_list = [torch.stack(list(t['bold_o_t'].values())) for t in traj]
        full_obs_batch = torch.stack(full_obs_list).to(self.device, non_blocking=True)
        actions_batch = torch.tensor(
            [[t['bold_a_t'][i] for i in range(N)] for t in traj],
            dtype=torch.long,
            device=self.device,
        )
        rewards_batch = self._build_reward_batch_tensor(traj)
        dones_batch = torch.tensor(
            [[t['dones'][i] for i in range(N)] for t in traj],
            dtype=torch.float32,
            device=self.device,
        )

        # Z^i_t = h^i_{t-1}. This tensor is currently unused
        # by compute_causal_rewards,
        # but we keep it aligned with the active proxy definition for future use.
        hidden_states_batch = {}
        if 'hidden_states' in traj[0] and traj[0]['hidden_states'] is not None:
            for agent_id in range(N):
                if agent_id in traj[0]['hidden_states']:
                    p_states_list = [t['hidden_states'][agent_id][0] for t in traj]
                    v_states_list = [t['hidden_states'][agent_id][1] for t in traj]
                    p_states = torch.stack(p_states_list).to(self.device, non_blocking=True)
                    v_states = torch.stack(v_states_list).to(self.device, non_blocking=True)
                    # h_t (index [2]) stored for bridge network (option 3)
                    if len(traj[0]['hidden_states'][agent_id]) > 2:
                        p_next_states = torch.stack(
                            [t['hidden_states'][agent_id][2] for t in traj]
                        ).to(self.device, non_blocking=True)
                        hidden_states_batch[agent_id] = (p_states, v_states, p_next_states)
                    else:
                        hidden_states_batch[agent_id] = (p_states, v_states)

        if self.use_recurrent_policy:
            z_t_list = []
            for agent_id in range(N):
                if agent_id in hidden_states_batch:
                    z_t_list.append(hidden_states_batch[agent_id][0].view(B, -1))
                else:
                    z_t_list.append(torch.zeros(B, self.recurrent_N * self.hidden_dim, device=self.device))
            z_t_batch = torch.stack(z_t_list, dim=1)
        else:
            z_t_batch = torch.zeros(B, N, 0, device=self.device)

        expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
        mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.obs_dim)
        obs_others_all = expanded_obs[mask].view(B, N, (N - 1) * self.obs_dim)
        w_t_all = obs_others_all

        available_actions_batch = build_available_actions_batch(
            traj,
            N,
            self.device,
        )

        prev_actions_batch_cr = torch.tensor(
            [[t['bold_a_t_minus_1'][i] for i in range(N)] for t in traj],
            dtype=torch.long,
            device=self.device,
        )

        return (
            full_obs_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            z_t_batch,
            w_t_all,
            hidden_states_batch,
            available_actions_batch,
            prev_actions_batch_cr,
        )

    def _log_transition_level_pci_metrics(
        self,
        agent_losses: defaultdict,
        transition: Dict,
        agent_id: int,
        current_causal_lambda: float,
    ):
        if 'delta' not in transition or agent_id not in transition['delta']:
            return
        if transition.get('active_mask', 1.0) <= 0.0:
            return

        if self.env.__class__.__name__ == 'StagHuntEnv':
            scenario = classify_staghunt_delta_scenario(
                transition,
                agent_id,
                self.num_agents,
                self.env_required_agents,
            )
            agent_losses[f'delta_agent_{agent_id}_{scenario}_sum'] += transition['delta'][agent_id]
            agent_losses[f'delta_agent_{agent_id}_{scenario}_count'] += 1
            if 'pci_raw_advantages' in transition and agent_id in transition['pci_raw_advantages']:
                pci_raw = transition['pci_raw_advantages'][agent_id]
                agent_losses[f'pci_raw_adv_agent_{agent_id}_{scenario}_sum'] += pci_raw
                agent_losses[f'pci_raw_adv_agent_{agent_id}_{scenario}_count'] += 1
            if 'pci_advantages' in transition and agent_id in transition['pci_advantages']:
                pci_res = transition['pci_advantages'][agent_id]
                agent_losses[f'pci_res_adv_agent_{agent_id}_{scenario}_sum'] += pci_res
                agent_losses[f'pci_res_adv_agent_{agent_id}_{scenario}_count'] += 1
                agent_losses[f'pci_bonus_abs_agent_{agent_id}_{scenario}_sum'] += abs(
                    current_causal_lambda * pci_res
                )
                agent_losses[f'pci_bonus_abs_agent_{agent_id}_{scenario}_count'] += 1
            return

        if self._is_smac_attack_reward_event(transition, agent_id):
            scenario = 'attack_reward'
            agent_losses[f'delta_agent_{agent_id}_{scenario}_sum'] += transition['delta'][agent_id]
            agent_losses[f'delta_agent_{agent_id}_{scenario}_count'] += 1
            if 'pci_raw_advantages' in transition and agent_id in transition['pci_raw_advantages']:
                pci_raw = transition['pci_raw_advantages'][agent_id]
                agent_losses[f'pci_raw_adv_agent_{agent_id}_{scenario}_sum'] += pci_raw
                agent_losses[f'pci_raw_adv_agent_{agent_id}_{scenario}_count'] += 1
            if 'pci_advantages' in transition and agent_id in transition['pci_advantages']:
                pci_res = transition['pci_advantages'][agent_id]
                agent_losses[f'pci_res_adv_agent_{agent_id}_{scenario}_sum'] += pci_res
                agent_losses[f'pci_res_adv_agent_{agent_id}_{scenario}_count'] += 1
                agent_losses[f'pci_bonus_abs_agent_{agent_id}_{scenario}_sum'] += abs(
                    current_causal_lambda * pci_res
                )
                agent_losses[f'pci_bonus_abs_agent_{agent_id}_{scenario}_count'] += 1
            damage_share = self._smac_damage_share_from_transition(transition, agent_id)
            if damage_share is not None:
                agent_losses[f'damage_share_agent_{agent_id}_{scenario}_sum'] += damage_share
                agent_losses[f'damage_share_agent_{agent_id}_{scenario}_count'] += 1
            return

        others_rewards_sum = sum(
            [transition['bold_r_t'][j] for j in range(self.num_agents) if j != agent_id]
        )
        if others_rewards_sum == 0:
            agent_losses[f'delta_agent{agent_id}_no_reward_sum'] += transition['delta'][agent_id]
            agent_losses[f'delta_agent{agent_id}_no_reward_count'] += 1
        else:
            agent_losses[f'delta_agent{agent_id}_positive_reward_sum'] += transition['delta'][agent_id]
            agent_losses[f'delta_agent{agent_id}_positive_reward_count'] += 1

    def _is_smac_attack_reward_event(
        self,
        transition: Dict,
        agent_id: int,
    ) -> bool:
        env_name = self.env.__class__.__name__
        if env_name not in {'SMACWrapper', 'SMACv2Wrapper', 'SMAXWrapper', 'SMAXBatchWrapper'}:
            return False

        action = transition.get('bold_a_t', {}).get(agent_id, None)
        if action is None:
            return False
        attack_threshold = self._attack_action_threshold(transition)
        if attack_threshold is None:
            return False
        if int(action) < int(attack_threshold):
            return False

        available_actions = transition.get('available_actions')
        if isinstance(available_actions, dict) and agent_id in available_actions:
            avail = available_actions.get(agent_id)
            if torch.is_tensor(avail):
                avail_vals = avail.detach().cpu().tolist()
            else:
                avail_vals = avail
            if avail_vals is not None and len(avail_vals) > 1:
                alive = float(sum(avail_vals[1:])) > 0.0
                if not alive:
                    return False

        if self._global_reward_from_dict(transition['bold_r_t']) <= 1e-8:
            return False

        damage = self._smac_damage_from_transition(transition, agent_id)
        if damage is not None and damage <= 1e-8:
            return False
        return True

    def _attack_action_threshold(self, transition: Dict) -> Optional[int]:
        info = transition.get('info', {})
        threshold = None
        if isinstance(info, dict):
            threshold = info.get('attack_action_threshold', info.get('n_actions_no_attack'))
            if threshold is None:
                n_enemies = info.get('n_enemies')
                try:
                    if n_enemies is not None:
                        threshold = self.action_dim - int(n_enemies)
                except (TypeError, ValueError):
                    threshold = None
        if threshold is None:
            env_n_enemies = getattr(self.env, 'num_enemies', None)
            try:
                if env_n_enemies is not None:
                    threshold = self.action_dim - int(env_n_enemies)
            except (TypeError, ValueError):
                threshold = None
        if threshold is None:
            if self.action_dim <= 6:
                return None
            return 6
        try:
            threshold = int(threshold)
            threshold = max(0, min(threshold, self.action_dim - 1))
            return threshold
        except (TypeError, ValueError):
            if self.action_dim <= 6:
                return None
            return 6

    def _smac_damage_from_transition(
        self,
        transition: Dict,
        agent_id: int,
    ) -> Optional[float]:
        info = transition.get('info', {})
        if not isinstance(info, dict):
            return None
        damages = info.get('individual_damages')
        if damages is None:
            return None
        if agent_id >= len(damages):
            return None
        try:
            return float(damages[agent_id])
        except (TypeError, ValueError):
            return None

    def _smac_damage_share_from_transition(
        self,
        transition: Dict,
        agent_id: int,
    ) -> Optional[float]:
        info = transition.get('info', {})
        if not isinstance(info, dict):
            return None
        damages = info.get('individual_damages')
        if damages is None or agent_id >= len(damages):
            return None
        total_damage = 0.0
        for d in damages:
            try:
                total_damage += max(float(d), 0.0)
            except (TypeError, ValueError):
                continue
        if total_damage <= 1e-8:
            return None
        try:
            return max(float(damages[agent_id]), 0.0) / total_damage
        except (TypeError, ValueError):
            return None

    def _prepare_update_batch_tensors(self, batch_data: List[Dict]):
        full_obs_list = [torch.stack(list(t['bold_o_t'].values())) for t in batch_data]
        full_obs_batch = torch.stack(full_obs_list).to(self.device, non_blocking=True)
        actions_batch = torch.tensor(
            [[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data],
            dtype=torch.long,
            device=self.device,
        )

        available_actions_batch = build_available_actions_batch(
            batch_data,
            self.num_agents,
            self.device,
        )

        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
            init_hidden_states = batch_data[0]['hidden_states']
        else:
            init_hidden_states = {
                i: (
                    torch.zeros(self.recurrent_N, self.hidden_dim),
                    torch.zeros(self.recurrent_N, self.hidden_dim),
                )
                for i in range(self.num_agents)
            }

        masks_batch = build_agent_masks_batch(
            batch_data,
            self.num_agents,
            self.device,
        )

        return full_obs_batch, actions_batch, available_actions_batch, init_hidden_states, masks_batch

    def _prepare_agent_training_batch(
        self,
        batch_data: List[Dict],
        agent_id: int,
        agent_obs: torch.Tensor,
        my_actions: torch.Tensor,
        agent_advantage: torch.Tensor,
        agent_return: torch.Tensor,
        agent_delta_return: torch.Tensor,
        masks_batch: Dict[int, torch.Tensor],
        init_hidden_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        available_actions_all: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        adv = normalize_advantages(agent_advantage) # NOTE: 2nd normalization
        p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
        v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
        delta_states_all = torch.stack([
            (
                t['hidden_states'][agent_id][3]
                if len(t['hidden_states'][agent_id]) > 3
                else torch.zeros(self.recurrent_N, self.hidden_dim)
            )
            for t in batch_data
        ]).to(self.device)

        T_total = agent_obs.size(0)
        chunk_len = self.rnn_chunk_length
        num_chunks = T_total // chunk_len
        agent_active_masks = build_agent_active_masks(batch_data, agent_id, self.device)

        if num_chunks > 0:
            rnn_obs_chunked = chunk_sequence(agent_obs, chunk_len)
            my_actions_chunked = chunk_sequence(my_actions, chunk_len)
            rnn_masks_main = masks_batch[agent_id]
            rnn_masks_chunked = chunk_sequence(rnn_masks_main, chunk_len)

            prev_actions_raw = torch.tensor(
                [t['bold_a_t_minus_1'][agent_id] for t in batch_data[:num_chunks * chunk_len]],
                dtype=torch.long,
                device=self.device,
            )
            prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
            prev_actions_chunked = chunk_sequence(prev_actions_onehot_seq, chunk_len)

            active_masks_chunked = chunk_sequence(agent_active_masks, chunk_len)

            avail_actions_chunked = None
            if available_actions_all is not None:
                avail_actions_chunked = chunk_sequence(available_actions_all, chunk_len)

            chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
            p_h0_chunked = p_states_all[chunk_indices]
            v_h0_chunked = v_states_all[chunk_indices]
            delta_v_h0_chunked = delta_states_all[chunk_indices]
            adv_chunked = chunk_sequence(adv, chunk_len)
            ret_chunked = chunk_sequence(agent_return, chunk_len)
            delta_ret_chunked = chunk_sequence(agent_delta_return, chunk_len)

            avail_actions_batch_merged = None
            if avail_actions_chunked is not None:
                avail_actions_batch_merged = avail_actions_chunked.reshape(-1, avail_actions_chunked.size(-1))

            old_log_probs_raw = torch.tensor(
                [t['log_probs'][agent_id] for t in batch_data],
                dtype=torch.float32,
                device=self.device,
            )
            old_log_probs = chunk_sequence(old_log_probs_raw, chunk_len).reshape(-1)

            return {
                "num_chunks": num_chunks,
                "rnn_obs_batch": rnn_obs_chunked,
                "rnn_masks_batch": rnn_masks_chunked,
                "p_h0_batch": p_h0_chunked,
                "v_h0_batch": v_h0_chunked,
                "delta_v_h0_batch": delta_v_h0_chunked,
                "my_actions_batch": my_actions_chunked.reshape(-1),
                "adv_batch": adv_chunked.reshape(-1),
                "ret_batch": ret_chunked.reshape(-1),
                "delta_ret_batch": delta_ret_chunked.reshape(-1),
                "active_masks_batch": active_masks_chunked.reshape(-1),
                "prev_actions_batch": prev_actions_chunked,
                "prev_actions_batch_merged": prev_actions_chunked.reshape(-1, self.action_dim),
                "avail_actions_batch_merged": avail_actions_batch_merged,
                "old_log_probs": old_log_probs,
            }

        p_h0_batch = init_hidden_states[agent_id][0]
        p_h0_batch = p_h0_batch.to(self.device).unsqueeze(0)
        v_h0_batch = init_hidden_states[agent_id][1]
        v_h0_batch = v_h0_batch.to(self.device).unsqueeze(0)
        if len(init_hidden_states[agent_id]) > 3:
            delta_v_h0_batch = init_hidden_states[agent_id][3].to(self.device).unsqueeze(0)
        else:
            delta_v_h0_batch = torch.zeros(1, self.recurrent_N, self.hidden_dim, device=self.device)
        prev_actions_raw = torch.tensor(
            [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
            dtype=torch.long,
            device=self.device,
        )
        prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
        old_log_probs = torch.tensor(
            [t['log_probs'][agent_id] for t in batch_data],
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "num_chunks": 0,
            "rnn_obs_batch": agent_obs.unsqueeze(1),
            "rnn_masks_batch": masks_batch[agent_id],
            "p_h0_batch": p_h0_batch,
            "v_h0_batch": v_h0_batch,
            "delta_v_h0_batch": delta_v_h0_batch,
            "my_actions_batch": my_actions,
            "adv_batch": adv,
            "ret_batch": agent_return,
            "delta_ret_batch": agent_delta_return,
            "active_masks_batch": agent_active_masks,
            "prev_actions_batch": prev_actions_onehot_seq.unsqueeze(1),
            "prev_actions_batch_merged": prev_actions_onehot_seq,
            "avail_actions_batch_merged": available_actions_all,
            "old_log_probs": old_log_probs,
        }

    def _debug_causal_effects(
        self,
        r_hat_all: torch.Tensor,      # (B*N, action_dim, N)
        actions_flat: torch.Tensor,    # (B*N,)
        r_factual: torch.Tensor,       # (B*N, N)
        baseline: torch.Tensor,        # (B*N, N)
        raw_effects: torch.Tensor,     # (B*N, N)
        action_dim: int,
        alive_mask: torch.Tensor = None  # (B*N,) bool — True if alive
    ):
        """Debug print for causal effect analysis in SMACv2 domain."""
        import numpy as np
        np.set_printoptions(precision=4, suppress=True, linewidth=200)

        total = actions_flat.size(0)
        if alive_mask is None:
            alive_mask = torch.ones(total, dtype=torch.bool, device=actions_flat.device)
        n_alive = alive_mask.sum().item()
        n_dead = total - n_alive

        print(f"\n{'='*80}")
        print(f"[Causal Debug] samples={total}, alive={n_alive}, dead={n_dead}")
        print(f"[Causal Debug] Outcome Bridge Predictions (r_hat_all) per action  (alive only)")
        print(f"  r_hat_all shape: {r_hat_all.shape}  (batch*agents, action_dim, num_agents)")
        if n_alive > 0:
            r_hat_alive = r_hat_all[alive_mask]
            r_hat_mean = r_hat_alive.detach().mean(dim=0).cpu().numpy()
        else:
            r_hat_mean = r_hat_all.detach().mean(dim=0).cpu().numpy()
        action_labels = {0: 'noop', 1: 'stop', 2: 'north', 3: 'south', 4: 'east', 5: 'west'}
        for a_idx in range(min(action_dim, r_hat_mean.shape[0])):
            label = action_labels.get(a_idx, f'attack_{a_idx-6}') if a_idx < 6 else f'attack_{a_idx-6}'
            print(f"  action {a_idx:2d} ({label:>8s}): r_hat mean = {r_hat_mean[a_idx]}")
        
        print(f"\n[Causal Debug] Baseline & Raw Effects by action group  (alive only)")
        alive_actions = actions_flat[alive_mask]
        alive_r_factual = r_factual[alive_mask]
        alive_baseline = baseline[alive_mask]
        alive_raw_effects = raw_effects[alive_mask]

        move_mask = alive_actions < 6
        attack_mask = alive_actions >= 6
        if move_mask.any():
            print(f"  Movement (0~5): count={move_mask.sum().item()}")
            print(f"    r_factual  mean: {alive_r_factual[move_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    baseline   mean: {alive_baseline[move_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    raw_effect mean: {alive_raw_effects[move_mask].mean(dim=0).detach().cpu().numpy()}")
        if attack_mask.any():
            print(f"  Attack   (6~ ): count={attack_mask.sum().item()}")
            print(f"    r_factual  mean: {alive_r_factual[attack_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    baseline   mean: {alive_baseline[attack_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    raw_effect mean: {alive_raw_effects[attack_mask].mean(dim=0).detach().cpu().numpy()}")
        print(f"{'='*80}")

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()
