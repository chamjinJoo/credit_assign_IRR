import torch
import torch.nn as nn

###########################################################################
# From https://github.com/marlbenchmark/on-policy, edited for our usage.
# Modified: prev_action (one-hot) is concatenated to obs before feature extraction
#           so that the RNN hidden state encodes (obs, action) pair history.
###########################################################################

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


class RNNPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, recurrent_N=1):
        super(RNNPolicyNetwork, self).__init__()
        self._hidden_dim = hidden_dim
        self._recurrent_N = recurrent_N
        self._action_dim = action_dim
        # Input: [obs || one_hot(prev_action)]
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.rnn = RNNLayer(hidden_dim, hidden_dim, recurrent_N, use_orthogonal=True)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, rnn_states, masks, available_actions=None, prev_actions=None):
        # prev_actions: one-hot (B, action_dim) or (T*N, action_dim)
        if prev_actions is None:
            prev_actions = torch.zeros(*obs.shape[:-1], self._action_dim, device=obs.device)
        x = torch.cat([obs, prev_actions], dim=-1)
        x = self.feature_extractor(x)
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        x, rnn_states = self.rnn(x, rnn_states, masks)
        logits = self.output_layer(x)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10
        return logits, rnn_states

    def get_action_and_log_prob(self, obs, rnn_states, masks, available_actions=None, prev_actions=None):
        logits, rnn_states = self.forward(obs, rnn_states, masks, available_actions, prev_actions=prev_actions)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, prev_actions=None):
        logits, _ = self.forward(obs, rnn_states, masks, available_actions, prev_actions=prev_actions)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class RNNValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, recurrent_N=1, out_dim=1, action_dim=0, num_agents=1):
        super(RNNValueNetwork, self).__init__()
        self._hidden_dim = hidden_dim
        self._recurrent_N = recurrent_N
        self._action_dim = action_dim
        self._num_agents = num_agents
        # Input: [obs || one_hot(joint_prev_actions)]
        # For centralized critic: joint_prev_actions is (action_dim * num_agents)
        # For independent critic: joint_prev_actions is (action_dim)
        prev_action_input_dim = action_dim * num_agents
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim + prev_action_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.rnn = RNNLayer(hidden_dim, hidden_dim, recurrent_N, use_orthogonal=True)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs, rnn_states, masks, prev_actions=None):
        # prev_actions: one-hot (B, action_dim * num_agents) or (T*N, action_dim * num_agents)
        expected_dim = self._action_dim * self._num_agents
        if prev_actions is None:
            prev_actions = torch.zeros(*obs.shape[:-1], expected_dim, device=obs.device)
        x = torch.cat([obs, prev_actions], dim=-1)
        x = self.feature_extractor(x)
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        x, rnn_states = self.rnn(x, rnn_states, masks)
        values = self.output_layer(x)
        return values, rnn_states