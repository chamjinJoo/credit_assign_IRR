import torch
import torch.nn as nn
from typing import Tuple, Optional


class RNNLayer(nn.Module):
    """
    GRU-based RNN layer with mask handling for variable-length sequences.
    Based on on-policy implementation, supports episode boundaries and invalid timesteps.
    """
    
    def __init__(
        self, 
        inputs_dim: int, 
        outputs_dim: int, 
        recurrent_N: int = 1, 
        use_orthogonal: bool = True
    ):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        # Use GRU as specified
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        
        # Initialize weights following on-policy style
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
                    
        # Layer normalization
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        Forward pass through RNN with mask handling.
        
        Args:
            x: Input tensor [T*N, input_dim] or [N, input_dim]
            hxs: Hidden states [N, recurrent_N, hidden_dim] 
            masks: Masks [T*N, 1] or [N, 1] indicating valid timesteps (1=valid, 0=reset)
            
        Returns:
            output: RNN output [T*N, output_dim] or [N, output_dim]
            new_hxs: New hidden states [N, recurrent_N, hidden_dim]
        """
        if x.size(0) == hxs.size(0):
            # Single timestep case
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous()
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # Multi-timestep case: x is (T*N, -1) tensor that has been flattened from (T, N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # Unflatten to (T, N, input_dim)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            # Find timesteps with zeros (episode boundaries)
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

        x = self.norm(x)
        return x, hxs

class MultiAgentSharedRNNLayer(nn.Module):
    """
    Shared RNN layer for encoding all agents' observations into a shared history.
    Following pseudocode: h_t = RNN(𝐨_t, h_{t-1}) where 𝐨_t is the set of all agents' observations.
    This version processes all agents' observations together.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        hidden_dim: int, 
        recurrent_N: int = 1, 
        use_orthogonal: bool = True
    ):
        super(MultiAgentSharedRNNLayer, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        # GRU for shared history encoding
        self.rnn = nn.GRU(obs_dim, hidden_dim, num_layers=self._recurrent_N)
        
        # Initialize weights following on-policy style
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
                    
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, hxs: Optional[torch.Tensor] = None, 
                masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for shared history encoding using all agents' observations.
        Handles both single timestep and multi-timestep cases with flexible processing.
        
        Args:
            x: All agents' observations tensor [num_agents, obs_dim] or [batch_size, num_agents, obs_dim] or [T*N, num_agents, obs_dim]
            hxs: Hidden states [1, recurrent_N, hidden_dim] or None
            masks: Masks [1, 1] or [T*N, 1] (1=valid, 0=reset) or None
            
        Returns:
            output: RNN output [batch_size, hidden_dim] or [T*N, hidden_dim]
            new_hxs: New hidden states [1, recurrent_N, hidden_dim]
        """
        # Handle different input shapes
        if x.dim() == 2:
            # [num_agents, obs_dim] -> [1, num_agents, obs_dim]
            x = x.unsqueeze(0)
            batch_size = 1
            num_agents = x.size(1)
        elif x.dim() == 3:
            # [batch_size, num_agents, obs_dim] or [T*N, num_agents, obs_dim]
            batch_size = x.size(0)
            num_agents = x.size(1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")
        
        # Initialize hidden state if not provided
        if hxs is None:
            hxs = torch.zeros(1, self._recurrent_N, self.hidden_dim,
                            device=x.device, dtype=x.dtype)
        
        # Initialize masks if not provided (all valid)
        if masks is None:
            masks = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # Check if this is a multi-timestep case (flattened from sequence)
        if batch_size > 1 and masks.size(0) == batch_size:
            # Multi-timestep case: x is (T*N, num_agents, obs_dim) that has been flattened
            N = 1  # For shared history, we treat each timestep as a single batch
            T = batch_size
            
            # Reshape to (T, N, num_agents, obs_dim)
            x = x.view(T, N, num_agents, x.size(-1))
            masks = masks.view(T, N)
            
            # Find timesteps with zeros (episode boundaries)
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
            
            # Prepare hidden state for RNN
            hxs = hxs.transpose(0, 1)
            
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                
                # Apply mask to hidden state
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                
                # Process this segment
                segment_x = x[start_idx:end_idx]  # [segment_len, N, num_agents, obs_dim]
                segment_len = segment_x.size(0)
                
                # Process each timestep in the segment
                segment_outputs = []
                for t in range(segment_len):
                    timestep_x = segment_x[t]  # [N, num_agents, obs_dim]
                    
                    # For shared history, we process all agents' observations sequentially
                    # Reshape to [num_agents, N, obs_dim] for RNN processing
                    timestep_x_reshaped = timestep_x.transpose(0, 1)  # [num_agents, N, obs_dim]
                    
                    # Process through RNN with proper gradient flow
                    rnn_scores, temp = self.rnn(timestep_x_reshaped, temp)
                    
                    # Take the last output (after processing all agents)
                    last_output = rnn_scores[-1]  # [N, hidden_dim]
                    segment_outputs.append(last_output)
                
                # Concatenate segment outputs
                segment_output = torch.stack(segment_outputs, dim=0)  # [segment_len, N, hidden_dim]
                outputs.append(segment_output)
            
            # Concatenate all segments
            x = torch.cat(outputs, dim=0)  # [T, N, hidden_dim]
            
            # Flatten back to [T*N, hidden_dim]
            x = x.reshape(T * N, -1)
            
            # Transpose hidden state back
            hxs = hxs.transpose(0, 1)
            
        else:
            # Single timestep case
            # Apply mask to hidden state
            hxs = hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)
            
            # Process all agents' observations sequentially
            # Reshape to [num_agents, batch_size, obs_dim] for RNN processing
            x_reshaped = x.transpose(0, 1)  # [num_agents, batch_size, obs_dim]
            
            # RNN expects hidden state of shape [num_layers, batch_size, hidden_size]
            hxs = hxs.transpose(0, 1)
            
            # Process through RNN
            rnn_scores, hxs = self.rnn(x_reshaped, hxs)
            
            # Take the last output (after processing all agents)
            x = rnn_scores[-1]  # [batch_size, hidden_dim]
            
            # Transpose hidden state back
            hxs = hxs.transpose(0, 1)
        
        x = self.norm(x)
        return x, hxs
    
    def reset_hidden_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Reset hidden state to zeros."""
        return torch.zeros(1, self._recurrent_N, self.hidden_dim, device=device, dtype=dtype)