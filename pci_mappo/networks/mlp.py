import torch
import torch.nn as nn


def init_layer(module, init_method, gain=1.0):
    """Initialize layer weights."""
    if isinstance(init_method, str):
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=gain)
        else:
            raise ValueError(f"Unknown init method: {init_method}")
    else:
        init_method(module.weight, gain=gain)
    
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    
    return module


class MLPLayer(nn.Module):
    """
    Multi-layer perceptron with configurable activation and normalization.
    Based on on-policy implementation but adapted for PCI-MAPPO usage.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_size: int, 
        layer_N: int, 
        use_orthogonal: bool = True, 
        use_ReLU: bool = True,
        use_layer_norm: bool = True
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        self.use_layer_norm = use_layer_norm

        # Choose activation function
        active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        
        # Choose initialization method
        init_method = 'orthogonal' if use_orthogonal else 'xavier_uniform'
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')

        # First layer
        self.fc1 = nn.Sequential()
        self.fc1.add_module('linear', init_layer(nn.Linear(input_dim, hidden_size), init_method, gain))
        self.fc1.add_module('activation', active_func)
        if self.use_layer_norm:
            self.fc1.add_module('norm', nn.LayerNorm(hidden_size))

        # Hidden layers
        self.fc_hidden = nn.ModuleList()
        for i in range(self._layer_N):
            layer = nn.Sequential()
            layer.add_module('linear', init_layer(nn.Linear(hidden_size, hidden_size), init_method, gain))
            layer.add_module('activation', active_func)
            if self.use_layer_norm:
                layer.add_module('norm', nn.LayerNorm(hidden_size))
            self.fc_hidden.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            output: Output tensor [batch_size, hidden_size]
        """
        x = self.fc1(x)
        for layer in self.fc_hidden:
            x = layer(x)
        return x


class MLPBase(nn.Module):
    """
    MLP base network with feature normalization option.
    Adapted from on-policy for PCI-MAPPO usage patterns.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int = 64,
        layer_N: int = 1,
        use_feature_normalization: bool = False,
        use_orthogonal: bool = True,
        use_ReLU: bool = True,
        use_layer_norm: bool = True
    ):
        super(MLPBase, self).__init__()
        
        self._use_feature_normalization = use_feature_normalization
        self.hidden_size = hidden_size

        # Feature normalization
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # Main MLP
        self.mlp = MLPLayer(
            obs_dim, 
            hidden_size,
            layer_N, 
            use_orthogonal, 
            use_ReLU,
            use_layer_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP base.
        
        Args:
            x: Input observations [batch_size, obs_dim]
            
        Returns:
            features: Encoded features [batch_size, hidden_size]
        """
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        return x