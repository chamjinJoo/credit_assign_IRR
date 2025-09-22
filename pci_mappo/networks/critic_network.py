import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .mlp import MLPBase


class RBFKernel(nn.Module):
    """
    Radial Basis Function (RBF) kernel for RKHS.
    K(x, y) = exp(-gamma * ||x - y||^2)
    """
    
    def __init__(self, gamma: float = 1.0):
        super(RBFKernel, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel between x and y.
        
        Args:
            x: [batch_size, feature_dim]
            y: [num_centers, feature_dim]
            
        Returns:
            kernel_matrix: [batch_size, num_centers]
        """
        # Compute squared distances
        x_norm = (x ** 2).sum(1).view(-1, 1)  # [batch_size, 1]
        y_norm = (y ** 2).sum(1).view(1, -1)  # [1, num_centers]
        dists = x_norm + y_norm - 2.0 * torch.mm(x, y.t())  # [batch_size, num_centers]
        
        # Apply RBF kernel
        return torch.exp(-self.gamma * dists)


class PolynomialKernel(nn.Module):
    """
    Polynomial kernel for RKHS.
    K(x, y) = (gamma * <x, y> + coef0)^degree
    """
    
    def __init__(self, degree: int = 2, gamma: float = 1.0, coef0: float = 0.0):
        super(PolynomialKernel, self).__init__()
        self.degree = degree
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.coef0 = nn.Parameter(torch.tensor(coef0))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial kernel between x and y.
        
        Args:
            x: [batch_size, feature_dim]
            y: [num_centers, feature_dim]
            
        Returns:
            kernel_matrix: [batch_size, num_centers]
        """
        # Compute dot products
        dot_products = torch.mm(x, y.t())  # [batch_size, num_centers]
        
        # Apply polynomial kernel
        return (self.gamma * dot_products + self.coef0) ** self.degree


class RKHSNetwork(nn.Module):
    """
    Reproducing Kernel Hilbert Space (RKHS) based network.
    Uses kernel functions to map inputs to a high-dimensional feature space.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_centers: int = 100,
        kernel_type: str = "rbf",
        kernel_params: Optional[dict] = None,
        output_dim: int = 1
    ):
        super(RKHSNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.kernel_type = kernel_type
        self.output_dim = output_dim
        
        # Initialize kernel centers randomly
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        
        # Initialize kernel function
        if kernel_type == "rbf":
            gamma = kernel_params.get("gamma", 1.0) if kernel_params else 1.0
            self.kernel = RBFKernel(gamma=gamma)
        elif kernel_type == "polynomial":
            degree = kernel_params.get("degree", 2) if kernel_params else 2
            gamma = kernel_params.get("gamma", 1.0) if kernel_params else 1.0
            coef0 = kernel_params.get("coef0", 0.0) if kernel_params else 0.0
            self.kernel = PolynomialKernel(degree=degree, gamma=gamma, coef0=coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        # Output layer to map kernel features to final output
        self.output_layer = nn.Linear(num_centers, output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RKHS network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        # Compute kernel features
        kernel_features = self.kernel(x, self.centers)  # [batch_size, num_centers]
        # Apply output layer
        output = self.output_layer(kernel_features)  # [batch_size, output_dim]
        
        return output


class CriticNetwork(nn.Module):
    """
    RKHS-based Critic network q'_φ for adversarial training with Bridge network.
    Uses Reproducing Kernel Hilbert Space to map context variables (z) and 
    full action vector to scalar output. This provides more flexible function
    approximation compared to standard MLPs.
    """
    
    def __init__(
        self, 
        z_dim: int,  # z dimension = concatenated observations dim
        action_dim: int,   # single agent action dimension
        num_agents: int,   # number of agents
        action_embedding_dim: int = 8,  # Action embedding dimension
        num_centers: int = 100,  # Number of kernel centers for RKHS
        kernel_type: str = "rbf",  # Kernel type: "rbf" or "polynomial"
        kernel_params: Optional[dict] = None  # Kernel parameters
    ):
        super(CriticNetwork, self).__init__()
        
        self.z_dim = z_dim 
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.action_embedding_dim = action_embedding_dim
        
        # Action embedding for better action representation
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        # Input is concatenation of context and all agents' action embeddings
        input_dim = z_dim + num_agents * action_embedding_dim
        
        # RKHS-based critic network
        self.critic_network = RKHSNetwork(
            input_dim=input_dim,
            num_centers=num_centers,
            kernel_type=kernel_type,
            kernel_params=kernel_params,
            output_dim=1
        )
        
    def forward(
        self, 
        z: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of RKHS-based critic network.
        
        Args:
            z: Context variables z [batch_size, z_dim]
            actions: Full action vector a = (a^1_t, a^2_t, ..., a^n_t) [batch_size, num_agents]
            
        Returns:
            critic_output: Scalar critic values [batch_size, 1]
        """
        batch_size = z.shape[0]
        
        # Get action embeddings for all agents
        # actions: [batch_size, num_agents] -> action_embeds: [batch_size, num_agents, action_embedding_dim]
        action_embeds = self.action_embedding(actions)  # [batch_size, num_agents, action_embedding_dim]
        # Flatten action embeddings: [batch_size, num_agents * action_embedding_dim]
        action_embeds_flat = action_embeds.view(batch_size, -1)
        
        # Concatenate z and all action embeddings
        critic_input = torch.cat([z, action_embeds_flat], dim=-1)
        
        # Forward through RKHS-based critic network
        critic_output = self.critic_network(critic_input)  # [batch_size, 1]
        
        return critic_output
