import torch
from typing import Union, Iterable, Tuple

def get_grad_norm(parameters: Union[torch.nn.Module, Iterable[torch.Tensor]]) -> Tuple[float, int]:
    """
    Calculates the global L2 gradient norm and the total number of parameters.
    
    Args:
        parameters: A PyTorch model or an iterable of Tensors.
        
    Returns:
        total_norm: The global L2 norm of the gradients.
        num_params: The total number of parameters with gradients.
    """
    if isinstance(parameters, torch.nn.Module):
        parameters = parameters.parameters()
        
    total_norm = 0.0
    num_params = 0
    
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += p.numel()
            
    total_norm = total_norm ** 0.5
    return total_norm, num_params

def is_value_network_shared(algorithm) -> bool:
    """
    Check if value networks use parameter sharing.
    
    Args:
        algorithm: Algorithm instance
        
    Returns:
        True if parameter sharing (single value_net), False if individual networks (value_nets)
    """
    if hasattr(algorithm, 'value_nets') and algorithm.value_nets is not None:
        return False
    
    if hasattr(algorithm, 'value_net') and algorithm.value_net is not None:
        return True
    
    raise AttributeError(
        "Algorithm must have either 'value_nets' or 'value_net' attribute"
    )