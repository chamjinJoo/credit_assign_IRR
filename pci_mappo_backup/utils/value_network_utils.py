"""
Utility functions for handling value networks in both parameter sharing and individual network scenarios.
"""
from typing import Dict, Any, Optional
import torch


def get_value_networks_dict(algorithm) -> Dict[int, Any]:
    """
    Get value networks as a dictionary from algorithm, supporting both:
    - Individual value networks (value_nets: Dict[int, network])
    - Parameter sharing (value_net: single network)
    
    Args:
        algorithm: Algorithm instance (PCIMAPPOAlgorithm or similar)
        
    Returns:
        Dict[int, network]: Dictionary mapping agent_id to value network
    """
    # Check for individual value networks first (newer approach)
    if hasattr(algorithm, 'value_nets') and algorithm.value_nets is not None:
        return algorithm.value_nets
    
    # Fallback to parameter sharing (older approach)
    if hasattr(algorithm, 'value_net') and algorithm.value_net is not None:
        num_agents = getattr(algorithm, 'num_agents', 1)
        return {agent_id: algorithm.value_net for agent_id in range(num_agents)}
    
    raise AttributeError(
        "Algorithm must have either 'value_nets' (individual networks) "
        "or 'value_net' (parameter sharing) attribute"
    )


def get_value_network_for_agent(algorithm, agent_id: int) -> Any:
    """
    Get value network for a specific agent, supporting both parameter sharing and individual networks.
    
    Args:
        algorithm: Algorithm instance
        agent_id: Agent ID
        
    Returns:
        Value network for the agent
    """
    value_networks = get_value_networks_dict(algorithm)
    
    if agent_id not in value_networks:
        raise KeyError(f"Value network not found for agent {agent_id}")
    
    return value_networks[agent_id]


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

