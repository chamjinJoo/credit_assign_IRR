"""
Proximal Causal Inference Multi-Agent PPO (PCI-MAPPO)
A multi-agent reinforcement learning framework incorporating causal inference.
"""

__version__ = "0.1.0"
__author__ = "PCI-MAPPO Team"

# Import main classes for easy access
from .envs import (
    SimpleCooperativeEnv,
    PettingZooWrapper,
    ToyCausalEnv,
    DecTigerEnv,
    StagHuntEnv,
    dict_to_tensor
)

from .algorithms import PCIMAPPOAlgorithm
from .training import PCIMAPPOTrainer
from .utils import Logger, create_config_from_args
