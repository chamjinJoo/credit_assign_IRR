"""Compatibility exports for environment classes.

Concrete environment implementations live in smaller modules. Keep this file
so legacy imports such as `from envs.multi_agent_env import StagHuntEnv` keep
working.
"""

from .base import MultiAgentEnv
from .stag_hunt import StagHuntEnv
from .smax import SMAXWrapper, SMAXBatchWrapper
from .lbf import LBForagingWrapper, JumanjiLBFBatchWrapper

__all__ = [
    "MultiAgentEnv",
    "StagHuntEnv",
    "SMAXWrapper",
    "SMAXBatchWrapper",
    "LBForagingWrapper",
    "JumanjiLBFBatchWrapper",
]
