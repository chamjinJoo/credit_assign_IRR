"""
Reproducibility utilities for PCI-MAPPO.
Provides comprehensive seed setting for all random number generators.
"""

import random
import numpy as np
import torch
import os
from typing import Optional, Union


def set_global_seed(seed: Union[int, str], deterministic: bool = True) -> int:
    """
    Set seed for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed. If string, will be converted to integer hash.
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
        
    Returns:
        The actual seed value used (as integer)
    """
    if isinstance(seed, str):
        seed = hash(seed) % (2**32)
    
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    print(f"Global seed set to: {seed} (deterministic={deterministic})")
    return seed


def set_worker_seed(worker_id: int, base_seed: int) -> int:
    """
    Set seed for a specific worker (useful for multi-process training).
    
    Args:
        worker_id: Worker process ID
        base_seed: Base seed value
        
    Returns:
        Worker-specific seed
    """
    worker_seed = base_seed + worker_id
    set_global_seed(worker_seed, deterministic=True)
    return worker_seed


def get_random_state() -> dict:
    """
    Get current random state of all generators.
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
        'torch_cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def set_random_state(state: dict) -> None:
    """
    Restore random state from a saved state.
    
    Args:
        state: Dictionary containing random states
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    if torch.cuda.is_available() and state['torch_cuda_random'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda_random'])


def ensure_reproducible_environment():
    """
    Set environment variables to ensure reproducibility.
    Call this before any other imports or operations.
    """
    # Disable OpenMP threading to avoid non-deterministic behavior
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Set PyTorch threading
    torch.set_num_threads(1)
    
    print("Reproducible environment configured")


class ReproducibleContext:
    """
    Context manager for reproducible experiments.
    Automatically saves and restores random states.
    """
    
    def __init__(self, seed: Optional[Union[int, str]] = None, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
        
    def __enter__(self):
        if self.seed is not None:
            self.original_state = get_random_state()
            set_global_seed(self.seed, self.deterministic)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_state is not None:
            set_random_state(self.original_state)


def log_reproducibility_info(seed: int, logger=None):
    """
    Log reproducibility information for experiment tracking.
    
    Args:
        seed: The seed used
        logger: Logger instance (optional)
    """
    info = {
        'seed': seed,
        'python_version': f"{random.__version__ if hasattr(random, '__version__') else 'built-in'}",
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_deterministic': torch.backends.cudnn.deterministic,
        'cuda_benchmark': torch.backends.cudnn.benchmark,
    }
    
    if logger:
        logger.info("Reproducibility Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    else:
        print("Reproducibility Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return info
