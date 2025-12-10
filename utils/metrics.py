import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, deque


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_metrics = defaultdict(list)
    
    def add_scalar(self, name: str, value: float):
        """Add a scalar metric."""
        self.metrics[name].append(value)
    
    def add_episode_metric(self, name: str, value: float):
        """Add an episode-level metric."""
        self.episode_metrics[name].append(value)
    
    def get_mean(self, name: str) -> float:
        """Get mean of recent values."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return np.mean(self.metrics[name])
        return 0.0
    
    def get_std(self, name: str) -> float:
        """Get standard deviation of recent values."""
        if name in self.metrics and len(self.metrics[name]) > 1:
            return np.std(self.metrics[name])
        return 0.0
    
    def get_episode_mean(self, name: str, last_n: int = None) -> float:
        """Get mean of episode metrics."""
        if name in self.episode_metrics:
            values = self.episode_metrics[name]
            if last_n:
                values = values[-last_n:]
            return np.mean(values) if len(values) > 0 else 0.0
        return 0.0
    
    def get_episode_std(self, name: str, last_n: int = None) -> float:
        """Get standard deviation of episode metrics."""
        if name in self.episode_metrics:
            values = self.episode_metrics[name]
            if last_n:
                values = values[-last_n:]
            return np.std(values) if len(values) > 1 else 0.0
        return 0.0
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        
        # Recent scalar metrics
        for name in self.metrics:
            summary[name] = {
                'mean': self.get_mean(name),
                'std': self.get_std(name),
                'count': len(self.metrics[name])
            }
        
        # Episode metrics
        for name in self.episode_metrics:
            summary[f"{name}_episodes"] = {
                'mean': self.get_episode_mean(name),
                'std': self.get_episode_std(name),
                'count': len(self.episode_metrics[name])
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.episode_metrics.clear()


class PerformanceMonitor:
    """Monitor performance and detect convergence."""
    
    def __init__(
        self, 
        patience: int = 50,
        min_improvement: float = 0.01,
        window_size: int = 20
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.window_size = window_size
        
        self.rewards = deque(maxlen=window_size)
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        self.is_converged = False
    
    def update(self, reward: float) -> bool:
        """
        Update with new reward and check for convergence.
        
        Returns:
            True if converged, False otherwise
        """
        self.rewards.append(reward)
        
        if len(self.rewards) < self.window_size:
            return False
        
        current_mean = np.mean(self.rewards)
        
        if current_mean > self.best_reward + self.min_improvement:
            self.best_reward = current_mean
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        self.is_converged = self.no_improvement_count >= self.patience
        return self.is_converged
    
    def get_status(self) -> Dict[str, Any]:
        """Get convergence status."""
        return {
            'is_converged': self.is_converged,
            'best_reward': self.best_reward,
            'no_improvement_count': self.no_improvement_count,
            'current_mean': np.mean(self.rewards) if len(self.rewards) > 0 else 0.0,
            'patience_remaining': max(0, self.patience - self.no_improvement_count)
        }
