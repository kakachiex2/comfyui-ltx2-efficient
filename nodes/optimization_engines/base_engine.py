"""
Base Optimization Engine

Abstract base class for all temporal attention optimization engines.
Provides common interface and hooks for the sampling loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
import torch


@dataclass
class EngineConfig:
    """Configuration for optimization engines."""
    # Adaptive Cache settings
    cache_ratio: float = 0.5  # Fraction of steps to cache
    cache_threshold: float = 0.1  # Difference threshold for cache invalidation
    
    # Token Merge settings
    merge_ratio: float = 0.25  # Fraction of tokens to merge
    merge_temporal: bool = True  # Enable cross-frame merging
    
    # Step Adaptive settings
    sparse_start: float = 0.5  # Step ratio to start sparse attention
    sparse_ratio: float = 0.5  # Fraction of tokens to attend to in sparse mode
    
    # General settings
    enabled: bool = True
    verbose: bool = False
    
    # Custom settings dict for engine-specific options
    custom: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizationEngine(ABC):
    """
    Abstract base class for optimization engines.
    
    Engines hook into the sampling loop to optimize attention computations.
    Each engine implements specific optimization techniques.
    """
    
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.enabled = self.config.enabled
        self._step = 0
        self._total_steps = 1
        self._cache = {}
        
    @property
    def name(self) -> str:
        """Engine name for logging."""
        return self.__class__.__name__
    
    @property
    def progress(self) -> float:
        """Current progress ratio (0.0 to 1.0)."""
        return self._step / max(self._total_steps, 1)
    
    def setup(self, model, total_steps: int, **kwargs):
        """
        Initialize engine for a sampling run.
        
        Args:
            model: The diffusion model
            total_steps: Total number of denoising steps
            **kwargs: Additional context (conditioning, seed, etc.)
        """
        self._step = 0
        self._total_steps = total_steps
        self._cache.clear()
        
        if self.config.verbose:
            print(f"[{self.name}] Initialized for {total_steps} steps")
    
    def step_start(self, step: int, latent: torch.Tensor, **kwargs):
        """
        Called at the start of each denoising step.
        
        Args:
            step: Current step number (0-indexed)
            latent: Current latent tensor
            **kwargs: Additional step context
        """
        self._step = step
    
    def step_end(self, step: int, latent: torch.Tensor, **kwargs):
        """
        Called at the end of each denoising step.
        
        Args:
            step: Current step number
            latent: Result latent tensor
            **kwargs: Additional step context
        """
        pass
    
    @abstractmethod
    def optimize_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        layer_name: str = "",
        **kwargs
    ) -> tuple:
        """
        Optimize attention computation.
        
        Args:
            query: Query tensor (B, heads, seq_len, dim)
            key: Key tensor
            value: Value tensor
            layer_name: Name of the attention layer
            **kwargs: Additional context
            
        Returns:
            Tuple of (optimized_query, optimized_key, optimized_value, metadata)
            metadata contains info for post-processing (e.g., unmerge map)
        """
        pass
    
    def restore_output(
        self,
        output: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        """
        Restore output after optimized attention (e.g., unmerge tokens).
        
        Args:
            output: Attention output tensor
            metadata: Metadata from optimize_attention
            **kwargs: Additional context
            
        Returns:
            Restored output tensor
        """
        return output
    
    def should_cache(self, layer_name: str = "") -> bool:
        """Check if this layer's output should be cached."""
        return False
    
    def get_cached(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get cached output for a layer if available."""
        return self._cache.get(layer_name)
    
    def set_cached(self, layer_name: str, value: torch.Tensor):
        """Cache a layer's output."""
        self._cache[layer_name] = value
    
    def cleanup(self):
        """Cleanup after sampling run."""
        self._cache.clear()
        if self.config.verbose:
            print(f"[{self.name}] Cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics for logging."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "step": self._step,
            "total_steps": self._total_steps,
            "cache_size": len(self._cache),
        }
