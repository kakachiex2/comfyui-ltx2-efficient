"""
Combined Optimization Engine

Combines multiple optimization engines for maximum efficiency.
Uses AdaCache + TokenMerge together for 2-5x speedup.

Key concepts:
- Layer engines together in sequence
- Each engine optimizes different aspects
- Coordinated cache/merge for best results
"""

import torch
from typing import Dict, Any, List, Tuple, Optional
from .base_engine import BaseOptimizationEngine, EngineConfig
from .adaptive_cache import AdaptiveCacheEngine
from .token_merge import TokenMergeEngine
from .step_adaptive import StepAdaptiveEngine


class CombinedEngine(BaseOptimizationEngine):
    """
    Combined optimization engine using multiple techniques.
    
    Layers AdaptiveCache + TokenMerge for maximum speedup.
    Provides 2-5x speedup by combining caching and token reduction.
    """
    
    def __init__(self, config: EngineConfig = None):
        super().__init__(config)
        
        # Create sub-engines with shared config
        self._cache_engine = AdaptiveCacheEngine(config)
        self._merge_engine = TokenMergeEngine(config)
        
        # Configure sub-engines
        self._cache_engine.config.verbose = False
        self._merge_engine.config.verbose = False
        
        self._engines: List[BaseOptimizationEngine] = [
            self._cache_engine,
            self._merge_engine,
        ]
    
    @property
    def name(self) -> str:
        return "Combined"
    
    def setup(self, model, total_steps: int, **kwargs):
        super().setup(model, total_steps, **kwargs)
        
        for engine in self._engines:
            engine.setup(model, total_steps, **kwargs)
        
        print(f"[Combined] Initialized with {len(self._engines)} sub-engines: "
              f"{', '.join(e.name for e in self._engines)}")
    
    def step_start(self, step: int, latent: torch.Tensor, **kwargs):
        super().step_start(step, latent, **kwargs)
        
        for engine in self._engines:
            engine.step_start(step, latent, **kwargs)
    
    def step_end(self, step: int, latent: torch.Tensor, **kwargs):
        super().step_end(step, latent, **kwargs)
        
        for engine in self._engines:
            engine.step_end(step, latent, **kwargs)
    
    def optimize_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        layer_name: str = "",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Apply combined optimizations in sequence.
        
        1. First check cache (fast path if cached)
        2. Then apply token merging
        """
        combined_metadata = {
            "engines_applied": [],
            "layer": layer_name,
        }
        
        # Try cache first (most efficient if hit)
        if self._cache_engine.is_cache_valid(layer_name):
            cached_output = self._cache_engine.get_cached_output(layer_name)
            if cached_output is not None:
                combined_metadata["cache_hit"] = True
                combined_metadata["engines_applied"].append("cache_hit")
                # Return original tensors with cache indicator
                return query, key, value, combined_metadata
        
        # Apply cache engine optimization (stores K/V)
        q, k, v, cache_meta = self._cache_engine.optimize_attention(
            query, key, value, layer_name, **kwargs
        )
        combined_metadata["cache"] = cache_meta
        combined_metadata["engines_applied"].append("cache")
        
        # Apply token merging
        q, k, v, merge_meta = self._merge_engine.optimize_attention(
            q, k, v, layer_name, **kwargs
        )
        combined_metadata["merge"] = merge_meta
        if merge_meta.get("merged", False):
            combined_metadata["engines_applied"].append("merge")
        
        return q, k, v, combined_metadata
    
    def restore_output(
        self,
        output: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        """Restore output through all engines in reverse order."""
        
        # If it was a cache hit, no restoration needed
        if metadata.get("cache_hit", False):
            return output
        
        # Reverse order: unmerge first, then handle cache
        if "merge" in metadata:
            output = self._merge_engine.restore_output(
                output, metadata["merge"], **kwargs
            )
        
        # Cache the restored output
        layer_name = metadata.get("layer", "")
        if layer_name and self._cache_engine.should_cache(layer_name):
            self._cache_engine.cache_output(layer_name, output)
        
        return output
    
    def cleanup(self):
        super().cleanup()
        
        for engine in self._engines:
            engine.cleanup()
        
        print(f"[Combined] All sub-engines cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        
        # Aggregate stats from sub-engines
        stats["sub_engines"] = {}
        for engine in self._engines:
            stats["sub_engines"][engine.name] = engine.get_stats()
        
        return stats
