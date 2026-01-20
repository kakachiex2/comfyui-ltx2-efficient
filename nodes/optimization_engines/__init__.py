"""
LTX2 Optimization Engines Module

Provides various temporal attention optimization techniques for video diffusion models:
- AdaptiveCacheEngine: Cache and reuse attention outputs across steps
- TokenMergeEngine: Merge similar tokens across frames (VidToMe)
- StepAdaptiveEngine: Dynamic sparsity based on denoising progress
- CombinedEngine: Multi-engine wrapper for maximum efficiency
"""

from .base_engine import BaseOptimizationEngine, EngineConfig
from .adaptive_cache import AdaptiveCacheEngine
from .token_merge import TokenMergeEngine
from .step_adaptive import StepAdaptiveEngine
from .combined import CombinedEngine

# Engine registry for dropdown menus
ENGINE_REGISTRY = {
    "None": None,
    "Adaptive Cache (2-4x speedup)": AdaptiveCacheEngine,
    "Token Merge / VidToMe (1.3-1.5x)": TokenMergeEngine,
    "Step Adaptive (1.2-1.4x)": StepAdaptiveEngine,
    "Combined (2-5x speedup)": CombinedEngine,
}

def get_engine(name: str, config: EngineConfig = None):
    """Factory function to create an optimization engine by name."""
    engine_cls = ENGINE_REGISTRY.get(name)
    if engine_cls is None:
        return None
    return engine_cls(config or EngineConfig())

def get_engine_names():
    """Get list of available engine names for dropdown."""
    return list(ENGINE_REGISTRY.keys())

__all__ = [
    "BaseOptimizationEngine",
    "EngineConfig",
    "AdaptiveCacheEngine",
    "TokenMergeEngine",
    "StepAdaptiveEngine", 
    "CombinedEngine",
    "ENGINE_REGISTRY",
    "get_engine",
    "get_engine_names",
]
