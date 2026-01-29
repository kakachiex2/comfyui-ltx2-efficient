"""
FFN Chunking VRAM Optimization for LTX Video

Reduces peak VRAM by processing FFN layers in sequence chunks.
Based on: https://github.com/ox1111/comfyui_ltx-2_vram_memory_management

LTX-2's FFN layers expand hidden dimension by 4x:
  Input:        (batch, 57000, 4096)   ~0.9 GB
  Intermediate: (batch, 57000, 16384)  ~3.7 GB  ← Memory bottleneck!
  Output:       (batch, 57000, 4096)   ~0.9 GB

By chunking the sequence dimension, we reduce peak VRAM ~8x per FFN layer.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any


class ChunkedFFN(nn.Module):
    """
    Wrapper that processes FFN in sequence chunks to reduce VRAM.
    
    Args:
        num_chunks: Number of chunks to split sequence into (default: 8)
        min_sequence_per_chunk: Minimum tokens per chunk before chunking kicks in
    """
    
    def __init__(self, num_chunks: int = 8, min_sequence_per_chunk: int = 100):
        super().__init__()
        self.num_chunks = num_chunks
        self.min_sequence_per_chunk = min_sequence_per_chunk
        self._original_ffn = None
    
    def set_original(self, ffn: nn.Module):
        """Store reference to original FFN module."""
        self._original_ffn = ffn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through FFN in chunks.
        
        Args:
            x: Input tensor of shape (batch, sequence, hidden)
            
        Returns:
            Output tensor of same shape as input
        """
        if self._original_ffn is None:
            raise RuntimeError("ChunkedFFN: Original FFN not set!")
        
        # Handle different input shapes
        if x.dim() == 2:
            # (sequence, hidden) - no chunking needed
            return self._original_ffn(x)
        
        batch_size, seq_len = x.shape[:2]
        
        # Don't chunk if sequence is too short
        if seq_len < self.num_chunks * self.min_sequence_per_chunk:
            return self._original_ffn(x)
        
        # Calculate chunk size (ceiling division for even distribution)
        chunk_size = max(1, (seq_len + self.num_chunks - 1) // self.num_chunks)
        
        # Process in chunks
        outputs = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx]
            out = self._original_ffn(chunk)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)


def find_ffn_modules(model: nn.Module, 
                      ffn_names: Tuple[str, ...] = ('ff', 'ffn', 'mlp', 'feed_forward')) -> Dict[str, nn.Module]:
    """
    Find all FFN modules in a model.
    
    Args:
        model: PyTorch model to search
        ffn_names: Tuple of common FFN module name patterns
        
    Returns:
        Dictionary mapping module path to module
    """
    ffn_modules = {}
    
    for name, module in model.named_modules():
        # Check if this looks like an FFN module
        name_lower = name.lower()
        if any(ffn_name in name_lower for ffn_name in ffn_names):
            # Must be a container module with weight parameters (not a leaf)
            if hasattr(module, 'forward') and list(module.parameters()):
                ffn_modules[name] = module
    
    return ffn_modules


def apply_ffn_chunking(model: nn.Module, 
                       num_chunks: int = 8,
                       verbose: bool = True) -> Dict[str, Tuple[nn.Module, Any]]:
    """
    Apply FFN chunking to all FFN modules in a model.
    
    Args:
        model: The diffusion model (typically model.model.diffusion_model)
        num_chunks: Number of sequence chunks (1 = disabled)
        verbose: Print which modules are wrapped
        
    Returns:
        Dictionary of original module info for restoration
    """
    if num_chunks <= 1:
        if verbose:
            print("[FFNChunking] Disabled (num_chunks <= 1)")
        return {}
    
    originals = {}
    wrapped_count = 0
    
    # Get the actual diffusion model if wrapped
    diffusion_model = model
    if hasattr(model, 'model'):
        if hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        elif hasattr(model.model, 'model'):
            diffusion_model = model.model.model
    
    # Find transformer blocks - LTX uses transformer_blocks
    blocks = None
    for attr in ['transformer_blocks', 'blocks', 'layers']:
        if hasattr(diffusion_model, attr):
            blocks = getattr(diffusion_model, attr)
            break
    
    if blocks is None:
        if verbose:
            print("[FFNChunking] No transformer blocks found in model")
        return {}
    
    # Wrap FFN in each block
    for idx, block in enumerate(blocks):
        # LTX uses 'ff' for feedforward
        for ffn_attr in ['ff', 'ffn', 'mlp', 'feed_forward']:
            if hasattr(block, ffn_attr):
                original_ffn = getattr(block, ffn_attr)
                
                # Create chunked wrapper
                chunked = ChunkedFFN(num_chunks=num_chunks)
                chunked.set_original(original_ffn)
                
                # Store original for restoration
                originals[f"block_{idx}_{ffn_attr}"] = (block, ffn_attr, original_ffn)
                
                # Replace with chunked version
                setattr(block, ffn_attr, chunked)
                wrapped_count += 1
                break  # Only wrap one FFN per block
    
    if verbose:
        print(f"[FFNChunking] Wrapped {wrapped_count} FFN modules with {num_chunks} chunks")
        if wrapped_count > 0:
            # Estimate memory savings
            savings_factor = num_chunks
            print(f"[FFNChunking] Estimated peak FFN VRAM reduction: ~{savings_factor}x")
    
    return originals


def restore_ffn(originals: Dict[str, Tuple[nn.Module, str, nn.Module]], 
                verbose: bool = True):
    """
    Restore original FFN modules after sampling.
    
    Args:
        originals: Dictionary from apply_ffn_chunking()
        verbose: Print restoration info
    """
    if not originals:
        return
    
    restored_count = 0
    for key, (block, attr_name, original_ffn) in originals.items():
        setattr(block, attr_name, original_ffn)
        restored_count += 1
    
    if verbose:
        print(f"[FFNChunking] Restored {restored_count} original FFN modules")
