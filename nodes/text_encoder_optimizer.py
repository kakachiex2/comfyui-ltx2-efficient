"""
LTXAV Text Encoder Memory Optimizer

Reduces VRAM usage when encoding text with the massive LTXAVTEModel by:
1. Processing the text_embedding_projection in chunks on CPU
2. Moving only final results to GPU
3. Aggressive memory cleanup between chunks
"""

import torch
import torch.nn.functional as F
import comfy.model_management


def patch_ltxav_text_encoder(clip_model, chunk_size=64, force_cpu=True):
    """
    Patches LTXAVTEModel's encode_token_weights to use chunked CPU processing.
    
    Args:
        clip_model: The CLIP model wrapper from ComfyUI
        chunk_size: Number of sequence positions to process per chunk (default: 64)
        force_cpu: If True, process projection on CPU to save VRAM
        
    Returns:
        bool: True if patching was successful
    """
    # Access the underlying text encoder model
    if not hasattr(clip_model, 'cond_stage_model'):
        return False
        
    te_model = clip_model.cond_stage_model
    
    # Check if this is LTXAVTEModel (has text_embedding_projection)
    if not hasattr(te_model, 'text_embedding_projection'):
        print("[LTX2Pro] Text encoder doesn't have text_embedding_projection, skipping optimization")
        return False
    
    # Already patched?
    if getattr(te_model, '_ltx2_te_patched', False):
        return True
        
    # Store original method
    original_encode = te_model.encode_token_weights
    projection_layer = te_model.text_embedding_projection
    
    def chunked_encode_token_weights(token_weight_pairs):
        """
        Optimized encoding that processes the large projection in chunks.
        """
        # Get tokens for gemma
        token_weight_pairs_gemma = token_weight_pairs["gemma3_12b"]
        
        # Run Gemma encoding (this already uses offloading)
        out, pooled, extra = te_model.gemma3_12b.encode_token_weights(token_weight_pairs_gemma)
        out_device = out.device
        
        # Prepare tensor (normalization step from original)
        import comfy.model_management
        if comfy.model_management.should_use_bf16(te_model.execution_device):
            out = out.to(device=te_model.execution_device, dtype=torch.bfloat16)
        out = out.movedim(1, -1).to(te_model.execution_device)
        out = 8.0 * (out - out.mean(dim=(1, 2), keepdim=True)) / (out.amax(dim=(1, 2), keepdim=True) - out.amin(dim=(1, 2), keepdim=True) + 1e-6)
        out = out.reshape((out.shape[0], out.shape[1], -1))
        
        # Now comes the CHUNKED projection (the memory bottleneck)
        B, seq_len, features = out.shape  # [1, 1024, 188160]
        
        print(f"[LTX2Pro TE] Chunked projection: input shape={out.shape}, chunk_size={chunk_size}")
        
        if force_cpu:
            # Move projection weights to CPU if not already
            weight = projection_layer.weight
            if weight.device.type != 'cpu':
                weight = weight.cpu()
            weight = weight.to(torch.float32)  # CPU compute in fp32 for stability
            
            # Process in chunks
            output_chunks = []
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                
                # Move chunk to CPU
                chunk = out[:, i:end_idx, :].cpu().float()
                
                # Compute projection on CPU
                out_chunk = F.linear(chunk, weight, None)
                
                # Move result back to GPU
                output_chunks.append(out_chunk.to(out_device).to(out.dtype))
                
                # Cleanup
                del chunk
                if i > 0 and i % (chunk_size * 4) == 0:
                    torch.cuda.empty_cache()
            
            # Concatenate results
            projected = torch.cat(output_chunks, dim=1)
            del output_chunks
        else:
            # Standard GPU projection (for reference)
            projected = projection_layer(out)
        
        # Continue with embeddings connectors
        projected = projected.float()
        out_vid = te_model.video_embeddings_connector(projected)[0]
        out_audio = te_model.audio_embeddings_connector(projected)[0]
        out_final = torch.concat((out_vid, out_audio), dim=-1)
        
        # Cleanup
        del out, projected
        torch.cuda.empty_cache()
        
        print(f"[LTX2Pro TE] Chunked encoding complete: output shape={out_final.shape}")
        
        return out_final.to(out_device), pooled
    
    # Apply patch
    te_model.encode_token_weights = chunked_encode_token_weights
    te_model._ltx2_te_patched = True
    te_model._ltx2_te_original_encode = original_encode
    te_model._ltx2_te_chunk_size = chunk_size
    
    print(f"[LTX2Pro] ✓ Patched LTXAVTEModel for chunked projection (chunk_size={chunk_size}, force_cpu={force_cpu})")
    return True


def restore_ltxav_text_encoder(clip_model):
    """
    Restores the original encode_token_weights method.
    """
    if not hasattr(clip_model, 'cond_stage_model'):
        return False
        
    te_model = clip_model.cond_stage_model
    
    if hasattr(te_model, '_ltx2_te_original_encode'):
        te_model.encode_token_weights = te_model._ltx2_te_original_encode
        te_model._ltx2_te_patched = False
        print("[LTX2Pro] Restored original text encoder")
        return True
    
    return False
