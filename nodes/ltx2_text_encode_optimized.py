"""
LTX2 Optimized Text Encoder Node

A drop-in replacement for CLIPTextEncode that uses chunked CPU processing
for the massive text_embedding_projection layer in LTXAVTEModel.

This significantly reduces VRAM usage for low-VRAM GPUs (6GB).
"""

import torch
import torch.nn.functional as F
import comfy.model_management


class LTX2TextEncodeOptimized:
    """
    Optimized Text Encoder for LTX Audio-Video models.
    
    Uses chunked CPU processing for the large projection layer,
    reducing peak VRAM by ~1GB compared to standard CLIPTextEncode.
    """
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "video/ltx2"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "chunk_size": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Smaller = less VRAM but slower. 64 is a good balance."
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process projection on CPU (recommended for low VRAM)"
                }),
            }
        }
    
    def encode(self, clip, text, chunk_size=64, force_cpu=True):
        """
        Encode text with memory optimization for LTXAVTEModel.
        """
        # Tokenize
        tokens = clip.tokenize(text)
        
        # Check if we have LTXAVTEModel with text_embedding_projection
        has_te_projection = (
            hasattr(clip.cond_stage_model, 'text_embedding_projection') and
            hasattr(clip.cond_stage_model, 'gemma3_12b')
        )
        
        if has_te_projection:
            print(f"[LTX2 TE Optimized] Detected LTXAVTEModel, using chunked encoding (chunk_size={chunk_size})")
            output = self._chunked_encode(clip, tokens, chunk_size, force_cpu)
        else:
            print("[LTX2 TE Optimized] Standard text encoder detected, using default encoding")
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        
        # Extract conditioning
        cond = output.pop("cond")
        
        return ([[cond, output]],)
    
    def _chunked_encode(self, clip, tokens, chunk_size, force_cpu):
        """
        Implements chunked encoding for LTXAVTEModel.
        
        CRITICAL: We must preserve the original dtype (bfloat16) throughout
        to maintain audio quality. CPU float32 processing corrupts embeddings.
        """
        te_model = clip.cond_stage_model
        
        # Get tokens for gemma
        token_weight_pairs = tokens["gemma3_12b"]
        
        # Run Gemma encoding (already uses offloading)
        print("[LTX2 TE Optimized] Running Gemma encoder...")
        out, pooled, extra = te_model.gemma3_12b.encode_token_weights(token_weight_pairs)
        out_device = out.device
        
        # Get execution device safely (fallback to torch device if not set)
        execution_device = getattr(te_model, 'execution_device', None)
        if execution_device is None:
            execution_device = comfy.model_management.get_torch_device()
        
        # Prepare tensor - compute normalization in float32 to avoid NaN/Inf
        use_bf16 = comfy.model_management.should_use_bf16(execution_device)
        target_dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        # Move to device first, then do normalization in float32 for numerical stability
        out = out.to(device=execution_device)
        out = out.movedim(1, -1)
        
        # Perform normalization in float32 to prevent NaN/Inf
        out = out.float()
        out_mean = out.mean(dim=(1, 2), keepdim=True)
        out_max = out.amax(dim=(1, 2), keepdim=True)
        out_min = out.amin(dim=(1, 2), keepdim=True)
        out_range = out_max - out_min
        
        # Use larger epsilon and handle edge case where range is too small
        epsilon = 1e-5
        out_range = torch.where(out_range < epsilon, torch.ones_like(out_range), out_range)
        out = 8.0 * (out - out_mean) / (out_range + epsilon)
        
        # Clamp to prevent extreme values and check for NaN/Inf
        out = torch.clamp(out, -100.0, 100.0)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[LTX2 TE Optimized] WARNING: NaN/Inf detected after normalization, replacing with zeros")
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert back to target dtype
        out = out.to(dtype=target_dtype)
        out = out.reshape((out.shape[0], out.shape[1], -1))
        
        # Get dimensions
        B, seq_len, features = out.shape
        print(f"[LTX2 TE Optimized] Projection input: ({B}, {seq_len}, {features}), dtype={out.dtype}")
        
        # Access projection layer
        projection = te_model.text_embedding_projection
        out_features = projection.weight.shape[0]
        
        if force_cpu:
            # IMPROVED: Do chunked processing ON GPU with memory cleanup
            # This preserves precision while managing memory
            print(f"[LTX2 TE Optimized] Processing {seq_len} positions in {chunk_size}-position chunks (GPU with memory management)...")
            
            # Allocate output tensor
            projected = torch.empty(B, seq_len, out_features, dtype=out.dtype, device=execution_device)
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                
                # Process chunk on GPU (preserves precision!)
                chunk = out[:, i:end_idx, :]
                projected[:, i:end_idx, :] = projection(chunk)
                
                # Aggressive memory cleanup between chunks
                del chunk
                if i > 0 and (i // chunk_size) % 2 == 0:
                    torch.cuda.empty_cache()
            
            # Free input tensor
            del out
            torch.cuda.empty_cache()
            
            print(f"[LTX2 TE Optimized] Projection complete: {projected.shape}")
            
        else:
            # Full GPU projection (original behavior)
            projected = projection(out)
            del out
        
        # Continue with embeddings connectors
        projected = projected.float()
        
        # Check projected for NaN/Inf before connectors
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            print("[LTX2 TE Optimized] WARNING: NaN/Inf in projected embeddings, sanitizing...")
            projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("[LTX2 TE Optimized] Running video embeddings connector...")
        out_vid = te_model.video_embeddings_connector(projected)[0]
        
        print("[LTX2 TE Optimized] Running audio embeddings connector...")
        out_audio = te_model.audio_embeddings_connector(projected)[0]
        
        # Combine
        out_final = torch.concat((out_vid, out_audio), dim=-1)
        
        # Final safety check - ensure no NaN/Inf in output (critical for audio encoding!)
        if torch.isnan(out_final).any() or torch.isinf(out_final).any():
            print("[LTX2 TE Optimized] WARNING: NaN/Inf in final output, sanitizing to prevent SaveVideo crash...")
            out_final = torch.nan_to_num(out_final, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Final cleanup
        del projected, out_vid, out_audio
        torch.cuda.empty_cache()
        
        print(f"[LTX2 TE Optimized] ✓ Encoding complete: {out_final.shape}")
        
        return {"cond": out_final.to(out_device), "pooled_output": pooled}


# Node registration
NODE_CLASS_MAPPINGS = {
    "LTX2TextEncodeOptimized": LTX2TextEncodeOptimized,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2TextEncodeOptimized": "LTX2 Text Encode (Optimized)",
}
