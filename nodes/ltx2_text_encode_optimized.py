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
        Implements chunked CPU encoding for LTXAVTEModel.
        """
        te_model = clip.cond_stage_model
        
        # Get tokens for gemma
        token_weight_pairs = tokens["gemma3_12b"]
        
        # Run Gemma encoding (already uses offloading)
        print("[LTX2 TE Optimized] Running Gemma encoder...")
        out, pooled, extra = te_model.gemma3_12b.encode_token_weights(token_weight_pairs)
        out_device = out.device
        
        # Prepare tensor (normalization step)
        if comfy.model_management.should_use_bf16(te_model.execution_device):
            out = out.to(device=te_model.execution_device, dtype=torch.bfloat16)
        
        out = out.movedim(1, -1).to(te_model.execution_device)
        out = 8.0 * (out - out.mean(dim=(1, 2), keepdim=True)) / (
            out.amax(dim=(1, 2), keepdim=True) - out.amin(dim=(1, 2), keepdim=True) + 1e-6
        )
        out = out.reshape((out.shape[0], out.shape[1], -1))
        
        # Get dimensions
        B, seq_len, features = out.shape
        print(f"[LTX2 TE Optimized] Projection input: ({B}, {seq_len}, {features})")
        
        # Access projection layer
        projection = te_model.text_embedding_projection
        
        if force_cpu:
            # Get weight and move to CPU
            weight = projection.weight
            weight_cpu = weight.detach().cpu().to(torch.float32)
            
            print(f"[LTX2 TE Optimized] Processing {seq_len} positions in chunks of {chunk_size}...")
            
            # Process in chunks
            output_chunks = []
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                
                # Move chunk to CPU
                chunk = out[:, i:end_idx, :].detach().cpu().float()
                
                # Compute projection on CPU
                out_chunk = F.linear(chunk, weight_cpu, None)
                
                # Move result back to original device
                output_chunks.append(out_chunk.to(device=out_device, dtype=out.dtype))
                
                # Cleanup
                del chunk
                
                # Periodic cache clear
                if (i // chunk_size) % 4 == 0 and i > 0:
                    torch.cuda.empty_cache()
                    
            # Concatenate
            projected = torch.cat(output_chunks, dim=1)
            del output_chunks, weight_cpu
            
            print(f"[LTX2 TE Optimized] Projection complete: ({projected.shape})")
            
        else:
            # GPU projection
            projected = projection(out)
        
        # Cleanup after projection
        del out
        torch.cuda.empty_cache()
        
        # Continue with embeddings connectors
        projected = projected.float()
        
        print("[LTX2 TE Optimized] Running video embeddings connector...")
        out_vid = te_model.video_embeddings_connector(projected)[0]
        
        print("[LTX2 TE Optimized] Running audio embeddings connector...")
        out_audio = te_model.audio_embeddings_connector(projected)[0]
        
        # Combine
        out_final = torch.concat((out_vid, out_audio), dim=-1)
        
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
