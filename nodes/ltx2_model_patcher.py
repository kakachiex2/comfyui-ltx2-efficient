import torch
import comfy.model_patcher
import comfy.model_base

class LTX2ModelPatcherNode:
    """
    Patches LTX model to handle:
    1. Missing attention_mask argument (REQUIRED for LTX models - no default!)
    2. Context dimension mismatch (7680 -> 4096)
    3. 3D latent tensors from SamplerCustomAdvanced (reshapes to 5D)
    
    IMPORTANT: Use this node BEFORE SamplerCustomAdvanced to patch the model.
    The LTX model's forward() requires attention_mask as a positional argument,
    but ComfyUI doesn't pass it unless it's in the conditioning. This node fixes that.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Connect your latent here to capture shape for 3D->5D reconstruction"}),
                "context_slice_method": ([
                    "Keep Last 4096 (Slot 2 - LTX Connector)", 
                    "Keep First 4096 (Slot 1 - T5)", 
                    "Auto-detect"
                ], {"default": "Keep Last 4096 (Slot 2 - LTX Connector)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "LTX2-Video"
    DESCRIPTION = "Patches LTX model for GGUF/SamplerCustomAdvanced compatibility. Injects the required attention_mask argument and handles context slicing."

    def patch_model(self, model, latent=None, context_slice_method="Keep Last 4096 (Slot 2 - LTX Connector)"):
        model_clone = model.clone()
        
        # Extract latent shape for 3D->5D reconstruction if needed
        latent_shape = None
        if latent is not None:
            samples = latent.get("samples")
            if samples is not None:
                if hasattr(samples, 'shape'):
                    latent_shape = samples.shape
                elif isinstance(samples, tuple) and len(samples) > 0:
                    latent_shape = samples[0].shape
                
                if latent_shape is not None:
                    print(f"[LTX2ModelPatcher] Captured latent shape: {latent_shape}")
        
        # Store config for use in wrapper
        patcher_config = {
            'latent_shape': latent_shape,
            'context_slice_method': context_slice_method,
            'context_logged': False
        }
        
        # Check if we have a diffusion model to patch
        if not hasattr(model_clone.model, 'diffusion_model'):
            print("[LTX2ModelPatcher] Warning: Could not find diffusion_model to patch")
            return (model_clone,)
        
        diffusion_model = model_clone.model.diffusion_model
        
        # Only patch if not already patched
        if getattr(diffusion_model, '_ltx2_patched', False):
            print("[LTX2ModelPatcher] Model already patched, skipping")
            return (model_clone,)
        
        # Save the original forward method
        original_forward = diffusion_model.forward
        
        def patched_forward(x, timestep, context=None, attention_mask=None, frame_rate=25, 
                          transformer_options=None, keyframe_idxs=None, denoise_mask=None, **kwargs):
            """
            Patched forward that:
            1. Always provides attention_mask (even as None)
            2. Handles context slicing (7680 -> 4096)
            3. Handles 3D -> 5D latent reshaping
            """
            if transformer_options is None:
                transformer_options = {}
            
            # Handle context slicing if needed
            if context is not None and hasattr(context, 'shape') and len(context.shape) >= 2:
                dim = context.shape[-1]
                if dim == 7680:
                    slice_method = patcher_config['context_slice_method']
                    if "Last 4096" in slice_method or "Slot 2" in slice_method:
                        context = context[..., -4096:]
                        if not patcher_config['context_logged']:
                            print(f"[LTX2ModelPatcher] Sliced context 7680->4096 (Last 4096 / LTX Connector)")
                            patcher_config['context_logged'] = True
                    elif slice_method == "Auto-detect":
                        gemma_var = context[..., :3584].var().item()
                        first_var = context[..., :4096].var().item()
                        if abs(gemma_var - first_var) > 0.05:
                            context = context[..., -4096:]
                            if not patcher_config['context_logged']:
                                print(f"[LTX2ModelPatcher] Auto-detected Gemma in Slot 1, using Slot 2")
                                patcher_config['context_logged'] = True
                        else:
                            context = context[..., :4096]
                            if not patcher_config['context_logged']:
                                print(f"[LTX2ModelPatcher] Auto-detected LTX in Slot 1")
                                patcher_config['context_logged'] = True
                    else:
                        context = context[..., :4096]
                        if not patcher_config['context_logged']:
                            print(f"[LTX2ModelPatcher] Sliced context 7680->4096 (First 4096 / Slot 1)")
                            patcher_config['context_logged'] = True
            
            # Handle 3D->5D latent reshaping if needed
            if x.ndim == 3:
                stored_shape = patcher_config.get('latent_shape', None)
                if stored_shape is not None and len(stored_shape) == 5:
                    b, c, l = x.shape
                    _, _, f, h, w = stored_shape
                    if l == f * h * w:
                        x = x.view(b, c, f, h, w)
                        print(f"[LTX2ModelPatcher] Reshaped 3D ({b},{c},{l}) -> 5D {x.shape}")
            
            # Call original forward with all positional arguments
            # attention_mask is now always explicitly passed (even if None)
            return original_forward(
                x, timestep, context, 
                attention_mask=attention_mask,
                frame_rate=frame_rate, 
                transformer_options=transformer_options,
                keyframe_idxs=keyframe_idxs, 
                denoise_mask=denoise_mask, 
                **kwargs
            )
        
        # Apply the patch
        diffusion_model.forward = patched_forward
        diffusion_model._ltx2_patched = True
        print("[LTX2ModelPatcher] Successfully patched diffusion_model.forward")
        print("[LTX2ModelPatcher] attention_mask will now default to None if not provided")
        
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "LTX2ModelPatcher": LTX2ModelPatcherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2ModelPatcher": "LTX2 Model Patcher",
}
