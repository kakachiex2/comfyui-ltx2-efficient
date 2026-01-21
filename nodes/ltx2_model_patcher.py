import torch
import comfy.model_patcher
import comfy.model_base

class LTX2ModelPatcherNode:
    """
    Patches LTX model to handle:
    1. Missing attention_mask argument
    2. Context dimension mismatch (7680 -> 4096)
    3. 3D latent tensors from SamplerCustomAdvanced (reshapes to 5D)
    
    IMPORTANT: This node stores the original latent shape for proper 3D->5D reconstruction.
    Connect this AFTER your latent generation node to capture the correct shape.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Connect your latent here to capture shape for 3D->5D reconstruction"}),
                "context_slice_method": (["Keep First 4096 (T5)", "Keep Last 4096 (Gemma)", "Auto-detect"], {"default": "Keep First 4096 (T5)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "LTX2-Video"
    DESCRIPTION = "Patches LTX model for GGUF compatibility. Connect latent input to enable 3D->5D reconstruction for SamplerCustomAdvanced."

    def patch_model(self, model, latent=None, context_slice_method="Keep First 4096 (T5)"):
        model_clone = model.clone()
        
        # Extract latent shape for 3D->5D reconstruction
        latent_shape = None
        if latent is not None:
            samples = latent.get("samples")
            if samples is not None:
                # Handle various latent formats
                if hasattr(samples, 'shape'):
                    latent_shape = samples.shape
                elif isinstance(samples, tuple) and len(samples) > 0:
                    latent_shape = samples[0].shape
                
                if latent_shape is not None:
                    print(f"[LTX2ModelPatcher] Captured latent shape: {latent_shape}")
        
        # Access the underlying diffusion model
        if hasattr(model_clone.model, "diffusion_model"):
            diffusion_model = model_clone.model.diffusion_model
        else:
            print("[LTX2ModelPatcher] Warning: Could not find diffusion_model.")
            return (model_clone,)

        self._patch_forward(diffusion_model, latent_shape, context_slice_method)
        
        return (model_clone,)

    def _patch_forward(self, diffusion_model, latent_shape, context_slice_method):
        original_forward = diffusion_model.forward
        if getattr(diffusion_model, "_ltx2_patched", False):
            # Already patched, just update stored shape if provided
            if latent_shape is not None:
                diffusion_model._ltx2_latent_shape = latent_shape
            return

        # Store latent shape on the model for access during forward
        diffusion_model._ltx2_latent_shape = latent_shape
        diffusion_model._ltx2_context_slice_method = context_slice_method

        def ltx2_patched_forward(x, sigma, attention_mask=None, **kwargs):
            stored_shape = getattr(diffusion_model, '_ltx2_latent_shape', None)
            slice_method = getattr(diffusion_model, '_ltx2_context_slice_method', 'Keep First 4096 (T5)')
            
            # 1. Inject attention_mask if missing
            if attention_mask is None:
                attention_mask = None  # Explicit None is fine for SDPA

            # 2. Fix Context Dimensions (7680 -> 4096)
            if "context" in kwargs:
                context = kwargs["context"]
                if hasattr(context, 'shape') and len(context.shape) >= 2:
                    dim = context.shape[-1]
                    if dim == 7680:
                        if slice_method == "Keep Last 4096 (Gemma)":
                            kwargs["context"] = context[..., -4096:]
                            print(f"[LTX2ModelPatcher] Sliced context 7680->4096 (Last 4096 / Gemma)")
                        elif slice_method == "Auto-detect":
                            # Auto-detect: Check variance of first vs last - T5 typically has different distribution
                            first_var = context[..., :4096].var().item()
                            last_var = context[..., -4096:].var().item()
                            # T5 embeddings typically have lower variance than Gemma
                            if first_var < last_var:
                                kwargs["context"] = context[..., :4096]
                                print(f"[LTX2ModelPatcher] Auto-detected T5 in first 4096 (var={first_var:.4f} < {last_var:.4f})")
                            else:
                                kwargs["context"] = context[..., -4096:]
                                print(f"[LTX2ModelPatcher] Auto-detected T5 in last 4096 (var={last_var:.4f} < {first_var:.4f})")
                        else:
                            # Default: Keep First 4096 (T5)
                            kwargs["context"] = context[..., :4096]
                            # Only print once per run
                            if not getattr(diffusion_model, '_context_slice_logged', False):
                                print(f"[LTX2ModelPatcher] Sliced context 7680->4096 (First 4096 / T5)")
                                diffusion_model._context_slice_logged = True
                    elif dim > 4096 and dim != 4096:
                        print(f"[LTX2ModelPatcher] ⚠️ Warning: Context dimension {dim} > 4096. May cause issues.")

            # 3. Handle 3D Latents (B, C, L) -> 5D (B, C, F, H, W)
            if x.ndim == 3:
                b, c, l = x.shape
                reshaped = False
                
                # Method 1: Use stored latent shape
                if stored_shape is not None and len(stored_shape) == 5:
                    _, _, f, h, w = stored_shape
                    expected_l = f * h * w
                    if l == expected_l:
                        x = x.view(b, c, f, h, w)
                        reshaped = True
                        if not getattr(diffusion_model, '_reshape_logged', False):
                            print(f"[LTX2ModelPatcher] Reshaped 3D ({b},{c},{l}) -> 5D {x.shape} using stored shape")
                            diffusion_model._reshape_logged = True
                
                # Method 2: Try to infer from model's patchifier config
                if not reshaped and hasattr(diffusion_model, 'patchifier'):
                    try:
                        # LTX patchifier has patch size info
                        patchifier = diffusion_model.patchifier
                        if hasattr(patchifier, 'patch_size'):
                            patch_t, patch_h, patch_w = patchifier.patch_size
                            # Common LTX latent: 128 channels, 8x divisible H/W
                            # Try common video resolutions
                            for test_dims in [
                                (4, 8, 12),   # 4 frames, 256x384
                                (4, 12, 8),   # 4 frames, 384x256
                                (4, 8, 8),    # 4 frames, 256x256
                                (8, 8, 12),   # 8 frames, 256x384
                                (1, 8, 12),   # 1 frame, 256x384
                            ]:
                                f, h, w = test_dims
                                if f * h * w == l:
                                    x = x.view(b, c, f, h, w)
                                    reshaped = True
                                    print(f"[LTX2ModelPatcher] Inferred 3D -> 5D: {x.shape}")
                                    break
                    except Exception as e:
                        pass
                
                # Method 3: Extract from transformer_options
                if not reshaped:
                    transformer_options = kwargs.get("transformer_options", {})
                    if "original_shape" in transformer_options:
                        orig = transformer_options["original_shape"]
                        if len(orig) == 5:
                            _, _, f, h, w = orig
                            if f * h * w == l:
                                x = x.view(b, c, f, h, w)
                                reshaped = True
                                print(f"[LTX2ModelPatcher] Used transformer_options shape: {x.shape}")
                
                # Method 4: Fallback - try to find valid factorization
                if not reshaped:
                    # Find factors of l that could represent F*H*W
                    # Prefer small F (frames), medium H/W
                    for f in [1, 2, 4, 8, 16]:
                        if l % f == 0:
                            hw = l // f
                            # Try to find h, w such that h*w = hw
                            import math
                            sqrt_hw = int(math.sqrt(hw))
                            for h in range(sqrt_hw, 0, -1):
                                if hw % h == 0:
                                    w = hw // h
                                    # Prefer aspect ratios between 0.5 and 2.0
                                    if 0.5 <= h/w <= 2.0:
                                        x = x.view(b, c, f, h, w)
                                        reshaped = True
                                        print(f"[LTX2ModelPatcher] ⚠️ Fallback reshape 3D -> 5D: {x.shape} (may be incorrect)")
                                        break
                            if reshaped:
                                break
                
                if not reshaped:
                    print(f"[LTX2ModelPatcher] ❌ ERROR: Cannot reshape 3D tensor ({b},{c},{l}) to 5D!")
                    print(f"[LTX2ModelPatcher] ❌ Connect a latent input to this node to provide the correct shape.")
                    raise ValueError(f"LTX2ModelPatcher: Cannot reshape 3D latent ({b},{c},{l}) to 5D. "
                                   f"Please connect a latent input to capture the original shape.")
            
            return original_forward(x, sigma, attention_mask=attention_mask, **kwargs)

        diffusion_model.forward = ltx2_patched_forward
        diffusion_model._ltx2_patched = True
        print("[LTX2ModelPatcher] Model patched successfully.")

NODE_CLASS_MAPPINGS = {
    "LTX2ModelPatcher": LTX2ModelPatcherNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2ModelPatcher": "LTX2 Model Patcher"
}
