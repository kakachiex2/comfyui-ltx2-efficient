import torch

class LTX2ConditioningHelper:
    """
    Helper node to fix conditioning data for LTX Video models.
    
    Addresses:
    1. Missing 'attention_mask' (adds ones mask).
    2. Dimension mismatch (slices 7680 -> 4096 for T5-only models).
    3. Auto-detection of T5 vs Gemma position in dual-encoder output.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "fix_dimensions": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "Fix Dims (7680->4096)", 
                    "label_off": "Keep Dims"
                }),
                "slice_method": ([
                    "Keep First 4096 (T5 default)",
                    "Keep Last 4096 (Gemma position)", 
                    "Auto-detect T5 position",
                    "Middle 4096"
                ], {"default": "Keep First 4096 (T5 default)"}),
                "add_attention_mask": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "Add Mask", 
                    "label_off": "No Mask"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "patch_conditioning"
    CATEGORY = "LTX2-Video/utils"
    DESCRIPTION = "Fixes conditioning for LTX models. Use 'Auto-detect' if unsure which CLIP encoder is T5."

    def _detect_t5_position(self, tensor):
        """
        Auto-detect T5 embedding position based on statistical properties.
        T5-XXL embeddings typically have different variance/distribution than Gemma.
        
        Returns: "first" or "last"
        """
        if tensor.shape[-1] != 7680:
            return "first"  # Default
        
        first_chunk = tensor[..., :4096]
        last_chunk = tensor[..., -4096:]
        
        # Method 1: Variance comparison
        # T5 embeddings often have lower overall variance
        first_var = first_chunk.var().item()
        last_var = last_chunk.var().item()
        
        # Method 2: Check for padding patterns
        # T5 may have more zeros in padding positions
        first_zero_ratio = (first_chunk.abs() < 1e-6).float().mean().item()
        last_zero_ratio = (last_chunk.abs() < 1e-6).float().mean().item()
        
        # Combined heuristic
        # T5 typically: lower variance, possibly more padding
        first_score = first_var - first_zero_ratio * 0.1
        last_score = last_var - last_zero_ratio * 0.1
        
        if first_score < last_score:
            return "first"
        else:
            return "last"

    def patch_conditioning(self, conditioning, fix_dimensions, slice_method, add_attention_mask):
        new_conditioning = []
        
        for t, d in conditioning:
            # Clone dictionary to avoid side effects
            new_dict = d.copy()
            new_tensor = t
            
            # 1. Fix Dimension Mismatch
            if fix_dimensions:
                original_dim = new_tensor.shape[-1]
                
                if original_dim == 7680:
                    # Determine slice method
                    if slice_method == "Auto-detect T5 position":
                        position = self._detect_t5_position(new_tensor)
                        if position == "first":
                            new_tensor = new_tensor[..., :4096]
                            method_used = "First 4096 (auto-detected T5)"
                        else:
                            new_tensor = new_tensor[..., -4096:]
                            method_used = "Last 4096 (auto-detected T5)"
                    elif slice_method == "Keep Last 4096 (Gemma position)":
                        new_tensor = new_tensor[..., -4096:]
                        method_used = "Last 4096"
                    elif slice_method == "Middle 4096":
                        # Take middle portion (skip first and last 1792 dims)
                        start = (7680 - 4096) // 2  # 1792
                        new_tensor = new_tensor[..., start:start+4096]
                        method_used = "Middle 4096"
                    else:
                        # Default: Keep First 4096 (T5 default)
                        new_tensor = new_tensor[..., :4096]
                        method_used = "First 4096 (T5 default)"
                    
                    print(f"[LTX2ConditioningHelper] Sliced {original_dim} -> 4096 using {method_used}")
                    
                elif original_dim > 4096:
                    print(f"[LTX2ConditioningHelper] ⚠️ Dimension {original_dim} > 4096. Consider manual adjustment.")
                elif original_dim == 4096:
                    print(f"[LTX2ConditioningHelper] ✓ Dimension already 4096, no slicing needed.")
                else:
                    print(f"[LTX2ConditioningHelper] ⚠️ WARNING: Dimension {original_dim} < 4096! This may cause errors.")
            
            # 2. Add Attention Mask
            if add_attention_mask:
                if "attention_mask" not in new_dict:
                    # Injecting None is safer than creating a tensor mask (ones), 
                    # as explicit shapes can clash with LTX's internal reshaping/broadcasting.
                    # 'None' typically implies "attend to everything" in SDPA.
                    print(f"[LTX2ConditioningHelper] Injected attention_mask=None for tensor {new_tensor.shape}")
                    new_dict["attention_mask"] = None
                
            new_conditioning.append([new_tensor, new_dict])
            
        return (new_conditioning,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "LTX2ConditioningHelper": LTX2ConditioningHelper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2ConditioningHelper": "LTX2 Conditioning Helper"
}
