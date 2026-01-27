import torch

class LTX2ConditioningHelper:
    """
    Helper node to fix conditioning data for LTX Video models.
    
    Addresses:
    1. Missing 'attention_mask' (adds ones mask).
    2. Dimension mismatch (slices 7680 -> 4096 for LTX models).
    3. Different dual-encoder configurations.
    
    COMMON CONFIGURATIONS:
    - Gemma (3584) + LTX Embeddings Connector (4096) = 7680 → Use "Keep Last 4096 (Slot 2)"
    - T5-XXL (4096) + Gemma (3584) = 7680 → Use "Keep First 4096 (Slot 1)"
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
                    "Keep Last 4096 (Slot 2 - LTX Connector)",
                    "Keep First 4096 (Slot 1 - T5)",
                    "Auto-detect LTX position",
                ], {"default": "Keep Last 4096 (Slot 2 - LTX Connector)"}),
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
    DESCRIPTION = "Fixes conditioning for LTX models. For Gemma+LTX Connector, use 'Keep Last 4096 (Slot 2)'."

    def _detect_ltx_position(self, tensor):
        """
        Auto-detect which slot contains the LTX-compatible embeddings.
        LTX embeddings connector typically has specific statistical properties.
        
        Returns: "first" or "last"
        """
        if tensor.shape[-1] != 7680:
            return "last"  # Default to slot 2 for LTX Connector
        
        first_chunk = tensor[..., :4096]
        last_chunk = tensor[..., -4096:]
        
        # Method 1: Variance comparison
        # LTX embeddings connector and T5 have different characteristics than Gemma
        first_var = first_chunk.var().item()
        last_var = last_chunk.var().item()
        
        # Method 2: Check mean absolute value
        first_mean_abs = first_chunk.abs().mean().item()
        last_mean_abs = last_chunk.abs().mean().item()
        
        # Gemma typically has different distribution
        # If first 3584 dims have very different stats than next portion, Gemma is first
        gemma_portion = tensor[..., :3584]
        gemma_var = gemma_portion.var().item()
        
        # If the variance of first 3584 differs significantly from full first 4096,
        # it suggests Gemma is in slot 1
        if abs(gemma_var - first_var) > 0.1 * first_var:
            # Gemma likely in slot 1, so LTX is in slot 2
            return "last"
        
        # Compare variance - LTX/T5 embeddings often have more uniform distribution
        if last_var < first_var:
            return "last"
        else:
            return "first"

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
                    if slice_method == "Auto-detect LTX position":
                        position = self._detect_ltx_position(new_tensor)
                        if position == "first":
                            new_tensor = new_tensor[..., :4096]
                            method_used = "First 4096 (auto-detected LTX in Slot 1)"
                        else:
                            new_tensor = new_tensor[..., -4096:]
                            method_used = "Last 4096 (auto-detected LTX in Slot 2)"
                    elif slice_method == "Keep First 4096 (Slot 1 - T5)":
                        new_tensor = new_tensor[..., :4096]
                        method_used = "First 4096 (Slot 1)"
                    else:
                        # Default: Keep Last 4096 (Slot 2 - LTX Connector)
                        new_tensor = new_tensor[..., -4096:]
                        method_used = "Last 4096 (Slot 2 - LTX Connector)"
                    
                    print(f"[LTX2ConditioningHelper] Sliced {original_dim} -> 4096 using {method_used}")
                    print(f"[LTX2ConditioningHelper] ✓ For Gemma+LTX Connector: use 'Keep Last 4096 (Slot 2)'")
                    
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
