import torch

class LTX2ConditioningHelper:
    """
    Helper node to fix conditioning data for LTX Video models.
    
    Addresses:
    1. Missing 'attention_mask' (adds ones mask).
    2. Dimension mismatch (slices 7680 -> 4096 for T5-only models).
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "fix_dimensions": ("BOOLEAN", {"default": True, "label_on": "Fix Dims (7680->4096)", "label_off": "Keep Dims"}),
                "slice_method": (["Keep First 4096", "Keep Last 4096"], {"default": "Keep First 4096"}),
                "add_attention_mask": ("BOOLEAN", {"default": True, "label_on": "Add Mask", "label_off": "No Mask"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "patch_conditioning"
    CATEGORY = "LTX2-Video/utils"

    def patch_conditioning(self, conditioning, fix_dimensions, slice_method, add_attention_mask):
        new_conditioning = []
        
        for t, d in conditioning:
            # Clone dictionary to avoid side effects
            new_dict = d.copy()
            new_tensor = t
            
            # 1. Fix Dimension Mismatch
            if fix_dimensions:
                # Check for 7680 dimension (Gemma+T5 stack)
                if new_tensor.shape[-1] == 7680:
                    print(f"[LTX2ConditioningHelper] Slicing embedding: {new_tensor.shape} -> (..., 4096) using {slice_method}")
                    if slice_method == "Keep Last 4096":
                        new_tensor = new_tensor[..., -4096:]
                    else:
                        new_tensor = new_tensor[..., :4096]
                    
                    if new_tensor.shape[-1] == 4096:
                         print(f"[LTX2ConditioningHelper] Success. If video is noisy, try switching slice_method (T5 must be selected).")
                elif new_tensor.shape[-1] > 4096:
                     print(f"[LTX2ConditioningHelper] ⚠️ Tensor shape {new_tensor.shape} has > 4096 channels. You might need to disable 'Fix Dims' if this is intentional.")
            
            # 2. Add Attention Mask
            if add_attention_mask:
                if "attention_mask" not in new_dict:
                    # Injecting None is safer than creating a tensor mask (ones), 
                    # as explicit shapes can clash with LTX's internal reshaping/broadcasting.
                    # 'None' typically implies "attend to everything" in SDPA.
                    print(f"[LTX2ConditioningHelper] Injecting attention_mask=None for tensor {new_tensor.shape}")
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
